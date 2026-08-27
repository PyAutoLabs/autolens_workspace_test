# XLA CPU: `FftThunk` deadlocks the intra-op Eigen pool by re-entering it via ducc0

## Summary

On the CPU backend, `xla::cpu::FftThunk::Execute` can run **on** an intra-op Eigen
pool worker, and it passes that same pool to ducc0 as an
`Eigen::ThreadPoolInterface*`. ducc0 then fans the transform out **into the pool it
is already running on** and blocks on a latch waiting for sub-tasks that need a free
worker.

When every worker in the pool is doing this at once, no worker is left to run any of
their sub-tasks and the pool is wedged permanently. The process makes no further
progress: `jax.block_until_ready` never returns, and every thread sits in
`futex_do_wait` with nothing spinning.

The reproducer below hangs **8 times out of 8** on a 4-CPU machine, and passes 8/8
with `XLA_FLAGS=--xla_cpu_multi_thread_eigen=false` (Fisher exact two-sided
p = 0.000155).

## Environment

| | |
|---|---|
| jax / jaxlib | 0.11.1 |
| Python | 3.12.3 (also seen on 3.13) |
| Platform | Linux x86_64, glibc 2.39 |
| CPU | Intel Xeon @ 2.80GHz, 4 logical CPUs |
| Backend | CPU (`CpuDevice(id=0)`) |

Also reproduced on GitHub-hosted `ubuntu-latest` runners (4 CPUs), on both Python
3.12 and 3.13, over many runs.

## Reproducer

Self-contained; jax and numpy only. `faulthandler` is only there to turn the hang
into a printed traceback instead of an indefinite wait.

```python
import os, faulthandler
faulthandler.dump_traceback_later(150, exit=True)   # turns the hang into evidence

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

N, S = 48, 512                 # N independent FFT convolutions, S x S each
M = (S * S) // 2
vals = jax.random.normal(jax.random.key(0), (N, M), dtype=jnp.float64)
psfs = jax.random.normal(jax.random.key(1), (N, S, S), dtype=jnp.float64)
idx = jnp.arange(M)

@jax.jit
def f(vals, psfs):
    total = 0.0
    flat = jnp.zeros((S * S,), dtype=jnp.float64)
    for i in range(N):
        grid = flat.at[idx].set(vals[i]).reshape(S, S)      # scatter: see note (1)
        total = total + jnp.fft.irfft2(
            jnp.fft.rfft2(grid) * jnp.fft.rfft2(psfs[i]), s=(S, S)
        ).sum()
    return total

jax.block_until_ready(f(vals, psfs))
for r in range(20):            # it is a race; one call often survives, a short loop does not
    jax.block_until_ready(f(vals, psfs))
    print(f"iter {r}", flush=True)
print("completed without hanging")
```

Two properties of the graph are load-bearing. Each was established by removing it and
watching the hang disappear:

1. **A scatter feeding each FFT.** Without it XLA fuses the
   `rfft2`/multiply/`irfft2` chain into a `YnnFusionThunk`; ducc0 then runs the
   transform inline, never takes the latch, and nothing deadlocks. The scatter keeps
   the FFT a standalone `FftThunk`.
2. **Transforms large enough for ducc0 to fan out.** At 180x180 ducc0 ran the
   transform inline on this machine and the script completed; at 512x512 it fans out
   and the deadlock appears. That threshold is ducc0's own, so it is likely
   machine-dependent — raise `S` if the script completes for you.

The hang is a race, so it needs a few iterations rather than a single call.

## Observed behaviour

All four `tf_XLAEigen` workers, in every dump taken (11 of 11 in CI, plus locally):

```
#0  syscall
#1  absl::synchronization_internal::FutexWaiter::WaitUntil
#4  absl::CondVar::WaitCommon
#5  ducc0::detail_threading::latch::wait()
#6  ducc0::detail_threading::Distribution::thread_map(...)
#7  ducc0::detail_threading::Distribution::execParallel(...)
#8  ducc0::detail_threading::execParallel(...)
#9  ducc0::detail_fft::general_r2c<double>(...)
#11 ducc0::google::r2c<double>(..., Eigen::ThreadPoolInterface*)
#12 xla::cpu::FftThunk::Execute(xla::cpu::Thunk::ExecuteParams const&)
#13 xla::cpu::ThunkExecutor::TracedExecute(...)
#15 xla::cpu::ThunkExecutor::SplitReadyQueue<...>::{lambda()#1}::operator()()
#16 Eigen::ThreadPoolTempl<tsl::thread::EigenEnvironment>::WorkerLoop(int)
#18 tsl::(anonymous namespace)::PThread::ThreadFn(void*)
```

Frames **16** and **11** together are the defect: the thunk is executing on a pool
worker (16) and submits work back to that same pool (11), then blocks (5).

Some workers arrive at `FftThunk::Execute` by a second path, resuming the executor
inline from a parallel kernel's completion on the same worker:

```
#17 tsl::AsyncValue::EnqueueWaiter<ThunkExecutor::Execute lambda#2>::Node::RunWaiterAndDeleteWaiterNode()
#18 tsl::CountDownAsyncValueRef<tsl::Chain>::CountDown(...)
#19 xla::cpu::Worker::Parallelize<xla::cpu::Kernel::Task<true>>(...)
```

This second path is **not** required — the reproducer deadlocks with zero such
frames. It is simply another way a worker ends up running an FFT thunk.

Meanwhile the main thread waits forever for the result:

```
absl::Notification::WaitForNotificationWithTimeout
jax::BlockUntilReadyWithCancel(tsl::Future<void>&)
jax::AwaitBuffersReady
jax::PyArray::BlockUntilReady
  try_to_block        (jax/_src/api.py:2764)
  block_until_ready   (jax/_src/api.py:2781)
```

Every thread in the process is in `futex_do_wait`. Nothing is spinning, and CPU usage
is zero — the pool is deadlocked, not merely oversubscribed or slow.

## Analysis

Reading the stack, a pool worker calls into ducc0 with the very pool it is running
on, and ducc0's `execParallel` blocks the caller on a latch until its sub-tasks
complete. Those sub-tasks are queued to the same pool. With `P` workers and `P`
concurrent FFT thunks, all `P` workers are blocked in `latch::wait()` and none can
dequeue the sub-tasks, so the latch is never satisfied.

Consistent with that reading:

- **A pool of one cannot deadlock.** Pinning the process to a single CPU makes ducc0
  choose `nthreads=1` and run the transform inline, so no latch is taken. Measured on
  our production workload in CI: pool of 4 hung 5/6, pool of 1 passed 6/6.
- **The rate depends on how many FFT thunks are concurrently in flight**, which is
  why it is intermittent and why it hits graphs with many independent FFT
  convolutions hardest.

## What this is *not*

Ruled out by measurement, in case they look plausible:

- **Not CPU-count / cgroup-quota oversubscription.** On our runners there is no CFS
  quota at all (`/sys/fs/cgroup/cpu.max` absent), and `os.cpu_count()`,
  `sched_getaffinity`, `/proc/cpuinfo` and `cpuset.cpus.effective` all agree at 4. The
  pool is correctly sized for the machine and still deadlocks.
- **Not the persistent compilation cache** (6/6 hangs with it disabled).
- **Not a jax/jaxlib version change** (byte-identical 0.11.1 stack on a day it hung
  3/3 on both Python legs).
- **Not Python 3.12 vs 3.13** (both hang).
- **Not a compile stall.** Compilation completes ~12-18s before the hang; the failure
  is in execution, materialising the result.

## Impact and workaround

`XLA_FLAGS=--xla_cpu_multi_thread_eigen=false` avoids it reliably, and is what we
currently ship. On the reproducer it is 8/8 pass against 0/8, completing in 3-4s.
The cost on our real workload is roughly 15%, since it disables multithreaded Eigen
for every XLA CPU computation, not just the FFTs.

Pinning to a single CPU also avoids it, at a comparable cost, so pool sizing is not a
cheaper mitigation.

## Possible directions

Offered tentatively — you will know the runtime far better than we do:

- Have `FftThunk` detect that it is already executing on the intra-op pool and pass
  ducc0 a single-threaded pool (or `nthreads=1`) in that case, so the transform runs
  inline rather than re-entering.
- Or make the ducc0 call site not block a pool worker — completing asynchronously
  rather than waiting on a latch held by tasks queued behind it.

Happy to test any patch or diagnostic build against both the reproducer and the real
workload.
