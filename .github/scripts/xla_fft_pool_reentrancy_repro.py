"""
Standalone reproducer: XLA CPU's FftThunk deadlocks by re-entering its own pool.

Reproduces the hang in PyAutoFit#1530 with no PyAuto import at all -- jax and
numpy only -- so it can go upstream as-is. The workaround it explains is
XLA_FLAGS=--xla_cpu_multi_thread_eigen=false, carried in both test workspaces'
smoke and release profiles (PyAutoFit#1528).

    python3 .github/scripts/xla_fft_pool_reentrancy_repro.py

Hangs at `jax.block_until_ready` on jax/jaxlib 0.11.1, 4 CPUs. Nothing in CI
runs it.

THE BUG
-------
`xla::cpu::FftThunk::Execute` runs ON an Eigen intra-op pool worker and hands
ducc0 that same pool as an `Eigen::ThreadPoolInterface*`. ducc0 fans the
transform out into the pool it is already running on and blocks on a latch
waiting for sub-tasks that need a free worker. Once every worker is doing this,
no worker is left to run any of their sub-tasks and the pool is wedged forever.

The captured stack, identical in this script, in the real workspace script run
locally, and in 11 of 11 CI dumps (run 33099502356, both Python legs):

    #5  ducc0::detail_threading::latch::wait()
    #8  ducc0::detail_threading::execParallel(...)
    #11 ducc0::google::r2c<double>(..., Eigen::ThreadPoolInterface*)
    #12 xla::cpu::FftThunk::Execute(...)
    #15 xla::cpu::ThunkExecutor::SplitReadyQueue<...>
    #16 Eigen::ThreadPoolTempl<tsl::thread::EigenEnvironment>::WorkerLoop(int)

All four `tf_XLAEigen` workers sit there; the main thread sits in
`jax::PyArray::BlockUntilReady` -> `absl::Notification::WaitForNotification...`;
every thread in the process is in `futex_do_wait`. Nothing spins, which is why
a month of wall-clock evidence read as "no progress" rather than "slow".

Verify while it is hung:

    python3 .github/scripts/xla_fft_pool_reentrancy_repro.py &
    gdb -p $! --batch -ex 'set pagination off' -ex 'thread apply all bt 30' \
      | grep -c 'ducc0::detail_threading::latch::wait'      # -> 4

THE TWO INGREDIENTS, AND WHY EACH IS NEEDED
-------------------------------------------
Both are required; each was found by removing it and watching the hang go away.

1. A SCATTER FEEDING EACH FFT. Without it XLA fuses the rfft2/multiply/irfft2
   chain into a `YnnFusionThunk`, ducc0 runs the transform INLINE, and no latch
   is ever taken -- no latch, no deadlock. The real graph's HLO reads
   `fft(%wrapped_scatter.423)` for every one of its 282 fft ops, because
   autolens scatters a masked 1-D array into a 2-D grid before convolving.
   The scatter is what keeps the FFT a standalone FftThunk.

2. TRANSFORMS BIG ENOUGH THAT ducc0 FANS OUT. At 180x180 (the real graph's
   size) ducc0 runs inline here and the script completes; at 512x512 it fans
   out and the deadlock appears. ducc0's own work threshold decides this, so
   the size that reproduces is a property of the machine, not a constant --
   raise S if it does not hang for you.

Dead ends, recorded so nobody re-runs them:

  - A few LARGE FFTs with no scatter (N=8..32, S=512..2048). No fan-out, and
    for the reason in (1): they became YnnFusionThunks.
  - MANY TINY FFTs (N=64..1400, S=16..180), with and without the scatter, and
    looped 500x per process for ~1500 executions. Below ducc0's threshold.
  - Shrinking the pool with taskset to 2 or 3 CPUs, on the theory that fewer
    workers need fewer simultaneous FFTs to wedge. 8 trials, no hang.
  - `Worker::Parallelize` / `CountDownAsyncValueRef` /
    `RunWaiterAndDeleteWaiterNode` frames appear in 2 of CI's 4 wedged workers
    and were at first taken for a necessary ingredient. They are NOT: this
    script deadlocks with zero such frames, as does the real script locally.
    They are one of two ways a worker happens to arrive at the FFT thunk.

A pool of ONE cannot deadlock: ducc0 then gets nthreads=1 and runs inline, so
there is no latch to wait on. Confirmed on the real script in CI run
33103725546 -- pool of 4 hung 5/6, pool of 1 passed 6/6.
"""

import os
import time
import faulthandler

faulthandler.dump_traceback_later(int(os.environ.get("DEADLINE", "180")), exit=True)

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

N = int(os.environ.get("N", "48"))  # independent convolution chains
S = int(os.environ.get("S", "512"))  # image side; raise it if this does not hang
R = int(os.environ.get("R", "20"))  # repeats, since the wedge is a race
M = (S * S) // 2  # "masked" pixel count, as a circular mask would give

vals = jax.random.normal(jax.random.key(0), (N, M), dtype=jnp.float64)
psfs = jax.random.normal(jax.random.key(1), (N, S, S), dtype=jnp.float64)
idx = jnp.arange(M)


@jax.jit
def f(vals, psfs):
    """N independent FFT convolutions, each fed by a scatter. See ingredient (1)."""
    total = 0.0
    flat = jnp.zeros((S * S,), dtype=jnp.float64)
    for i in range(N):
        grid = flat.at[idx].set(vals[i]).reshape(S, S)
        total = total + jnp.fft.irfft2(
            jnp.fft.rfft2(grid) * jnp.fft.rfft2(psfs[i]), s=(S, S)
        ).sum()
    return total


def main():
    print(f"jax {jax.__version__} cpus={os.cpu_count()} N={N} S={S} R={R}", flush=True)
    jax.block_until_ready(f(vals, psfs))
    print("warm", flush=True)
    t = time.time()
    for r in range(R):
        jax.block_until_ready(f(vals, psfs))
        print(f"  iter {r} t={time.time() - t:.0f}s", flush=True)
    print(f"ALL {R} ITERATIONS OK in {time.time() - t:.0f}s -- did not reproduce", flush=True)
    faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    main()
