"""
Standalone reproducer for XLA CPU's FftThunk re-entering its own Eigen pool.

Nothing in CI runs this. It exists to be handed upstream, and to be edited by
whoever closes the last gap, so it says plainly what it does and does not yet
show. See PyAutoFit#1530; the workaround it explains is
XLA_FLAGS=--xla_cpu_multi_thread_eigen=false in both test workspaces' smoke and
release profiles (PyAutoFit#1528).

WHAT THE CI STACKS SHOW
-----------------------
On a hung `imaging/jax_likelihood/mge_group.py`, all four `tf_XLAEigen`
workers sit here -- 11 of 11 dumps, both Python legs, run 33099502356:

    #5  ducc0::detail_threading::latch::wait()
    #8  ducc0::detail_threading::execParallel(...)
    #11 ducc0::google::r2c<double>(..., Eigen::ThreadPoolInterface*)
    #12 xla::cpu::FftThunk::Execute(...)
    #16 Eigen::ThreadPoolTempl<tsl::thread::EigenEnvironment>::WorkerLoop(int)

Frames 16 and 11 are the bug: FftThunk runs ON a pool worker and hands ducc0
that same pool, so ducc0 fans the FFT back into the pool it is already running
on and blocks waiting for sub-tasks that need a free worker.

Two of the four workers reach FftThunk by a second, deeper path:

    #19 xla::cpu::Worker::Parallelize<xla::cpu::Kernel::Task<true>>(...)
    #18 tsl::CountDownAsyncValueRef<tsl::Chain>::CountDown(...)
    #17 tsl::AsyncValue::EnqueueWaiter<ThunkExecutor lambda#2>::RunWaiter...()
    #13 xla::cpu::FftThunk::Execute(...)

A PARALLELISED kernel finishes; its countdown runs the executor's waiter inline
on that same worker; the executor then starts an FFT thunk there -- while that
worker is still logically inside Worker::Parallelize.

WHAT THIS SCRIPT REPRODUCES
---------------------------
The re-entrancy itself, deterministically. Sampling a run of the heavy config
below catches 3 of 4 workers simultaneously in `FftThunk::Execute` ->
`ducc0::detail_threading::latch::wait()`, held continuously across six samples
over six seconds. Verify with:

    python3 .github/scripts/xla_fft_pool_reentrancy_repro.py &
    gdb -p $! --batch -ex 'set pagination off' -ex 'thread apply all bt 30' \
      | grep -c 'ducc0::detail_threading::latch::wait'

A pool worker blocking on a latch fed by its own pool is a bug whether or not
it fully wedges, and the cost is visible: the heavy config below took 27.8s and
67.5s to materialise on two identical runs, a 2.4x swing between runs of the
same graph, which is the pool thrashing rather than deadlocking.

WHAT IT DOES NOT YET REPRODUCE
------------------------------
A full deadlock. It reaches 3 of 4 workers, never 4 of 4, so one worker is
always left to drain the latches. The missing ingredient is identified and is
the second path above: sampling this script finds ZERO `Worker::Parallelize`,
`CountDownAsyncValueRef` and `RunWaiterAndDeleteWaiterNode` frames, so no
parallelised kernel is inline-resuming the executor on a worker. Closing the
gap means getting XLA to split a kernel into >1 workgroup here; until then this
demonstrates the re-entrancy, not the hang.

Dead ends, so nobody re-runs them:
  - A few LARGE FFTs (N=8..32, S=512..2048). Materialises in 0.1-6s, no
    saturation. Size is not what opens the window.
  - Many TINY FFTs alone (N=64..256, S=16..32). 24 trials, no hang.
  - Shrinking the pool with taskset to 2 or 3 CPUs, which should need fewer
    simultaneous FFTs to wedge. 8 trials, no hang -- the latches still drain.

vmap(jit(...)) is the ordering PyAutoFit's `Fitness._vmap` actually builds, and
sizes are small because CI runs under PYAUTO_SMALL_DATASETS=1 (15x15 grids):
breadth opens the window, not transform size.
"""

import os
import time
import faulthandler

faulthandler.dump_traceback_later(int(os.environ.get("DEADLINE", "300")), exit=True)

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

# Defaults are the heavy config the 3-of-4 observation was made on.
D = int(os.environ.get("D", "12"))  # independent datasets == convolution chains
B = int(os.environ.get("B", "128"))  # vmap batch (the parameter sets)
S = int(os.environ.get("S", "256"))  # image side
G = int(os.environ.get("G", "64"))  # "gaussians" -> elementwise breadth

imgs = [jax.random.normal(jax.random.key(d), (S, S), dtype=jnp.float64) for d in range(D)]
psfs = [jax.random.normal(jax.random.key(100 + d), (S, S), dtype=jnp.float64) for d in range(D)]


def model(params, img, psf):
    """Wide elementwise work feeding an FFT convolution -- one dataset's chain."""
    acc = jnp.zeros_like(img)
    for g in range(G):
        acc = acc + jnp.exp(-jnp.abs(params[g]) * (img * img + float(g)))
    return ((jnp.fft.irfft2(jnp.fft.rfft2(acc) * jnp.fft.rfft2(psf), s=img.shape) - img) ** 2).sum()


def call(params):
    """D independent chains in one graph, so their thunks can run concurrently."""
    return sum(model(params, imgs[d], psfs[d]) for d in range(D))


def main():
    fitness = jax.vmap(jax.jit(call))
    params = jax.random.normal(jax.random.key(7), (B, G), dtype=jnp.float64)

    print(f"jax {jax.__version__} cpus={os.cpu_count()} D={D} B={B} S={S} G={G}", flush=True)
    t0 = time.time()
    out = fitness(params)
    print(f"traced+dispatched {time.time() - t0:.1f}s", flush=True)
    t1 = time.time()
    jax.block_until_ready(out)
    print(f"MATERIALISED in {time.time() - t1:.1f}s", flush=True)
    faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    main()
