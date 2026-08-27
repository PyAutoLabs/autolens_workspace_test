"""
Run named scripts repeatedly and report the timing distribution of each.

This exists to answer one question that a single run cannot: is a script that
misses its cap **slow**, or does it **stall**?

- A slow script has a tight distribution. Every run takes about the same time,
  and that time is near or over the cap.
- A stalling script is bimodal. It completes in tens of seconds when it
  completes at all, and otherwise makes no progress until the cap kills it.

Those two route to completely different places -- one is the Profiling Agent's
speedup, the other is a bug -- and the markers in `config/build/no_run.yaml`
have been assigning them by which failure a run happened to show. See #271.

Nothing here *decides* how a script is run. The env-profile resolution, the
per-script cap and the process-group kill are imported straight from
PyAutoHands -- the same three primitives `autohands/build_util.py` uses to run
the mega-run -- so this harness, the PR gate and the release runner cannot
disagree about what environment a script runs in or how long it may take. Only
the subprocess loop that spends those decisions is local, because `run_smoke.py`
is now a shim over `autohands/run_python.py` (PyAutoHands#260) and has no
per-script entry point left to borrow. The ceremony around it (dependency chain,
install epilogue, cache dirs) is PyAutoHeart's reusable `smoke-tests.yml`,
reached through its `runner` input.

`--arm` is the one deliberate exception, and it is narrow by construction: an
overlay is applied ON TOP of the resolved profile environment, never instead of
it, and every run prints the arm it ran under. Without `--arm` the environment
is byte-for-byte what the gate would give the script. That an experiment can
say "the profile, but with this one variable moved" is the whole point -- the
alternative, used in the PyAutoFit#1528 campaign, was to edit
`profile_smoke.yaml` on a branch and re-dispatch per arm, which made every
comparison a comparison between dispatches.

An A/B mode sits on top of that (PyAutoFit#1530). `--arm` deals runs
round-robin between named environment overlays *within one dispatch*, and
`--dump-after` reads a still-running child's native stacks before the cap
kills it. Both exist for the XLA CPU Eigen-pool wedge, where the failure is a
hang with no Python-visible cause and the hang RATE wanders between dispatches
-- so arms compared across separate runs each carry their own drift, and a
stack taken after the kill is a stack of nothing.

Usage
-----
    python .github/scripts/retime.py --scripts a/b.py,c/d.py --repeats 5

    python .github/scripts/retime.py --scripts imaging/jax_likelihood/mge_group.py \
        --repeats 6 --dump-after 150 \
        --arm control:XLA_FLAGS= --arm quota:XLA_FLAGS=,affinity=auto

Exit status is 0 whenever the harness itself worked. A timeout here is the
measurement, not a failure -- exiting non-zero on one would make a red run the
expected outcome and hide the runs that genuinely broke.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = WORKSPACE / "scripts"
ENV_VARS_FILE = WORKSPACE / "config" / "build" / "profile_smoke.yaml"

# CI puts PyAutoHands/autohands on PYTHONPATH (PyAutoHeart's reusable
# smoke-tests.yml clones it alongside the dependency chain); for local runs,
# fall back to the sibling checkout. Exactly the contract `run_smoke.py` uses,
# so the two are reached the same way from the same runner input.
#
# There is deliberately no local fallback definition of these three. A harness
# that quietly re-implemented the cap or the env resolution when PyAutoHands was
# absent would be able to disagree with the gate it exists to explain -- the one
# thing this file must not do. Absent Hands, it fails at import instead.
try:
    from build_util import kill_group, timeout_for
    from env_config import build_env_for_script, load_env_config
except ImportError:  # pragma: no cover - local-run fallback
    sys.path.insert(0, str(WORKSPACE.parent / "PyAutoHands" / "autohands"))
    from build_util import kill_group, timeout_for
    from env_config import build_env_for_script, load_env_config

# A completion this far below the cap makes the gap between "completes" and
# "hits the cap" a difference in kind rather than in degree. 18s against an
# 1800s cap is 1%; 1600s against 1800s is 89%. The line is a judgement, so the
# ratio that drove each verdict is always reported alongside it.
BIMODAL_RATIO = 0.5


# ---------------------------------------------------------------------------
# Diagnostics for the XLA CPU Eigen-pool wedge (PyAutoFit#1530)
#
# Everything below is opt-in via --arm / --dump-after, except the CPU-topology
# banner, which is unconditional because it is the one reading no later run can
# reconstruct. None of it changes how a script is run: the env still comes from
# build_env_for_script and the cap still from timeout_for, and an arm's overlay
# is applied ON TOP of the resolved env and printed verbatim, so a run's actual
# environment stays readable off the log rather than inferred from the profile.
# ---------------------------------------------------------------------------

CGROUP_V2_CPU_MAX = "/sys/fs/cgroup/cpu.max"
CGROUP_V1_QUOTA = "/sys/fs/cgroup/cpu/cpu.cfs_quota_us"
CGROUP_V1_PERIOD = "/sys/fs/cgroup/cpu/cpu.cfs_period_us"


def _read(path):
    """File contents stripped, or None when it does not exist or cannot be read."""
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def cgroup_quota_cpus():
    """
    How many CPUs this cgroup may actually schedule, or None when unlimited.

    This is precisely the number XLA cannot see. TSL's NumSchedulableCPUs() --
    which sizes XLA CPU's Eigen intra-op thread pool -- reads
    sched_getaffinity(), and a CFS *quota* does not appear in the affinity
    mask. A runner advertising 4 schedulable CPUs under a 2-CPU quota
    therefore gets a 4-thread pool of which only 2 can ever run at once, which
    is the classic way that pool wedges.

    Floors rather than rounds: a pool sized above the quota is the failure
    mode under test, so erring low is the safe direction.
    """
    raw = _read(CGROUP_V2_CPU_MAX)
    if raw:
        quota, _, period = raw.partition(" ")
        if quota != "max" and period:
            try:
                return max(1, int(float(quota) // float(period)))
            except (ValueError, ZeroDivisionError):
                return None
        return None
    quota = _read(CGROUP_V1_QUOTA)
    period = _read(CGROUP_V1_PERIOD)
    if quota and period and quota != "-1":
        try:
            return max(1, int(float(quota) // float(period)))
        except (ValueError, ZeroDivisionError):
            return None
    return None


def cpu_topology():
    """Every reading that bears on how large XLA will size its thread pool."""
    affinity = sorted(os.sched_getaffinity(0))
    cpuinfo = _read("/proc/cpuinfo") or ""
    meminfo = _read("/proc/meminfo") or ""
    return [
        ("os.cpu_count()", os.cpu_count()),
        ("len(os.sched_getaffinity(0))", len(affinity)),
        ("sched_getaffinity mask", ",".join(str(c) for c in affinity)),
        ("/proc/cpuinfo processors", cpuinfo.count("processor\t")),
        (CGROUP_V2_CPU_MAX, _read(CGROUP_V2_CPU_MAX)),
        ("/sys/fs/cgroup/cpuset.cpus.effective", _read("/sys/fs/cgroup/cpuset.cpus.effective")),
        ("cgroup quota -> CPUs", cgroup_quota_cpus()),
        ("MemTotal", next((l.split(":", 1)[1].strip() for l in meminfo.splitlines()
                           if l.startswith("MemTotal")), None)),
    ]


def print_topology():
    """
    The Q3 comparison, recorded on every run whether or not an arm uses it.

    If `cgroup quota -> CPUs` is None there is no CFS quota in force, and the
    oversubscription hypothesis is dead on this runner before any arm runs --
    say so here rather than letting a null result read as a refutation.
    """
    print("::group::CPU topology — what XLA will size its Eigen pool from")
    for key, value in cpu_topology():
        print(f"  {key:<40} {value}")
    if cgroup_quota_cpus() is None:
        print("  NOTE: no CFS quota in force — sched_getaffinity IS the real")
        print("        limit here, so an affinity= arm cannot correct anything.")
    print("::endgroup::")


def parse_arm(raw):
    """
    Parse one --arm value: NAME:KEY=VALUE[,KEY=VALUE...].

    Values may contain neither a comma nor whitespace: the reusable workflow
    runs `python "$RUNNER" $RUNNER_ARGS` unquoted, so a token with a space in
    it would arrive as two arguments. An empty VALUE sets the variable to the
    empty string, which is how an arm turns a profile default off while
    leaving the key visible in the log.

    The key `affinity` is reserved and is not an environment variable:
    `affinity=auto` pins the child to the cgroup quota (cgroup_quota_cpus),
    `affinity=N` to the first N schedulable CPUs.
    """
    name, sep, body = raw.partition(":")
    if not sep or not name.strip():
        raise argparse.ArgumentTypeError(f"--arm wants NAME:KEY=VALUE, got {raw!r}")
    overlay = {}
    for pair in body.split(","):
        pair = pair.strip()
        if not pair:
            continue
        key, eq, value = pair.partition("=")
        if not eq or not key.strip():
            raise argparse.ArgumentTypeError(f"--arm {name}: {pair!r} is not KEY=VALUE")
        overlay[key.strip()] = value
    return (name.strip(), overlay)


def resolve_affinity(spec):
    """
    Turn an `affinity=` spec into a CPU set, or None to leave the mask alone.

    None from `auto` means no quota is in force, so there is nothing to
    correct -- an honest no-op rather than an arbitrary narrowing that would
    make the arm look like it tested something.
    """
    available = sorted(os.sched_getaffinity(0))
    if spec == "auto":
        want = cgroup_quota_cpus()
        if want is None:
            return None
    else:
        try:
            want = int(spec)
        except ValueError:
            raise SystemExit(f"affinity= wants 'auto' or an integer, got {spec!r}")
    return set(available[: max(1, min(want, len(available)))])


_DEBUG_TOOLS = {}


def _helper(cmd, timeout):
    """Run a diagnostic helper. Its failure is evidence, never fatal."""
    try:
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return (done.stdout or "") + (done.stderr or "")
    except FileNotFoundError:
        return f"[{cmd[0]}: not installed]\n"
    except subprocess.TimeoutExpired:
        return f"[{cmd[0]}: timed out after {timeout}s]\n"
    except Exception as exc:  # pragma: no cover - diagnostics must not raise
        return f"[{cmd[0]}: {exc}]\n"


def _ensure_debug_tools():
    """
    Install py-spy and gdb once, on first hang.

    Deliberately lazy: a run that never hangs must not pay for them, and a
    failed install must not turn a measurement into a red run.

    ptrace_scope is dropped to 0 because both tools attach as SIBLINGS of the
    target -- the harness is the child's parent, gdb and py-spy are not -- so
    Yama's default scope 1 (ancestors only) would deny them.
    """
    if _DEBUG_TOOLS:
        return
    _DEBUG_TOOLS["ptrace_scope"] = _helper(
        ["sudo", "-n", "sh", "-c", "echo 0 > /proc/sys/kernel/yama/ptrace_scope"], 30
    )
    _DEBUG_TOOLS["py-spy"] = _helper(
        [sys.executable, "-m", "pip", "install", "-q", "py-spy"], 300
    )
    _helper(["sudo", "-n", "apt-get", "update", "-qq"], 300)
    _DEBUG_TOOLS["gdb"] = _helper(
        ["sudo", "-n", "apt-get", "install", "-y", "-qq", "gdb"], 600
    )


def thread_states(pid):
    """
    Kernel-side state of every thread in the process. The reading that always
    works, and the one that discriminates without any symbols at all: a pool
    deadlocked on a futex and a pool spinning in a work-stealing loop look
    completely different here even when libxla_extension.so is stripped.
    """
    lines = []
    status = _read(f"/proc/{pid}/status") or ""
    for line in status.splitlines():
        if line.startswith(("State:", "Threads:")):
            lines.append("  " + line)
    task_dir = Path(f"/proc/{pid}/task")
    try:
        tids = sorted(task_dir.iterdir(), key=lambda p: int(p.name))
    except OSError as exc:
        lines.append(f"  [cannot read {task_dir}: {exc}]")
        return "\n".join(lines)
    lines.append(f"  {'TID':>8}  {'COMM':<16} {'ST':<3} WCHAN")
    for tid in tids:
        comm = _read(tid / "comm") or "?"
        wchan = _read(tid / "wchan") or "?"
        stat = _read(tid / "stat") or ""
        # Field 3 of /proc/<tid>/stat is the state char. comm (field 2) can
        # itself contain spaces inside its parentheses, so split after the ')'
        # rather than on whitespace from the left.
        state = stat.rpartition(")")[2].split()[0] if ")" in stat else "?"
        lines.append(f"  {tid.name:>8}  {comm:<16} {state:<3} {wchan}")
    return "\n".join(lines)


def _py_spy():
    """py-spy as pip installed it, next to the interpreter running us."""
    candidate = Path(sys.executable).parent / "py-spy"
    if candidate.exists():
        return str(candidate)
    return shutil.which("py-spy") or "py-spy"


def native_dump(pid):
    """
    Everything obtainable about where the wedged threads are parked.

    Ordered cheapest-and-surest first. /proc costs nothing and cannot fail;
    py-spy --native unwinds through the C++ frames faulthandler cannot see;
    gdb is last because attaching stops the process, and a failed detach must
    not cost us the two readings already taken.
    """
    _ensure_debug_tools()
    blocks = [f"--- /proc/{pid}: thread states ---", thread_states(pid)]
    blocks.append("--- py-spy dump --native ---")
    spy = [_py_spy(), "dump", "--native", "--pid", str(pid)]
    out = _helper(["sudo", "-n"] + spy, 240)
    if not out.strip() or "sudo:" in out[:200]:
        out = _helper(spy, 240)
    blocks.append(out)
    blocks.append("--- gdb: thread apply all bt ---")
    blocks.append(
        _helper(
            ["sudo", "-n", "gdb", "-p", str(pid), "--batch",
             "-ex", "set pagination off", "-ex", "thread apply all bt"],
            300,
        )
    )
    for tool, result in _DEBUG_TOOLS.items():
        if result.strip():
            blocks.append(f"[setup: {tool}] {result.strip()[:400]}")
    return "\n".join(blocks)


def classify(runs, cap):
    """
    Turn one script's runs into a verdict.

    `runs` is a list of (returncode, elapsed). Returns (verdict, detail).
    """
    timeouts = [elapsed for rc, elapsed in runs if rc == 124]
    completions = sorted(elapsed for rc, elapsed in runs if rc == 0)
    errors = [rc for rc, _ in runs if rc not in (0, 124)]

    if errors and not completions and not timeouts:
        return "ERROR", f"every run failed outright (exit {errors[0]})"

    if not timeouts:
        if not completions:
            return "ERROR", "no run completed and none hit the cap"
        worst = max(completions)
        return (
            "NEITHER",
            f"completed {len(completions)}/{len(runs)} runs, "
            f"slowest {worst:.1f}s = {worst / cap:.0%} of the {cap:.0f}s cap",
        )

    if not completions:
        # Everything hit the cap. That is consistent with a slow script AND
        # with a stall that never got lucky; without a completion time there is
        # nothing to compare the cap against, so say so rather than guess.
        return (
            "AMBIGUOUS",
            f"hit the cap in all {len(runs)} runs and never completed -- "
            f"no completion time to compare; re-run with a higher cap",
        )

    median = statistics.median(completions)
    ratio = median / cap

    if ratio < BIMODAL_RATIO:
        verdict = "STALL"
        why = "bimodal"
    else:
        verdict = "SLOW"
        why = "completions cluster near the cap"

    return (
        verdict,
        f"{why}: {len(timeouts)}/{len(runs)} hit the {cap:.0f}s cap, "
        f"the rest completed in {min(completions):.1f}-{max(completions):.1f}s "
        f"(median {median:.1f}s = {ratio:.0%} of the cap)",
    )


def parse_scripts(raw):
    """Split a comma / newline separated list, dropping blanks."""
    out = []
    for chunk in raw.replace("\n", ",").split(","):
        chunk = chunk.strip()
        if chunk:
            out.append(chunk)
    return out


def load_cfg():
    """Parsed env profile, or None when the workspace has none.

    None flows through build_env_for_script -> None -> subprocess inherits the
    parent environment unchanged.
    """
    if not ENV_VARS_FILE.exists():
        return None
    return load_env_config(ENV_VARS_FILE)


def run_one(
    script_rel: str,
    cfg: dict | None,
    arm: tuple[str, dict] | None = None,
    dump_after: float | None = None,
) -> tuple[str, int, float, str, int]:
    """Run one script once, capped at the per-script resolved timeout.

    The script runs in its own session (``start_new_session=True``) so that a
    timeout can kill the whole process group rather than just the direct child.
    That distinction is load-bearing: capturing output means waiting for the
    stdout pipe to reach EOF, and any grandchild that inherited the pipe holds
    it open even after the child itself has exited. A script whose work has
    finished can therefore hang the harness indefinitely -- and a harness that
    hangs measures nothing.

    The wall clock is the PARENT's, so a stall inside the child is timed the
    same way a completion is, which is the whole point of re-timing.

    `arm` is an optional (name, overlay) pair applied on top of the resolved
    environment; `dump_after` is the age in seconds at which a still-running
    child gets its native stack read. Both default to off, so an ordinary
    re-timing run behaves exactly as it did before they existed.
    """
    env = build_env_for_script(Path(script_rel), cfg)
    affinity = None
    if arm:
        for key, value in arm[1].items():
            if key == "affinity":
                affinity = resolve_affinity(value)
            else:
                env[key] = value
    timeout_secs = timeout_for(env)
    script_path = SCRIPTS_DIR / script_rel

    # The mask is set on the PARENT and restored immediately: the child
    # inherits it across fork/exec, so there is no window in which it could
    # read a wider one. Setting it on the child after Popen would race the
    # import of jax, which is when XLA sizes its pool.
    parent_affinity = os.sched_getaffinity(0)
    if affinity:
        os.sched_setaffinity(0, affinity)
    t0 = time.time()
    try:
        proc = subprocess.Popen(
            [sys.executable, str(script_path)],
            cwd=str(WORKSPACE),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
    finally:
        if affinity:
            os.sched_setaffinity(0, parent_affinity)

    timed_out = False
    dumped = ""
    try:
        # Two-stage wait when --dump-after is set: the first stage exists only
        # to catch the process still alive, and while it IS still alive is the
        # only moment the wedged threads can be read at all. Waiting for the
        # cap and dumping afterwards would dump a corpse.
        if dump_after is not None and dump_after < timeout_secs:
            try:
                output, _ = proc.communicate(timeout=dump_after)
                returncode = proc.returncode
            except subprocess.TimeoutExpired:
                dumped = native_dump(proc.pid)
                remaining = timeout_secs - (time.time() - t0)
                output, _ = proc.communicate(timeout=max(1.0, remaining))
                returncode = proc.returncode
        else:
            output, _ = proc.communicate(timeout=timeout_secs)
            returncode = proc.returncode
    except subprocess.TimeoutExpired:
        timed_out = True
        kill_group(proc)
        # The group is gone, so this drains whatever was buffered and returns.
        output, _ = proc.communicate()
        # Always 124 (the conventional timeout code), never the signal we just
        # sent. Reporting proc.returncode here would surface -9 for a script
        # killed mid-run, and `classify` reads 124 as the cap being hit and
        # anything else as an outright failure -- the two verdicts this file
        # exists to keep apart.
        returncode = 124
    elapsed = time.time() - t0
    if timed_out:
        output = (output or "") + (
            f"\n::error::TIMEOUT after {timeout_secs}s — killed the process group.\n"
        )
    if dumped:
        # Appended to the run's own output so the stack sits next to the last
        # line the script managed to print, which is what places the hang.
        output = (output or "") + (
            f"\n::group::native dump after {dump_after:.0f}s — {script_rel}"
            f"{' [' + arm[0] + ']' if arm else ''}\n{dumped}\n::endgroup::\n"
        )
    # timeout_secs is returned so the caller reports the cap this script
    # actually ran under, not the run-wide default -- a quoted cap below the
    # enforced one biases every "too slow to un-skip?" call (the 60s-cap myth).
    return script_rel, returncode, elapsed, output or "", timeout_secs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scripts",
        required=True,
        help="Comma- or newline-separated script paths, relative to scripts/",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Total runs per script. With --arm, these are DEALT round-robin "
        "across the arms, so --repeats 6 over two arms is 3 each.",
    )
    parser.add_argument(
        "--arm",
        action="append",
        type=parse_arm,
        metavar="NAME:KEY=VALUE[,KEY=VALUE]",
        help="An environment overlay to A/B, repeatable. Runs alternate "
        "between arms within a single dispatch, so the design is ABAB by "
        "construction rather than by comparing dispatches hours apart — the "
        "hang rate under study demonstrably wanders between them. The "
        "reserved key `affinity` takes `auto` (the cgroup quota) or an "
        "integer, and pins the child rather than setting a variable.",
    )
    parser.add_argument(
        "--dump-after",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Read a still-running child's native stacks at this age, before "
        "the cap kills it. Set it above a normal completion and below the "
        "cap. Off by default.",
    )
    args = parser.parse_args()

    scripts = parse_scripts(args.scripts)
    if not scripts:
        print("ERROR: --scripts named nothing", file=sys.stderr)
        return 2

    missing = [s for s in scripts if not (SCRIPTS_DIR / s).exists()]
    if missing:
        # A path that does not exist would otherwise be reported as a fast,
        # clean failure and read as "not slow after all".
        print(f"ERROR: no such script(s): {', '.join(missing)}", file=sys.stderr)
        return 2

    cfg = load_cfg()
    results = {}
    print_topology()

    # No arms is one nameless arm with an empty overlay, so the loop below has
    # a single shape and the un-armed path stays the one that has been running
    # in CI since #272.
    arms = args.arm or [None]

    print(f"Re-timing {len(scripts)} script(s), {args.repeats} run(s) each")
    if args.arm:
        print(f"Arms dealt round-robin ({len(arms)}): " + ", ".join(
            f"{name}={overlay}" for name, overlay in arms
        ))
    print()

    for script_rel in scripts:
        per_arm = {}
        caps = {}
        for attempt in range(1, args.repeats + 1):
            arm = arms[(attempt - 1) % len(arms)]
            label = f"{script_rel} [{arm[0]}]" if arm else script_rel
            print(f"::group::{label} — run {attempt}/{args.repeats}")
            if arm:
                # The overlay as applied, not as requested: a run whose arm is
                # quoted from the log rather than from the dispatch inputs is
                # the difference between evidence and a claim about evidence.
                print(f"[arm] {arm[0]}: {arm[1]}")
            _, rc, elapsed, output, cap = run_one(
                script_rel, cfg, arm=arm, dump_after=args.dump_after
            )
            print(output, end="")
            status = "PASS" if rc == 0 else (f"TIMEOUT ({cap}s)" if rc == 124 else f"FAIL (exit {rc})")
            print(f"\n[{status}] {label} — {elapsed:.1f}s")
            print("::endgroup::")
            per_arm.setdefault(label, []).append((rc, elapsed))
            caps[label] = cap

        for label, runs in per_arm.items():
            verdict, detail = classify(runs, float(caps[label]))
            results[label] = {
                "cap": caps[label],
                "runs": [{"returncode": rc, "elapsed": round(e, 1)} for rc, e in runs],
                "verdict": verdict,
                "detail": detail,
            }
            print(f"\n=== {label}: {verdict} — {detail} ===\n")

    print("=== Re-timing summary ===")
    for script_rel, r in results.items():
        print(f"  {r['verdict']:10s} {script_rel} — {r['detail']}")

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a") as f:
            f.write(f"### Re-timing — {args.repeats} run(s) per script\n\n")
            f.write("| Script | Verdict | Evidence |\n|---|---|---|\n")
            for script_rel, r in results.items():
                f.write(f"| `{script_rel}` | **{r['verdict']}** | {r['detail']} |\n")

    Path("retime_results.json").write_text(json.dumps(results, indent=2))
    print("\nWrote retime_results.json")

    return 0


if __name__ == "__main__":
    sys.exit(main())
