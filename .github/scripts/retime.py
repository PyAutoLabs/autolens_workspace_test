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

Nothing here re-implements how a script is run. `run_one` and the env-profile
resolution are imported from `run_smoke.py`, which imports them in turn from
PyAutoHands, so this harness, the PR gate and the release runner cannot
disagree about what environment a script runs in or how long it may take. The
ceremony around it (dependency chain, install epilogue, cache dirs) is
PyAutoHeart's reusable `smoke-tests.yml`, reached through its `runner` input.

Usage
-----
    python .github/scripts/retime.py --scripts a/b.py,c/d.py --repeats 5

Exit status is 0 whenever the harness itself worked. A timeout here is the
measurement, not a failure -- exiting non-zero on one would make a red run the
expected outcome and hide the runs that genuinely broke.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_smoke import SCRIPTS_DIR, load_cfg, run_one  # noqa: E402

# A completion this far below the cap makes the gap between "completes" and
# "hits the cap" a difference in kind rather than in degree. 18s against an
# 1800s cap is 1%; 1600s against 1800s is 89%. The line is a judgement, so the
# ratio that drove each verdict is always reported alongside it.
BIMODAL_RATIO = 0.5


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scripts",
        required=True,
        help="Comma- or newline-separated script paths, relative to scripts/",
    )
    parser.add_argument("--repeats", type=int, default=5)
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

    print(f"Re-timing {len(scripts)} script(s), {args.repeats} run(s) each\n")

    for script_rel in scripts:
        runs = []
        cap = None
        for attempt in range(1, args.repeats + 1):
            print(f"::group::{script_rel} — run {attempt}/{args.repeats}")
            _, rc, elapsed, output, cap = run_one(script_rel, cfg)
            print(output, end="")
            status = "PASS" if rc == 0 else (f"TIMEOUT ({cap}s)" if rc == 124 else f"FAIL (exit {rc})")
            print(f"\n[{status}] {script_rel} — {elapsed:.1f}s")
            print("::endgroup::")
            runs.append((rc, elapsed))

        verdict, detail = classify(runs, float(cap))
        results[script_rel] = {
            "cap": cap,
            "runs": [{"returncode": rc, "elapsed": round(e, 1)} for rc, e in runs],
            "verdict": verdict,
            "detail": detail,
        }
        print(f"\n=== {script_rel}: {verdict} — {detail} ===\n")

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
