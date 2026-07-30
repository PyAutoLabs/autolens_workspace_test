#!/usr/bin/env bash
# Gallery runner: regenerate the visualization images then build the contact
# sheet + manifest (gallery/gallery_build.py).
#
# Runs the per-domain visualization scripts from the workspace root (they
# resolve datasets/simulators relative to it and self-bootstrap missing
# datasets). Default = the standard visualization.py per domain minus the
# slow tier (cluster, ~13 min); --all adds the slow tier + JAX-path variants.
#
# Usage (from anywhere):
#   bash gallery/gallery_run.sh                 # default set + build
#   bash gallery/gallery_run.sh --all           # + *_jax variants
#   bash gallery/gallery_run.sh imaging         # one domain + build
#   bash gallery/gallery_run.sh --build-only    # skip runs, just build

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

DEFAULT_SCRIPTS=(
    scripts/imaging/visualization/visualization.py            # ~150s
    scripts/interferometer/visualization/visualization.py     # ~130s
    scripts/point_source/visualization/visualization.py       # ~25s
    scripts/multi_dataset/visualization/imaging.py      # ~95s
    scripts/multi_dataset/visualization/interferometer.py  # ~140s
)
SLOW_SCRIPTS=(
    scripts/cluster/visualization.py            # ~800s — excluded from default
)
JAX_SCRIPTS=(
    scripts/imaging/visualization/visualization_jax.py
    scripts/interferometer/visualization/visualization_jax.py
    scripts/point_source/visualization/visualization_jax.py
)

run_list=()
build_only=0
for arg in "$@"; do
    case "$arg" in
        --build-only) build_only=1 ;;
        --all) run_list=("${DEFAULT_SCRIPTS[@]}" "${SLOW_SCRIPTS[@]}" "${JAX_SCRIPTS[@]}") ;;
        *)  # a domain name: select its scripts (default + slow, not jax)
            for s in "${DEFAULT_SCRIPTS[@]}" "${SLOW_SCRIPTS[@]}"; do
                [[ "$s" == scripts/"$arg"/* ]] && run_list+=("$s")
            done
            ;;
    esac
done
[[ ${#run_list[@]} -eq 0 && $build_only -eq 0 ]] && run_list=("${DEFAULT_SCRIPTS[@]}")

failures=()
for script in "${run_list[@]}"; do
    [[ -f "$script" ]] || { echo "SKIP (missing): $script"; continue; }
    echo "== $script"
    start=$SECONDS
    if python "$script" > /dev/null 2>&1; then
        echo "   ok ($((SECONDS - start))s)"
    else
        echo "   FAILED ($((SECONDS - start))s) — rerun manually for the traceback: python $script"
        failures+=("$script")
    fi
done

python gallery/gallery_build.py --check
status=$?

if [[ ${#failures[@]} -gt 0 ]]; then
    echo
    echo "Visualization scripts FAILED (${#failures[@]}):"
    printf '  %s\n' "${failures[@]}"
    exit 1
fi
exit $status
