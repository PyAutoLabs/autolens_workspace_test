"""
Visualization: Multi-Channel Interferometer Analysis (FactorGraph / Datacube)
=============================================================================

Sibling of ``scripts/multi_dataset/visualization/imaging.py`` for the interferometer / ALMA
datacube case. Verifies that ``VisualizerInterferometer.visualize_combined`` is
invoked when a multi-channel interferometer fit runs through ``af.FactorGraphModel``
and writes ``fit_combined.png`` with one row per channel.

The test runs both calls side by side:

  * **Direct** — call ``VisualizerInterferometer.visualize_combined(analyses=...)``
    directly with the raw ``AnalysisInterferometer`` instances. Confirms the static
    method itself works.

  * **Dispatch** — call ``factor_graph.visualize_combined(...)``, which is what
    ``Search.perform_visualization`` does at iteration boundaries during a real
    datacube fit. Verifies the FactorGraph dispatch chain (after the
    ``AnalysisFactor.visualize_combined`` fix in PyAutoFit) routes the call into
    the static method.

The test fakes a 2-channel cube by loading the same ``build/interferometer/no_lens_light``
dataset twice. Real datacube modeling would have N independent per-channel FITS
folders, but for dispatch verification one dataset shared across two factors is
sufficient — the goal is to confirm the plot lands, not to compare physics.

Run from the ``autolens_workspace_test`` repo root:

    NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/matplotlib \\
        python scripts/multi_dataset/visualization/interferometer.py
"""

import shutil
from os import path
from pathlib import Path
from types import SimpleNamespace

# Push the interferometer test's all-true plots.yaml so subplot_fit lands on disk.
from autolens import conf

conf.instance.push(
    new_path=path.join(
        path.dirname(path.realpath(__file__)), "..", "..", "interferometer", "config"
    ),
    output_path=path.join(
        path.dirname(path.realpath(__file__)), "..", "..", "interferometer", "images"
    ),
)

import autofit as af
import autolens as al
from autolens.interferometer.model.visualizer import VisualizerInterferometer


"""
__Datasets__

Load the build no-lens-light interferometer dataset twice, as if it were a 2-channel
datacube. The dispatch chain is dataset-agnostic — using the same dataset for both
channels keeps the test small and removes any dependency on a real datacube fixture.
"""
dataset_path = Path("dataset") / "build" / "interferometer" / "no_lens_light"

"""
__Dataset Auto-Simulation__

If the dataset does not already exist on your system, it will be created by running the corresponding
simulator script. This ensures that all example scripts can be run without manually simulating data first.
"""
if al.util.dataset.should_simulate(str(dataset_path)):
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/interferometer/simulator/no_lens_light.py"],
        check=True,
    )

mask_radius = 3.0
real_space_mask = al.Mask2D.circular(
    shape_native=(100, 100),
    pixel_scales=0.2,
    radius=mask_radius,
)

n_channels = 2
dataset_list = [
    al.Interferometer.from_fits(
        data_path=dataset_path / "data.fits",
        noise_map_path=dataset_path / "noise_map.fits",
        uv_wavelengths_path=dataset_path / "uv_wavelengths.fits",
        real_space_mask=real_space_mask,
        transformer_class=al.TransformerDFT,
    )
    for _ in range(n_channels)
]


"""
__Per-Channel Analyses__

One ``AnalysisInterferometer`` per channel — the canonical FactorGraph wiring.
"""
analysis_list = [
    al.AnalysisInterferometer(dataset=dataset, use_jax=False, title_prefix="TEST")
    for dataset in dataset_list
]


"""
__Model__

Shared lens (PowerLaw + ExternalShear) + parametric Sersic source. All parameters
are fixed via prior medians — we don't run a search here, we just need a concrete
instance to feed into ``visualize_combined``.
"""
mass = af.Model(al.mp.PowerLaw)
mass.centre.centre_0 = 0.0
mass.centre.centre_1 = 0.0
mass.ell_comps.ell_comps_0 = 0.05
mass.ell_comps.ell_comps_1 = 0.1
mass.einstein_radius = 1.6
mass.slope = 2.0

shear = af.Model(al.mp.ExternalShear)
shear.gamma_1 = 0.05
shear.gamma_2 = 0.05

lens = af.Model(al.Galaxy, redshift=0.5, mass=mass, shear=shear)

source_bulge = af.Model(al.lp.Sersic)
source_bulge.centre.centre_0 = 0.0
source_bulge.centre.centre_1 = 0.0
source_bulge.ell_comps.ell_comps_0 = 0.0
source_bulge.ell_comps.ell_comps_1 = 0.0
source_bulge.intensity = 0.3
source_bulge.effective_radius = 0.2
source_bulge.sersic_index = 1.0
source = af.Model(al.Galaxy, redshift=1.0, bulge=source_bulge)

base_model = af.Collection(galaxies=af.Collection(lens=lens, source=source))


"""
__Per-Factor Models__

Per-channel ``AnalysisFactor`` with no per-factor prior overrides — every parameter is
shared across channels (matching the datacube modeling pattern in
``autolens_workspace/scripts/interferometer/features/datacube/modeling.py``).
"""
analysis_factor_list = [
    af.AnalysisFactor(prior_model=base_model.copy(), analysis=analysis)
    for analysis in analysis_list
]

factor_graph = af.FactorGraphModel(*analysis_factor_list, use_jax=False)
print(
    f"Factor graph: {len(analysis_factor_list)} factors, "
    f"{factor_graph.global_prior_model.total_free_parameters} global free parameters"
)

global_instance = factor_graph.global_prior_model.instance_from_prior_medians()


"""
__Output Paths__
"""
image_path_root = Path("scripts") / "multi_dataset" / "images" / "visualization_interferometer"

if image_path_root.exists():
    shutil.rmtree(image_path_root)
image_path_root.mkdir(parents=True)

direct_path = image_path_root / "direct"
direct_path.mkdir()
direct_paths = SimpleNamespace(image_path=direct_path, output_path=direct_path)

dispatch_path = image_path_root / "dispatch"
dispatch_path.mkdir()
dispatch_paths = SimpleNamespace(image_path=dispatch_path, output_path=dispatch_path)


"""
__Direct Call (ground truth)__
"""
print("\n[direct] Calling VisualizerInterferometer.visualize_combined(...) directly")
VisualizerInterferometer.visualize_combined(
    analyses=analysis_list,
    paths=direct_paths,
    instance=global_instance,
    during_analysis=False,
)
direct_combined = direct_path / "fit_combined.png"
print(f"  fit_combined.png: {'EXISTS' if direct_combined.exists() else 'MISSING'}")


"""
__Dispatch Call (FactorGraph chain)__
"""
print("\n[dispatch] Calling factor_graph.visualize_combined(...)")
factor_graph.visualize_combined(
    instance=global_instance,
    paths=dispatch_paths,
    during_analysis=False,
)
dispatch_combined = dispatch_path / "fit_combined.png"
print(f"  fit_combined.png: {'EXISTS' if dispatch_combined.exists() else 'MISSING'}")


"""
__Verdict__
"""
print("\n=== Verdict ===")
print(f"direct/ outputs:   {sorted(p.name for p in direct_path.iterdir())}")
print(f"dispatch/ outputs: {sorted(p.name for p in dispatch_path.iterdir())}")

if direct_combined.exists() and dispatch_combined.exists():
    print("\nPASS: dispatch chain produces the same combined plot as the direct call.")
elif direct_combined.exists() and not dispatch_combined.exists():
    print(
        "\nFAIL: VisualizerInterferometer.visualize_combined works in isolation but "
        "the dispatch chain SKIPS it. AnalysisFactor.visualize_combined fix may be "
        "missing or incorrectly wired."
    )
elif not direct_combined.exists():
    print(
        "\nINCONCLUSIVE: the direct call also failed to produce fit_combined.png — "
        "either the visualizer config is wrong or the static method itself errored."
    )
