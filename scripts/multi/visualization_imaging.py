"""
Visualization: Multi-Band Imaging Analysis (FactorGraph)
=========================================================

Investigates whether ``VisualizerImaging.visualize_combined`` is actually invoked when
a multi-band imaging fit runs through ``af.FactorGraphModel``. The static method exists
and is intended to write ``fit_combined.png`` (the row-per-dataset combined subplot),
but a candidate bug in the dispatch chain causes it to be silently skipped.

The test runs both calls side by side and reports which artefacts each produces:

  * **Direct** — call ``VisualizerImaging.visualize_combined(analyses=..., ...)`` directly
    with the raw ``AnalysisImaging`` instances, bypassing PyAutoFit's dispatch chain.
    This is the ground truth: it confirms that the static method itself works.

  * **Dispatch** — call ``factor_graph.visualize_combined(instance=..., paths=..., ...)``,
    which is what ``Search.perform_visualization`` does at iteration boundaries during
    a real fit. The question is whether this routes the call into the static method.

Suspected bug
-------------
``FactorGraphModel.visualize_combined`` calls ``model_factors[0].visualize_combined(
model_factors, paths, instance, ...)``. ``AnalysisFactor`` does not define
``visualize_combined`` itself, so the call goes through ``Analysis.__getattr__``'s
auto-forwarder, which inspects the target method's signature and **skips combined
methods** ("Skipping {item} as this is not a combined analysis") whenever ``analyses``
appears in the parameters. The result: the static method is never called.

Run from the ``autolens_workspace_test`` repo root:

    NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/matplotlib \\
        python scripts/multi/visualization_imaging.py
"""

import shutil
from os import path
from pathlib import Path
from types import SimpleNamespace

# Push the imaging test's all-true plots.yaml so subplot_fit lands on disk.
from autoconf import conf

conf.instance.push(
    new_path=path.join(
        path.dirname(path.realpath(__file__)), "..", "imaging", "config"
    ),
    output_path=path.join(
        path.dirname(path.realpath(__file__)), "..", "imaging", "images"
    ),
)

import autofit as af
import autolens as al
from autolens.imaging.model.visualizer import VisualizerImaging


"""
__Datasets__

Load the two-band ``g``/``r`` imaging cube shipped under ``dataset/multi/lens_sersic/``.
The g-band uses pixel scale 0.08", the r-band uses 0.12".
"""

dataset_path = Path("dataset") / "multi" / "lens_sersic"

waveband_list = ["g", "r"]
pixel_scales_list = [0.08, 0.12]

dataset_list = []
for waveband, pixel_scale in zip(waveband_list, pixel_scales_list):
    dataset = al.Imaging.from_fits(
        data_path=dataset_path / f"{waveband}_data.fits",
        psf_path=dataset_path / f"{waveband}_psf.fits",
        noise_map_path=dataset_path / f"{waveband}_noise_map.fits",
        pixel_scales=pixel_scale,
        over_sample_size_lp=2,
        over_sample_size_pixelization=2,
    )

    mask = al.Mask2D.circular(
        shape_native=dataset.shape_native,
        pixel_scales=dataset.pixel_scales,
        radius=3.0,
    )
    dataset = dataset.apply_mask(mask=mask)
    dataset_list.append(dataset)


"""
__Per-Band Analyses__

One ``AnalysisImaging`` per band — this is the canonical multi-band wiring from
``autolens_workspace/scripts/multi/modeling.py``.
"""
analysis_list = [
    al.AnalysisImaging(dataset=dataset, use_jax=False, title_prefix="TEST")
    for dataset in dataset_list
]


"""
__Model__

Shared lens (PowerLaw + ExternalShear) + parametric Sersic source. Every parameter is
fixed to a sensible value via the prior median trick — we don't run a search here, we
just need a concrete instance to feed into ``visualize_combined``.
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
source_bulge.intensity = 1.0
source_bulge.effective_radius = 0.2
source_bulge.sersic_index = 1.0
source = af.Model(al.Galaxy, redshift=1.0, bulge=source_bulge)

base_model = af.Collection(galaxies=af.Collection(lens=lens, source=source))


"""
__Per-Factor Models with Per-Band Intensity__

Override the source ``intensity`` per factor — exactly the multi-band recipe.
"""
analysis_factor_list = []
for i, analysis in enumerate(analysis_list):
    model_analysis = base_model.copy()
    model_analysis.galaxies.source.bulge.intensity = af.LogUniformPrior(
        lower_limit=0.1, upper_limit=10.0
    )
    analysis_factor_list.append(
        af.AnalysisFactor(prior_model=model_analysis, analysis=analysis)
    )

factor_graph = af.FactorGraphModel(*analysis_factor_list, use_jax=False)
print(
    f"Factor graph: {len(analysis_factor_list)} factors, "
    f"{factor_graph.global_prior_model.total_free_parameters} global free parameters"
)

"""
__Concrete Instance__

The global instance is a ``Collection`` whose entries are per-factor model instances —
that is exactly the iterable shape that ``VisualizerImaging.visualize_combined`` expects
(its body does ``for analysis, single_instance in zip(analyses, instance)``).
"""
global_instance = factor_graph.global_prior_model.instance_from_prior_medians()


"""
__Output Paths__

Two output folders:
  * ``direct/``   — populated by the direct ``VisualizerImaging.visualize_combined`` call.
  * ``dispatch/`` — populated (or NOT) by the FactorGraph dispatch path.
"""
image_path_root = Path("scripts") / "multi" / "images" / "visualization_imaging"

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

Call the static method directly with the raw ``AnalysisImaging`` instances and the
per-factor instance Collection. This bypasses PyAutoFit's dispatch chain.
"""
print("\n[direct] Calling VisualizerImaging.visualize_combined(...) directly")
VisualizerImaging.visualize_combined(
    analyses=analysis_list,
    paths=direct_paths,
    instance=global_instance,
    during_analysis=False,
)
direct_combined = direct_path / "fit_combined.png"
print(f"  fit_combined.png: {'EXISTS' if direct_combined.exists() else 'MISSING'}")


"""
__Dispatch Call (the actual question)__

Call ``factor_graph.visualize_combined(...)`` exactly the way ``Search.perform_visualization``
does. If the dispatch chain forwards the call into ``VisualizerImaging.visualize_combined``,
``dispatch/fit_combined.png`` will appear. If the auto-forwarder skips it, the folder
will be empty.
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
        "\nFAIL: VisualizerImaging.visualize_combined works in isolation but the "
        "dispatch chain SKIPS it. This confirms the suspected bug."
    )
elif not direct_combined.exists():
    print(
        "\nINCONCLUSIVE: the direct call also failed to produce fit_combined.png — "
        "either the visualizer config is wrong or the static method itself errored."
    )
