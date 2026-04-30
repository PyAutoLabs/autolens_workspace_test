"""
Integration test: aggregator Tracer scrape.

Exercises TracerAgg randomly_drawn_via_pdf_gen_from and
all_above_weight_gen_from with weight verification.
"""
import os
import shutil
from os import path

from autoconf import conf
from autoconf.conf import with_config
import autofit as af
import autolens as al
from autolens import fixtures
from autofit.non_linear.samples import Sample

os.environ["PYAUTO_TEST_MODE"] = "1"

directory = path.dirname(path.realpath(__file__))

conf.instance.push(
    new_path=path.join(directory, "config"),
    output_path=path.join(directory, "output"),
)

database_file = "db_tracer"


def clean():
    database_sqlite = path.join(conf.instance.output_path, f"{database_file}.sqlite")
    if path.exists(database_sqlite):
        os.remove(database_sqlite)
    result_path = path.join(conf.instance.output_path, database_file)
    if path.exists(result_path):
        shutil.rmtree(result_path)


@with_config("general", "output", "samples_to_csv", value=True)
def aggregator_from(analysis, model, samples):
    result_path = path.join(conf.instance.output_path, database_file)
    clean()
    search = al.m.MockSearch(
        samples=samples, result=al.m.MockResult(model=model, samples=samples)
    )
    search.paths = af.DirectoryPaths(path_prefix=database_file)
    search.fit(model=model, analysis=analysis)
    analysis.modify_before_fit(paths=search.paths, model=model)
    analysis.visualize_before_fit(paths=search.paths, model=model)
    db_path = path.join(conf.instance.output_path, f"{database_file}.sqlite")
    agg = af.Aggregator.from_database(filename=db_path)
    agg.add_directory(directory=result_path)
    return agg


def make_model():
    dataset_model = af.Model(al.DatasetModel)
    dataset_model.background_sky_level = af.UniformPrior(
        lower_limit=0.5, upper_limit=1.5
    )
    return af.Collection(
        dataset_model=dataset_model,
        galaxies=af.Collection(
            lens=af.Model(al.Galaxy, redshift=0.5, light=al.lp.Sersic),
            source=af.Model(al.Galaxy, redshift=1.0, light=al.lp.Sersic),
        ),
    )


def make_samples(model):
    parameters = [model.prior_count * [1.0], model.prior_count * [10.0]]
    sample_list = Sample.from_lists(
        model=model,
        parameter_lists=parameters,
        log_likelihood_list=[1.0, 2.0],
        log_prior_list=[0.0, 0.0],
        weight_list=[0.0, 1.0],
    )
    return al.m.MockSamples(
        model=model,
        sample_list=sample_list,
        prior_means=[1.0] * model.prior_count,
    )


model = make_model()
samples = make_samples(model)
masked_imaging = fixtures.make_masked_imaging_7x7()
analysis = al.AnalysisImaging(dataset=masked_imaging, use_jax=False)

# --- Test 1: randomly_drawn_via_pdf_gen_from ---

print("Test 1: tracer_randomly_drawn_via_pdf_gen_from ... ", end="")

agg = aggregator_from(analysis=analysis, model=model, samples=samples)
tracer_agg = al.agg.TracerAgg(aggregator=agg)
tracer_pdf_gen = tracer_agg.randomly_drawn_via_pdf_gen_from(total_samples=2)

i = 0
for tracer_gen in tracer_pdf_gen:
    for tracer_list in tracer_gen:
        i += 1
        assert tracer_list[0].galaxies[0].redshift == 0.5
        assert tracer_list[0].galaxies[0].light.centre == (10.0, 10.0)
        assert tracer_list[0].galaxies[1].redshift == 1.0
assert i == 2
clean()

print("PASSED")

# --- Test 2: all_above_weight_gen_from ---

print("Test 2: tracer_all_above_weight_gen ... ", end="")

analysis_7x7 = fixtures.make_analysis_imaging_7x7()
agg = aggregator_from(analysis=analysis_7x7, model=model, samples=samples)
tracer_agg = al.agg.TracerAgg(aggregator=agg)
tracer_pdf_gen = tracer_agg.all_above_weight_gen_from(minimum_weight=-1.0)
weight_pdf_gen = tracer_agg.weights_above_gen_from(minimum_weight=-1.0)

i = 0
for tracer_gen, weight_gen in zip(tracer_pdf_gen, weight_pdf_gen):
    for tracer_list in tracer_gen:
        i += 1
        if i == 1:
            assert tracer_list[0].galaxies[0].redshift == 0.5
            assert tracer_list[0].galaxies[0].light.centre == (1.0, 1.0)
            assert tracer_list[0].galaxies[1].redshift == 1.0
        if i == 2:
            assert tracer_list[0].galaxies[0].redshift == 0.5
            assert tracer_list[0].galaxies[0].light.centre == (10.0, 10.0)
            assert tracer_list[0].galaxies[1].redshift == 1.0
    for weight in weight_gen:
        if i == 0:
            assert weight == 0.0
        if i == 1:
            assert weight == 1.0
assert i == 2
clean()

print("PASSED")

print("\nAll tracer aggregator tests passed.")
