import click
from loguru import logger
from vivarium.framework.utilities import handle_exceptions

from vivarium_csu_zenon import paths
import vivarium_csu_zenon.globals as project_globals

from vivarium_csu_zenon.tools import configure_logging_to_terminal
from vivarium_csu_zenon.tools import build_model_specifications
from vivarium_csu_zenon.tools import build_artifacts
from vivarium_csu_zenon.tools import build_results
from vivarium_csu_zenon.tools import build_fpg_thresholds
from vivarium_csu_zenon.tools import build_ldl_thresholds
from vivarium_csu_zenon.tools import build_results_from_logs
from vivarium_csu_zenon.tools import build_joint_pafs
from vivarium_csu_zenon.tools import build_sample_histories


@click.command()
@click.option('-t', '--template',
              default=str(paths.MODEL_SPEC_DIR / 'model_spec.in'),
              show_default=True,
              type=click.Path(exists=True, dir_okay=False),
              help='The model specification template file.')
@click.option('-l', '--location',
              default='all',
              show_default=True,
              type=click.Choice(project_globals.LOCATIONS + ['all']),
              help='Location to make specification for. Specify locations in globals.py')
@click.option('-o', '--output-dir',
              default=str(paths.MODEL_SPEC_DIR),
              show_default=True,
              type=click.Path(exists=True),
              help='Specify an output directory. Directory must exist.')
@click.option('-v', 'verbose',
              count=True,
              help='Configure logging verbosity.')
@click.option('--pdb', 'with_debugger',
              is_flag=True,
              help='Drop into python debugger if an error occurs.')
def make_specs(template: str, location: str, output_dir: str, verbose: int, with_debugger: bool) -> None:
    """
    Make model specifications

    click application that takes a template model specification file
    and locations for which to create model specs and uses jinja2 to
    render model specs with the correct location parameters plugged in.

    It will look for the model spec template in "model_spec.in" in the directory
    ``src/vivarium_csu_zenon/model_specifications``.
    Add location strings to the ``src/globals.py`` file. By default, specifications
    for all locations will be built. You can choose to make a model specification
    for a single location by specifying that location. However, the location
    string must exist in the list in ``src/globals.py``.

    The application will look for the model spec based on the python environment
    that is active and these files don't need to be specified if the
    default names and location are used.
    """
    configure_logging_to_terminal(verbose)
    main = handle_exceptions(build_model_specifications, logger, with_debugger=with_debugger)
    main(template, location, output_dir)


@click.command()
@click.option('-l', '--location',
              default='all',
              show_default=True,
              type=click.Choice(project_globals.LOCATIONS + ['all']),
              help='Location to make artifact for.')
@click.option('-o', '--output-dir',
              default=str(paths.ARTIFACT_ROOT),
              show_default=True,
              type=click.Path(exists=True),
              help='Specify an output directory. Directory must exist.')
@click.option('-a', '--append',
              is_flag=True,
              help='Append to the artifact instead of overwriting.')
@click.option('-v', 'verbose',
              count=True,
              help='Configure logging verbosity.')
@click.option('--pdb', 'with_debugger',
              is_flag=True,
              help='Drop into python debugger if an error occurs.')
def make_artifacts(location: str, output_dir: str, append: bool, verbose: int, with_debugger: bool) -> None:
    configure_logging_to_terminal(verbose)
    main = handle_exceptions(build_artifacts, logger, with_debugger=with_debugger)
    main(location, output_dir, append, verbose)


@click.command()
@click.argument('output_file', type=click.Path(exists=True))
@click.option('-v', 'verbose',
              count=True,
              help='Configure logging verbosity.')
@click.option('--pdb', 'with_debugger',
              is_flag=True,
              help='Drop into python debugger if an error occurs.')
def make_results(output_file: str, verbose: int, with_debugger: bool) -> None:
    configure_logging_to_terminal(verbose)
    main = handle_exceptions(build_results, logger, with_debugger=with_debugger)
    main(output_file)


@click.command()
@click.option('-l', '--location',
              default='all',
              show_default=True,
              type=click.Choice(project_globals.LOCATIONS + ['all']),
              help='Location to make fpg thresholds for.')
@click.option('-d', '--draws',
              default='all',
              show_default=True,
              help='Comma separated list of draws to make fpg thresholds for')
@click.option('-c', 'concat_only',
              is_flag=True,
              help='Only concatenate existing draws')
@click.option('-v', 'verbose',
              count=True,
              help='Configure logging verbosity.')
@click.option('--pdb', 'with_debugger',
              is_flag=True,
              help='Drop into python debugger if an error occurs.')
def make_fpg_exposure_thresholds(location: str, draws: str, concat_only: bool, verbose: int,
                                 with_debugger: bool) -> None:
    configure_logging_to_terminal(verbose)
    main = handle_exceptions(build_fpg_thresholds, logger, with_debugger=with_debugger)
    main(location, draws, concat_only, verbose)


@click.command()
@click.option('-l', '--location',
              default='all',
              show_default=True,
              type=click.Choice(project_globals.LOCATIONS + ['all']),
              help='Location to make fpg thresholds for.')
@click.option('-d', '--draws',
              default='all',
              show_default=True,
              help='Comma separated list of draws to make fpg thresholds for')
@click.option('-c', 'concat_only',
              is_flag=True,
              help='Only concatenate existing draws')
@click.option('-v', 'verbose',
              count=True,
              help='Configure logging verbosity.')
@click.option('--pdb', 'with_debugger',
              is_flag=True,
              help='Drop into python debugger if an error occurs.')
def make_ldl_exposure_thresholds(location: str, draws: str, concat_only: bool, verbose: int,
                                 with_debugger: bool) -> None:
    configure_logging_to_terminal(verbose)
    main = handle_exceptions(build_ldl_thresholds, logger, with_debugger=with_debugger)
    main(location, draws, concat_only, verbose)


@click.command()
@click.option('-l', '--location',
              default='all',
              show_default=True,
              type=click.Choice(project_globals.LOCATIONS + ['all']),
              help='Location to make joint pafs for.')
@click.option('-d', '--draws',
              default='all',
              show_default=True,
              help='Comma separated list of draws to make fpg thresholds for. None if post-processing only')
@click.option('-q', '--queue',
              default=project_globals.ALL_QUEUE,
              show_default=True,
              type=click.Choice(project_globals.CLUSTER_QUEUES),
              help='Queue to run calculations on.')
@click.option('-v', 'verbose',
              count=True,
              help='Configure logging verbosity.')
@click.option('--pdb', 'with_debugger',
              is_flag=True,
              help='Drop into python debugger if an error occurs.')
def make_joint_pafs(location: str, draws: str, verbose: int, with_debugger: bool, queue: str) -> None:
    configure_logging_to_terminal(verbose)
    main = handle_exceptions(build_joint_pafs, logger, with_debugger=with_debugger)
    main(location, draws, verbose, queue)


@click.command()
@click.argument('worker_log_directory', type=click.Path(exists=True, file_okay=False))
@click.argument('output_file', type=click.Path(exists=False, dir_okay=False))
@click.option('-v', 'verbose',
              count=True,
              help='Configure logging verbosity.')
@click.option('--pdb', 'with_debugger',
              is_flag=True,
              help='Drop into python debugger if an error occurs.')
def parse_logs(worker_log_directory, output_file, verbose, with_debugger):
    configure_logging_to_terminal(verbose)
    main = handle_exceptions(build_results_from_logs, logger, with_debugger=with_debugger)
    main(worker_log_directory, output_file)


@click.command()
@click.option('-l', '--location',
              default='all',
              show_default=True,
              type=click.Choice(project_globals.LOCATIONS + ['all']),
              help='Location to make sample histories for.')
@click.option('-s', '--scenarios',
              default='all',
              show_default=True,
              type=click.Choice(project_globals.SCENARIOS + ['all']),
              help='Scenarios to make sample histories for')
@click.option('-q', '--queue',
              default=project_globals.ALL_QUEUE,
              show_default=True,
              type=click.Choice(project_globals.CLUSTER_QUEUES),
              help='Queue to run calculations on.')
@click.option('-v', 'verbose',
              count=True,
              help='Configure logging verbosity.')
@click.option('--pdb', 'with_debugger',
              is_flag=True,
              help='Drop into python debugger if an error occurs.')
def make_sample_histories(location: str, scenarios: str, verbose: int, with_debugger: bool, queue: str) -> None:
    configure_logging_to_terminal(verbose)
    main = handle_exceptions(build_sample_histories, logger, with_debugger=with_debugger)
    main(location, scenarios, verbose, queue)
