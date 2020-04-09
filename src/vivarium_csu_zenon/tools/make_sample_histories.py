from pathlib import Path
import shutil
import sys
import time
from typing import Union, Dict, List

from loguru import logger
from vivarium import InteractiveContext

from vivarium_csu_zenon import paths, globals as project_globals
from vivarium_csu_zenon.components import SampleHistoryObserver
from vivarium_csu_zenon.tools import decode_status
from vivarium_csu_zenon.utilities import sanitize_location


def build_sample_histories(location: str, scenario: str, verbose: int, queue: str):
    output_dir = paths.SAMPLE_HISTORY_ROOT
    locations = project_globals.LOCATIONS if location == 'all' else [location]
    scenarios = project_globals.SCENARIOS if scenario == 'all' else [scenario]

    from vivarium_cluster_tools.psimulate.utilities import get_drmaa
    drmaa = get_drmaa()
    jobs = {}
    with drmaa.Session() as session:
        for location in locations:
            make_sample_history_single_location(drmaa, queue, jobs, location, scenarios, output_dir, session)

        if verbose:
            logger.info('Entering monitoring loop.')
            logger.info('-------------------------')
            logger.info('')

            while any([job[1] not in [drmaa.JobState.DONE, drmaa.JobState.FAILED] for job in jobs.values()]):
                for location, (job_id, status) in jobs.items():
                    jobs[location] = (job_id, session.jobStatus(job_id))
                    logger.info(f'{location:<35}: {decode_status(drmaa, jobs[location][1]):>15}')
                logger.info('')
                time.sleep(project_globals.MAKE_ARTIFACT_SLEEP)
                logger.info('Checking status again')
                logger.info('---------------------')
                logger.info('')

    logger.info('**Done**')


def make_sample_history_single_location(drmaa, queue: str, jobs: Dict, location: str, scenarios: List[str],
                                        output_dir: Path, session):
    sanitized_location = sanitize_location(location)
    path = output_dir / sanitized_location
    if not path.exists():
        path.mkdir(exist_ok=True, mode=0o775)

    for scenario in scenarios:
        job_template = session.createJobTemplate()
        job_template.remoteCommand = shutil.which("python")
        job_template.outputPath = f":{path}/output_logs"
        job_template.errorPath = f":{path}/error_logs"
        job_template.args = [__file__, str(path), f'"{location}"', scenario]
        job_template.nativeSpecification = (f'-V '  # Export all environment variables
                                            f'-b y '  # Command is a binary (python)
                                            f'-P {project_globals.CLUSTER_PROJECT} '
                                            f'-q {queue} '
                                            f'-l fmem={project_globals.MAKE_ARTIFACT_MEM} '
                                            f'-l fthread={project_globals.MAKE_ARTIFACT_CPU} '
                                            f'-l h_rt={project_globals.MAKE_ARTIFACT_RUNTIME} '
                                            f'-l archive=TRUE '  # Need J-drive access for data
                                            f'-N {sanitize_location(location)}_{scenario}_sample_history')  # Job name
        jobs[location] = (session.runJob(job_template), drmaa.JobState.UNDETERMINED)
        logger.info(f'Submitted job {jobs[location][0]} to generate sample history '
                    f'for {location} and scenario {scenario}.')
        session.deleteJobTemplate(job_template)


def build_sample_history_single_scenario(output_path: Union[str, Path], location: str, scenario: int):
    # NOTE: This is 100% necessary or the qsub will fail
    location = location.strip('"')
    output_path = Path(output_path)

    sim = InteractiveContext(paths.MODEL_SPEC_DIR / f'{location}.yaml', setup=False)
    sim.add_components([SampleHistoryObserver()])
    sim.configuration.update({
        'ldlc_treatment_algorithm': {
            'scenario': scenario
        },
        'metrics': {
            'disability': {
                'by_age': False,
                'by_sex': False,
                'by_year': False,
            },
            'mortality': {
                'by_age': False,
                'by_sex': False,
                'by_year': False,
            },
            'ischemic_heart_disease_observer': {
                'by_age': False,
                'by_sex': False,
                'by_year': False,
            },
            'ischemic_stroke_observer': {
                'by_age': False,
                'by_sex': False,
                'by_year': False,
            },
            'diabetes_mellitus_observer': {
                'by_age': False,
                'by_sex': False,
                'by_year': False,
            },
            'chronic_kidney_disease_observer': {
                'by_age': False,
                'by_sex': False,
                'by_year': False,
            },
            'miscellaneous_observer': {
                'by_age': False,
                'by_sex': False,
                'by_year': False,
            },
            'sample_history_observer': {
                'path': f'{output_path}/{scenario}_sample_history.hdf'
            },
        },
    })
    sim.setup()
    sim.run()
    sim.finalize()

    logger.info('**DONE**')


if __name__ == "__main__":
    sample_history_output_path = sys.argv[1]
    sample_history_location = sys.argv[2]
    sample_history_scenario = sys.argv[3]
    build_sample_history_single_scenario(sample_history_output_path, sample_history_location, sample_history_scenario)
