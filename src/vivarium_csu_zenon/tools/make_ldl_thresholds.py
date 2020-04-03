from pathlib import Path
import shutil
import sys
import time
from typing import Union, List, Tuple, Dict

from loguru import logger
import pandas as pd
from scipy.optimize import minimize, Bounds

from risk_distributions import EnsembleDistribution
from vivarium_inputs import interface
from gbd_mapping import risk_factors

from vivarium_csu_zenon import paths, globals as project_globals
from vivarium_csu_zenon.tools.app_logging import decode_status
from vivarium_csu_zenon.components.treatment.parameters import HIGH_LDL_BASELINE
from vivarium_csu_zenon.utilities import sanitize_location


def build_ldl_thresholds(location: str, draws: str, concat_only: bool, verbose: int):
    output_dir = paths.LDL_C_THRESHOLD_DIR
    locations = project_globals.LOCATIONS if location == 'all' else [location]

    if not concat_only:
        from vivarium_cluster_tools.psimulate.utilities import get_drmaa
        drmaa = get_drmaa()

        jobs = {}
        draw_list = range(1000) if draws == 'all' else draws.split(',')

        with drmaa.Session() as session:
            for location in locations:
                build_ldl_thresholds_single_location(drmaa, jobs, location, draw_list, output_dir, session)

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

    for location in locations:
        sanitized_location = f'{sanitize_location(location)}'
        path = output_dir / sanitized_location
        existing_data_path = output_dir / f'{sanitized_location}.hdf'
        existing_data = []
        if existing_data_path.exists():
            existing_data.append(pd.read_hdf(output_dir / f'{sanitized_location}.hdf'))
            existing_data[0].to_hdf(output_dir / f'{sanitized_location}-old.hdf', 'data')
        threshold_data = pd.concat(existing_data + [pd.read_hdf(file) for file in path.iterdir()], axis=1)
        threshold_data.to_hdf(output_dir / f'{sanitized_location}.hdf', 'data')
        shutil.rmtree(path)

    logger.info('**Done**')


def build_ldl_thresholds_single_location(drmaa, jobs, location, draws, output_dir, session):
    path = output_dir / f'{sanitize_location(location)}'
    if path.exists() and len(draws) == 1000:
        shutil.rmtree(path)
    path.mkdir(exist_ok=True, mode=0o775)
    for draw in draws:
        job_template = session.createJobTemplate()
        job_template.remoteCommand = shutil.which("python")
        job_template.args = [__file__, str(path), f'"{location}"', draw]
        job_template.nativeSpecification = (f'-V '  # Export all environment variables
                                            f'-b y '  # Command is a binary (python)
                                            f'-P {project_globals.CLUSTER_PROJECT} '
                                            f'-q {project_globals.ALL_QUEUE} '
                                            f'-l fmem={project_globals.MAKE_ARTIFACT_MEM} '
                                            f'-l fthread={project_globals.MAKE_ARTIFACT_CPU} '
                                            f'-l h_rt={project_globals.MAKE_ARTIFACT_RUNTIME} '
                                            f'-l archive=TRUE '  # Need J-drive access for data
                                            f'-N {sanitize_location(location)}_artifact')  # Name of the job
        jobs[location] = (session.runJob(job_template), drmaa.JobState.UNDETERMINED)
        logger.info(f'Submitted job {jobs[location][0]} to build fpg threshold for {location} and draw {draw}.')
        session.deleteJobTemplate(job_template)


def build_single_location_single_draw(path: Union[str, Path], location: str, draw: int):
    location = location.strip('"')
    path = Path(path)

    stratifications, means, sds, weights = _get_ldl_exposure_data(location)
    thresholds = _get_thresholds(stratifications, means, sds, weights, draw)
    thresholds.to_hdf(path / f'draw_{draw}.hdf', 'data')

    logger.info('**DONE**')


def _get_ldl_exposure_data(location: int) -> Tuple[List[Tuple], pd.DataFrame, pd.DataFrame, List[Dict[str, float]]]:
    means = (
        interface.get_measure(risk_factors.high_ldl_cholesterol, 'exposure', location)
        .reset_index()
        .drop(['parameter'], axis=1)
        .set_index(['location', 'sex', 'age_start', 'age_end', 'year_start', 'year_end'])
    )

    sds = interface.get_measure(risk_factors.high_ldl_cholesterol, 'exposure_standard_deviation', location)
    stratifications = [row[0] for row in list(means.draw_0.iteritems())]
    weights = interface.get_measure(risk_factors.high_ldl_cholesterol, 'exposure_distribution_weights', location)
    return stratifications, means, sds, weights


def _get_thresholds(stratifications: List[Tuple], means: pd.DataFrame, sds: pd.DataFrame,
                    weights_df: pd.DataFrame, draw: int) -> pd.Series:
    col = f'draw_{draw}'
    thresholds = pd.Series(0, index=means.index, name=col)

    ts = time.time()
    print(f'Start: {ts}')

    for i, stratification in enumerate(stratifications):
        mu = means.loc[stratification, col]
        sigma = sds.loc[stratification, col]
        threshold = 0
        if mu and sigma:
            weights = weights_df.loc[stratification].reset_index()
            weights = (weights[weights['parameter'] != 'glnorm']
                       .loc[:, ['parameter', 'value']]
                       .set_index('parameter')
                       .to_dict()['value'])
            weights = {k: [v] for k, v in weights.items()}
            ens_dist = EnsembleDistribution(weights=weights, mean=mu, sd=sigma)
            threshold = minimize(lambda x: (ens_dist.ppf(x) - HIGH_LDL_BASELINE) ** 2, [0.5],
                                 bounds=Bounds(0, 1.0), method='Nelder-Mead').x[0]

        print(f'mu: {mu}, sigma: {sigma}, threshold: {threshold}')
        thresholds.loc[stratification] = threshold

    tf = time.time()
    print(f'End: {tf}')
    print(f'Duration: {tf - ts}')

    return thresholds


if __name__ == "__main__":
    threshold_data_path = sys.argv[1]
    threshold_location = sys.argv[2]
    threshold_draw = sys.argv[3]
    build_single_location_single_draw(threshold_data_path, threshold_location, threshold_draw)
