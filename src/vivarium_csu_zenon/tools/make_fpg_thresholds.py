"""Main application functions for building artifacts.

.. admonition::

   Logging in this module should typically be done at the ``info`` level.
   Use your best judgement.

"""
from pathlib import Path
import shutil
import sys
import time
from time import time
from typing import Union, List, Tuple, Dict

import click
from loguru import logger
import pandas as pd
from scipy.optimize import minimize, Bounds

from risk_distributions import EnsembleDistribution
from vivarium_inputs import interface
from gbd_mapping import risk_factors

from vivarium_csu_zenon import globals as project_globals
from vivarium_csu_zenon.utilities import sanitize_location


def build_fpg_thresholds(output_dir: Path, verbose: int):
    """Builds artifacts for all locations in parallel.
    Parameters
    ----------
    output_dir
        The directory where the artifacts will be built.
    verbose
        How noisy the logger should be.
    Note
    ----
        This function should not be called directly.  It is intended to be
        called by the :func:`build_artifacts` function located in the same
        module.
    """
    from vivarium_cluster_tools.psimulate.utilities import get_drmaa
    drmaa = get_drmaa()

    jobs = {}
    with drmaa.Session() as session:
        for location in project_globals.LOCATIONS:
            path = output_dir / f'{sanitize_location(location)}'
            if path.exists():
                shutil.rmtree(path)
            path.mkdir(exist_ok=True, mode=0o775)

            # TODO uncomment
            # for draw in range(1000):
            for draw in range(10):
                job_template = session.createJobTemplate()
                job_template.remoteCommand = shutil.which("python")
                job_template.args = [__file__, str(path), f'"{location}"', draw]
                job_template.nativeSpecification = (f'-V '  # Export all environment variables
                                                    f'-b y '  # Command is a binary (python)
                                                    f'-P {project_globals.CLUSTER_PROJECT} '  
                                                    f'-q {project_globals.CLUSTER_QUEUE} '  
                                                    f'-l fmem={project_globals.MAKE_ARTIFACT_MEM} '
                                                    f'-l fthread={project_globals.MAKE_ARTIFACT_CPU} '
                                                    f'-l h_rt={project_globals.MAKE_ARTIFACT_RUNTIME} '
                                                    f'-l archive=TRUE '  # Need J-drive access for data
                                                    f'-N {sanitize_location(location)}_artifact')  # Name of the job
                jobs[location] = (session.runJob(job_template), drmaa.JobState.UNDETERMINED)
                logger.info(f'Submitted job {jobs[location][0]} to build fpg threshold for {location} and draw {draw}.')
                session.deleteJobTemplate(job_template)

        decodestatus = {drmaa.JobState.UNDETERMINED: 'undetermined',
                        drmaa.JobState.QUEUED_ACTIVE: 'queued_active',
                        drmaa.JobState.SYSTEM_ON_HOLD: 'system_hold',
                        drmaa.JobState.USER_ON_HOLD: 'user_hold',
                        drmaa.JobState.USER_SYSTEM_ON_HOLD: 'user_system_hold',
                        drmaa.JobState.RUNNING: 'running',
                        drmaa.JobState.SYSTEM_SUSPENDED: 'system_suspended',
                        drmaa.JobState.USER_SUSPENDED: 'user_suspended',
                        drmaa.JobState.DONE: 'finished',
                        drmaa.JobState.FAILED: 'failed'}

        if verbose:
            logger.info('Entering monitoring loop.')
            logger.info('-------------------------')
            logger.info('')

            while any([job[1] not in [drmaa.JobState.DONE, drmaa.JobState.FAILED] for job in jobs.values()]):
                for location, (job_id, status) in jobs.items():
                    jobs[location] = (job_id, session.jobStatus(job_id))
                    logger.info(f'{location:<35}: {decodestatus[jobs[location][1]]:>15}')
                logger.info('')
                time.sleep(project_globals.MAKE_ARTIFACT_SLEEP)
                logger.info('Checking status again')
                logger.info('---------------------')
                logger.info('')

    for location in project_globals.LOCATIONS:
        sanitized_location = f'{sanitize_location(location)}'
        path = output_dir / sanitized_location
        threshold_data = pd.concat([pd.read_hdf(file) for file in path.iterdir()], axis=1)
        threshold_data.to_hdf(output_dir / sanitized_location)
        shutil.rmtree(path)

    logger.info('**Done**')


def build_single_location_single_draw(path: Union[str, Path], location: str, draw: int):
    """Builds an artifact for a single location.
    Parameters
    ----------
    path
        The full path to the artifact to build.
    location
        The location to build the artifact for.  Must be one of the locations
        specified in the project globals.
    log_to_file
        Whether we should write the application logs to a file.
    Note
    ----
        This function should not be called directly.  It is intended to be
        called by the :func:`build_artifacts` function located in the same
        module.
    """
    location = location.strip('"')
    path = Path(path)

    stratifications, means, sds, weights = _get_fpg_exposure_data(location)
    thresholds = _get_thresholds(stratifications, means, sds, weights, draw)
    thresholds.to_hdf(path / f'draw_{draw}.hdf', 'data')

    logger.info('**DONE**')


def _get_fpg_exposure_data(location: int) -> Tuple[List[Tuple], pd.DataFrame, pd.DataFrame, List[Dict[str, float]]]:
    means = (
        interface.get_measure(risk_factors.high_fasting_plasma_glucose_continuous, 'exposure', location)
        .reset_index()
        .drop(['parameter'], axis=1)
        .set_index(['location', 'sex', 'age_start', 'age_end', 'year_start', 'year_end'])
    )

    sds = interface.get_measure(risk_factors.high_fasting_plasma_glucose_continuous,
                                'exposure_standard_deviation', location)

    stratifications = [row[0] for row in list(means.draw_0.iteritems())]

    weights = interface.get_measure(risk_factors.high_fasting_plasma_glucose_continuous,
                                    'exposure_distribution_weights', location)
    return stratifications, means, sds, weights


def _get_thresholds(stratifications: List[Tuple], means: pd.DataFrame, sds: pd.DataFrame,
                    weights_df: pd.DataFrame, draw: int) -> pd.Series:
    col = f'draw_{draw}'
    thresholds = pd.Series(0, index=means.index, name=col)

    ts = time()
    print(f'Start: {ts}')

    # TODO uncomment
    # for i, stratification in enumerate(stratifications):
    for i, stratification in enumerate(stratifications[-10:]):
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
            threshold = minimize(lambda x: (ens_dist.ppf(x) - 7) ** 2, [0.5],
                                 bounds=Bounds(0, 1.0), method='Nelder-Mead').x[0]

        print(f'mu: {mu}, sigma: {sigma}, threshold: {threshold}')
        thresholds.loc[stratification] = threshold

    tf = time()
    print(f'End: {tf}')
    print(f'Duration: {tf - ts}')

    return thresholds


if __name__ == "__main__":
    threshold_data_path = sys.argv[1]
    threshold_location = sys.argv[2]
    threshold_draw = sys.argv[3]
    build_single_location_single_draw(threshold_data_path, threshold_location, threshold_draw)
