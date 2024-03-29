from pathlib import Path
import shutil
import sys
import time
from typing import Union, Dict

import numpy as np
import pandas as pd
import scipy.stats
from loguru import logger

from gbd_mapping import risk_factors, causes
from risk_distributions import EnsembleDistribution

from vivarium_csu_zenon import paths, globals as project_globals
from vivarium_csu_zenon.tools import decode_status
from vivarium_csu_zenon.utilities import sanitize_location

CONTINUOUS_RISKS = [risk_factors.high_fasting_plasma_glucose_continuous,
                    risk_factors.high_systolic_blood_pressure,
                    risk_factors.high_ldl_cholesterol]
ALL_RISKS = CONTINUOUS_RISKS + [risk_factors.impaired_kidney_function]
AFFECTED_CAUSES = [causes.ischemic_heart_disease.name, causes.ischemic_stroke.name]
INDEX_COLS = ['location', 'sex', 'age_start', 'age_end', 'year_start', 'year_end']


def build_joint_pafs(location: str, draws: str, verbose: int, queue: str):
    # Local import to avoid data dependencies
    from vivarium_inputs import globals as vi_globals, utilities

    output_dir = paths.JOINT_PAF_DIR
    locations = project_globals.LOCATIONS if location == 'all' else [location]

    from vivarium_cluster_tools.psimulate.utilities import get_drmaa
    drmaa = get_drmaa()
    jobs = {}
    draw_list = {
        'all': range(1000),
        'none': []
    }.get(draws, draws.split(','))
    with drmaa.Session() as session:
        for location in locations:
            build_joint_pafs_single_location(drmaa, queue, jobs, location, draw_list, output_dir, session)

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
        logger.info(f'Merging data for location - {location}')
        sanitized_location = sanitize_location(location)
        location_dir = paths.JOINT_PAF_DIR / sanitized_location

        existing_data_path = output_dir / f'{sanitized_location}.hdf'
        joint_pafs = []
        if existing_data_path.exists():
            joint_pafs.append(pd.read_hdf(output_dir / f'{sanitized_location}.hdf'))
            joint_pafs[0].to_hdf(output_dir / f'{sanitized_location}-old.hdf', 'data')

        for file_path in location_dir.iterdir():
            draw = file_path.parts[-1].split('.')[0]
            draw_joint_paf = pd.read_hdf(file_path).rename(columns={0: draw})
            draw_joint_paf['affected_measure'] = 'incidence_rate'
            draw_joint_paf = draw_joint_paf.set_index(list(draw_joint_paf.columns.drop(draw)))
            joint_pafs.append(draw_joint_paf)

        joint_paf_data = pd.concat(joint_pafs, axis=1)
        joint_paf_data = joint_paf_data[vi_globals.DRAW_COLUMNS]  # sort the columns
        joint_paf_data = utilities.sort_hierarchical_data(joint_paf_data).convert_objects()
        joint_paf_data.to_hdf(output_dir / f'{sanitized_location}.hdf', 'data')
        shutil.rmtree(location_dir)

    logger.info('**Done**')


def build_joint_pafs_single_location(drmaa, queue: str, jobs: Dict, location: str, draws, output_dir: Path, session):
    sanitized_location = sanitize_location(location)
    path = output_dir / sanitized_location
    if path.exists() and len(draws) == 1000:
        shutil.rmtree(path)
    path.mkdir(exist_ok=True, mode=0o775)

    for draw in draws:
        job_template = session.createJobTemplate()
        job_template.remoteCommand = shutil.which("python")
        job_template.outputPath = f":{output_dir}/output_logs"
        job_template.errorPath = f":{output_dir}/error_logs"
        job_template.args = [__file__, str(path), f'"{location}"', draw]
        job_template.nativeSpecification = (f'-V '  # Export all environment variables
                                            f'-b y '  # Command is a binary (python)
                                            f'-P {project_globals.CLUSTER_PROJECT} '
                                            f'-q {queue} '
                                            f'-l fmem={project_globals.MAKE_ARTIFACT_MEM} '
                                            f'-l fthread={project_globals.MAKE_ARTIFACT_CPU} '
                                            f'-l h_rt={project_globals.MAKE_ARTIFACT_RUNTIME} '
                                            f'-l archive=TRUE '  # Need J-drive access for data
                                            f'-N {sanitize_location(location)}_joint_pafs')  # Name of the job
        jobs[location] = (session.runJob(job_template), drmaa.JobState.UNDETERMINED)
        logger.info(f'Submitted job {jobs[location][0]} to build joint pafs for {location} and draw {draw}.')
        session.deleteJobTemplate(job_template)


def build_joint_paf_single_draw(output_path: Union[str, Path], location: str, draw_number: int):
    # NOTE: This is 100% necessary or the qsub will fail
    location = location.strip('"')
    output_path = Path(output_path)

    correlation_data = load_correlation_data()
    exposure_data = {risk.name: load_exposure_data(risk, location) for risk in ALL_RISKS}
    rr_data = {risk.name: load_relative_risk_data(risk, location) for risk in ALL_RISKS}

    canonical_index = exposure_data[risk_factors.high_ldl_cholesterol.name].mean.index

    joint_pafs = pd.DataFrame(index=canonical_index, columns=AFFECTED_CAUSES)
    joint_pafs.columns.name = 'affected_entity'

    for stratification in canonical_index.tolist():
        corr = get_correlation(correlation_data, stratification)

        # monte carlo integration error goes down as 1/sqrt(sample_size)
        sample_size = 100000
        propensities = pd.DataFrame(
            scipy.stats.norm().cdf(
                scipy.stats.multivariate_normal(cov=corr).rvs(sample_size)
            ), columns=corr.columns)
        samples = pd.DataFrame(columns=propensities.columns, index=propensities.index)

        risk_dists = {risk_name: get_dist(dist_params, stratification, f'draw_{draw_number}')
                      for risk_name, dist_params in exposure_data.items()}

        for risk, dist in risk_dists.items():
            samples.loc[:, risk] = dist.ppf(propensities.loc[:, risk]) if dist else 0.0

        # Drop nan rows
        samples = samples[samples.isnull().sum(axis=1) == 0].reset_index(drop=True)

        for cause in AFFECTED_CAUSES:
            rrs = {risk_name: get_rr(rr_params, cause, stratification, f'draw_{draw_number}')
                   for risk_name, rr_params in rr_data.items()}
            total_rr = 1
            for risk in ALL_RISKS:
                total_rr *= rrs[risk.name](samples[risk.name])

            mean_rr = total_rr.sum() / len(samples)
            joint_pafs.loc[stratification, cause] = (mean_rr - 1) / mean_rr

    (joint_pafs.stack()
     .reset_index()
     .rename(columns={draw_number: f'draw_{draw_number}'})
     .to_hdf(output_path / f'draw_{draw_number}.hdf', 'data'))
    logger.info('**DONE**')


# Data Getters
# Correlation requires you to be on the latest branch of zenon until we do some pr review.

def load_correlation_data():
    # Local import to avoid data dependencies
    from vivarium_csu_zenon.data.loader import load_propensity_correlation_data

    raw_data = load_propensity_correlation_data('', '').reset_index()
    data = {}
    for label, group in raw_data.groupby(['age_group']):
        group_matrix = group.set_index(['risk_a', 'risk_b']).value.unstack()
        group_matrix.index.name = None
        group_matrix.columns.name = None
        age_start, age_end = [int(age) for age in label.split('_to_')]
        age_start = 0 if age_start == 30 else age_start
        age_end = 125 if age_end == 79 else age_end + 1
        data_key = (age_start, age_end)
        data[data_key] = group_matrix
    return data


class DistParams:
    def __init__(self, risk, location):
        # Local import to avoid data dependencies
        from vivarium_inputs.interface import get_measure

        self.mean = get_measure(risk, 'exposure', location).droplevel('parameter')
        self.sd = get_measure(risk, 'exposure_standard_deviation', location)
        self.weights = get_measure(risk, 'exposure_distribution_weights', location)


def load_exposure_data(risk, location):
    # Local import to avoid data dependencies
    from vivarium_csu_zenon.data.loader import load_ikf_exposure

    if risk.name == risk_factors.impaired_kidney_function.name:
        ikf = load_ikf_exposure(project_globals.IKF.EXPOSURE, location)
        return ikf
    return DistParams(risk, location)


class RelativeRiskParams:
    def __init__(self, risk, location):
        # Local import to avoid data dependencies
        from vivarium_inputs.interface import get_measure
        rr = get_measure(risk, 'relative_risk', location).droplevel(['affected_measure', 'parameter'])
        self.base = (
            rr.reorder_levels(['affected_entity'] + INDEX_COLS)
            .loc[AFFECTED_CAUSES]
        )
        self.tmrel = np.mean([risk.tmred.min, risk.tmred.max])
        self.scale = float(risk.relative_risk_scalar)


def load_relative_risk_data(risk, location):
    # Local import to avoid data dependencies
    from vivarium_csu_zenon.data.loader import load_ikf_relative_risk

    if risk.name == risk_factors.impaired_kidney_function.name:
        ikf_rr = load_ikf_relative_risk(project_globals.IKF.RELATIVE_RISK, location).droplevel('affected_measure')
        ikf_rr = (
            ikf_rr.reorder_levels(['affected_entity'] + INDEX_COLS + ['parameter'])
            .loc[AFFECTED_CAUSES]
        )
        return ikf_rr
    return RelativeRiskParams(risk, location)


# Stratifiers

def get_correlation(correlation_data, stratification):
    age_start = stratification[2]
    key = [k for k in correlation_data if k[0] <= age_start < k[1]].pop()
    corr = correlation_data[key]
    corr.columns = corr.columns.map(lambda x: x.split('_propensity')[0])
    corr.index = corr.index.map(lambda x: x.split('_propensity')[0])
    return corr


class IKFDist:

    def __init__(self, pmf):
        # this is done because sometimes the values were 0 causing cut to fail
        pmf = pmf[pmf > 1e-8]
        self.bins = [0] + pmf.cumsum().tolist()
        self.labels = pmf.index

    def ppf(self, propensity):
        return pd.cut(propensity, bins=self.bins, labels=self.labels).astype(object)


def get_dist(dist_params, stratification, draw):
    if isinstance(dist_params, pd.DataFrame):
        return IKFDist(dist_params.loc[stratification, draw])

    mu = dist_params.mean.loc[stratification, draw]
    sigma = dist_params.sd.loc[stratification, draw]
    if mu and sigma:
        weights = dist_params.weights.loc[stratification].reset_index()
        weights = (
            weights[weights['parameter'] != 'glnorm'].loc[:, ['parameter', 'value']]
            .set_index('parameter')
            .to_dict()['value']
        )
        weights = {k: [v] for k, v in weights.items()}
        return EnsembleDistribution(weights=weights, mean=mu, sd=sigma)
    else:
        return None


class IKFRelativeRisk:
    def __init__(self, rr_data):
        self.rr_data = rr_data

    def __call__(self, exposure):
        return pd.Series(self.rr_data.loc[exposure].values, index=exposure.index)


class ContinuousRelativeRisk:
    def __init__(self, base, tmrel, scale):
        self.base = base
        self.tmrel = tmrel
        self.scale = scale

    def __call__(self, exposure):
        rr = self.base ** ((exposure - self.tmrel) / self.scale)
        return np.maximum(rr, 1)


def get_rr(rr_params, affected_cause, stratification, draw):
    stratification = (affected_cause,) + stratification
    if isinstance(rr_params, pd.DataFrame):
        return IKFRelativeRisk(rr_params.loc[stratification, draw])
    else:
        return ContinuousRelativeRisk(rr_params.base.loc[stratification, draw], rr_params.tmrel, rr_params.scale)


if __name__ == "__main__":
    joint_paf_output_path = sys.argv[1]
    joint_paf_location = sys.argv[2]
    joint_paf_draw = sys.argv[3]
    build_joint_paf_single_draw(joint_paf_output_path, joint_paf_location, joint_paf_draw)
