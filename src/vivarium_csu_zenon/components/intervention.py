from typing import List

import pandas as pd

from vivarium.framework.randomness import get_hash

from vivarium_csu_zenon import paths
from vivarium_csu_zenon.utilities import sample_truncnorm_distribution


LOCATION_COLUMN = 'location'
MEAN_COLUMN = 'mean_value'
SD_COLUMN = 'sd_value'

INCREASE_DOSE = 'increasing_dose'
ADD_SECOND_DRUG = 'adding_2nd_drug'
SWITCH_DRUG = 'switching_drugs'


def sample_probability_testing_ldl_c(location: str, draw: int) -> float:
    seed = get_hash(f'testing_ldl_c_probability_draw_{draw}_location_{location}')
    data = pd.read_csv(paths.PROB_TESTING_LDL_C_PATH).set_index(LOCATION_COLUMN)
    params = data.loc[location, :]
    return sample_truncnorm_distribution(seed, params[MEAN_COLUMN], params[SD_COLUMN])


def sample_probability_rx_given_high_ldl_c(location: str, draw: int) -> float:
    seed = get_hash(f'rx_given_high_ldl_c_probability_draw_{draw}_location_{location}')
    data = pd.read_csv(paths.PROB_RX_GIVEN_HIGH_LDL_C).set_index(LOCATION_COLUMN)
    params = data.loc[location, :]
    return sample_truncnorm_distribution(seed, params[MEAN_COLUMN], params[SD_COLUMN])


def sample_probability_target_given_rx(location: str, draw: int) -> float:
    seed = get_hash(f'target_given_rx_probability_draw_{draw}_location_{location}')
    data = pd.read_csv(paths.PROB_TARGET_GIVEN_RX).set_index(LOCATION_COLUMN)
    params = data.loc[location, :]
    return sample_truncnorm_distribution(seed, params[MEAN_COLUMN], params[SD_COLUMN])


def sample_adherence(location: str, draw: int, multi_pill: bool, previous_cve: bool) -> float:
    seed = get_hash(f'adherence_probability_draw_{draw}_location_{location}')
    data = pd.read_csv(paths.ADHERENCE_PARAMETERS).set_index([LOCATION_COLUMN, 'multi_pill', 'previous_cve'])
    params = data.loc[(location, int(multi_pill), int(previous_cve)), :]
    return sample_truncnorm_distribution(seed, params[MEAN_COLUMN], params[SD_COLUMN])


def sample_raw_rx_change(location: str, draw: int, rx_change: str) -> float:
    """Raw result: needs to be adjusted"""
    seed = get_hash(f'{rx_change}_probability_draw_{draw}_location_{location}')
    data = pd.read_csv(paths.PROB_ADDING_DRUGS).set_index('probability_type')
    params = data.loc[rx_change, :]
    return sample_truncnorm_distribution(seed, params[MEAN_COLUMN], params[SD_COLUMN])


def sample_probability_increasing_dose(location: str, draw: int) -> float:
    seed = get_hash(f'target_given_rx_probability_draw_{draw}_location_{location}')
    data = pd.read_csv(paths.PROB_TARGET_GIVEN_RX).set_index(LOCATION_COLUMN)
    params = data.loc[location, :]
    return sample_truncnorm_distribution(seed, params[MEAN_COLUMN], params[SD_COLUMN])


def sample_raw_drug_prescription(location: str, draw: int, drug: str) -> float:
    """Raw result: needs to be adjusted"""
    seed = get_hash(f'{drug}_prescription_probability_draw_{draw}_location_{location}')
    data = pd.read_csv(paths.CURRENT_RX_DATA_PATH).set_index([LOCATION_COLUMN, 'current_prescription'])
    params = data.loc[(location, drug), :]
    return sample_truncnorm_distribution(seed, params[MEAN_COLUMN], params[SD_COLUMN])


def get_adjusted_probabilities(*drug_probabilities: float) -> List[float]:
    """Use on sets of raw results"""
    scaling_factor = sum(drug_probabilities)
    return [drug_probability / scaling_factor for drug_probability in drug_probabilities]