"""Loads, standardizes and validates input data for the simulation.

Abstract the extract and transform pieces of the artifact ETL.
The intent here is to provide a uniform interface around this portion
of artifact creation. The value of this interface shows up when more
complicated data needs are part of the project. See the BEP project
for an example.

`BEP <https://github.com/ihmeuw/vivarium_gates_bep/blob/master/src/vivarium_gates_bep/data/loader.py>`_

.. admonition::
"""
from loguru import logger
from vivarium_gbd_access import gbd
from gbd_mapping import causes, risk_factors, covariates, sequelae
import pandas as pd
from vivarium.framework.artifact import EntityKey
from vivarium_inputs import interface, utilities, utility_data, globals as vi_globals
from vivarium_inputs.mapping_extension import alternative_risk_factors

from vivarium_csu_zenon import paths, globals as project_globals


def get_data(lookup_key: str, location: str) -> pd.DataFrame:
    """Retrieves data from an appropriate source.

    Parameters
    ----------
    lookup_key
        The key that will eventually get put in the artifact with
        the requested data.
    location
        The location to get data for.

    Returns
    -------
        The requested data.

    """
    mapping = {
        project_globals.POPULATION.STRUCTURE: load_population_structure,
        project_globals.POPULATION.AGE_BINS: load_age_bins,
        project_globals.POPULATION.DEMOGRAPHY: load_demographic_dimensions,
        project_globals.POPULATION.TMRLE: load_theoretical_minimum_risk_life_expectancy,
        project_globals.POPULATION.ACMR: load_standard_data,

        project_globals.IHD.ACUTE_MI_PREVALENCE: load_ihd_prevalence,
        project_globals.IHD.POST_MI_PREVALENCE: load_ihd_prevalence,
        project_globals.IHD.ACUTE_MI_INCIDENCE_RATE: load_standard_data,
        project_globals.IHD.ACUTE_MI_DISABILITY_WEIGHT: load_ihd_disability_weight,
        project_globals.IHD.POST_MI_DISABILITY_WEIGHT: load_ihd_disability_weight,
        project_globals.IHD.ACUTE_MI_EMR: load_excess_mortality_rate,
        project_globals.IHD.POST_MI_EMR: load_excess_mortality_rate,
        project_globals.IHD.CSMR: load_standard_data,
        project_globals.IHD.RESTRICTIONS: load_metadata,

        project_globals.ISCHEMIC_STROKE.ACUTE_STROKE_PREVALENCE: load_ischemic_stroke_prevalence,
        project_globals.ISCHEMIC_STROKE.POST_STROKE_PREVALENCE: load_ischemic_stroke_prevalence,
        project_globals.ISCHEMIC_STROKE.ACUTE_STROKE_INCIDENCE_RATE: load_standard_data,
        project_globals.ISCHEMIC_STROKE.ACUTE_STROKE_DISABILITY_WEIGHT: load_ischemic_stroke_disability_weight,
        project_globals.ISCHEMIC_STROKE.POST_STROKE_DISABILITY_WEIGHT: load_ischemic_stroke_disability_weight,
        project_globals.ISCHEMIC_STROKE.ACUTE_STROKE_EMR: load_excess_mortality_rate,
        project_globals.ISCHEMIC_STROKE.POST_STROKE_EMR: load_excess_mortality_rate,
        project_globals.ISCHEMIC_STROKE.CSMR: load_standard_data,
        project_globals.ISCHEMIC_STROKE.RESTRICTIONS: load_metadata,

        project_globals.DIABETES_MELLITUS.MODERATE_DIABETES_PREVALENCE: load_diabetes_mellitus_prevalence,
        project_globals.DIABETES_MELLITUS.SEVERE_DIABETES_PREVALENCE: load_diabetes_mellitus_prevalence,
        project_globals.DIABETES_MELLITUS.ALL_DIABETES_INCIDENCE_RATE: load_standard_data,
        project_globals.DIABETES_MELLITUS.MODERATE_DIABETES_PROPORTION: load_diabetes_mellitus_incidence_proportion,
        project_globals.DIABETES_MELLITUS.SEVERE_DIABETES_PROPORTION: load_diabetes_mellitus_incidence_proportion,
        project_globals.DIABETES_MELLITUS.MODERATE_DIABETES_DISABILITY_WEIGHT: load_diabetes_mellitus_disability_weight,
        project_globals.DIABETES_MELLITUS.SEVERE_DIABETES_DISABILITY_WEIGHT: load_diabetes_mellitus_disability_weight,
        project_globals.DIABETES_MELLITUS.MODERATE_DIABETES_EMR: load_diabetes_mellitus_excess_mortality_rate,
        project_globals.DIABETES_MELLITUS.SEVERE_DIABETES_EMR: load_diabetes_mellitus_excess_mortality_rate,
        project_globals.DIABETES_MELLITUS.CSMR: load_standard_data,
        project_globals.DIABETES_MELLITUS.RESTRICTIONS: load_metadata,
    }
    return mapping[lookup_key](lookup_key, location)


def load_standard_data(key: str, location: str) -> pd.DataFrame:
    key = EntityKey(key)
    entity = get_entity(key)
    return interface.get_measure(entity, key.measure, location)


def load_metadata(key: str, location: str):
    key = EntityKey(key)
    entity = get_entity(key)
    metadata = entity[key.measure]
    if hasattr(metadata, 'to_dict'):
        metadata = metadata.to_dict()
    return metadata


def load_population_structure(key: str, location: str) -> pd.DataFrame:
    return interface.get_population_structure(location)


def load_age_bins(key: str, location: str) -> pd.DataFrame:
    return interface.get_age_bins()


def load_demographic_dimensions(key: str, location: str) -> pd.DataFrame:
    return interface.get_demographic_dimensions(location)


def load_theoretical_minimum_risk_life_expectancy(key: str, location: str) -> pd.DataFrame:
    return interface.get_theoretical_minimum_risk_life_expectancy()


def load_ihd_prevalence(key: str, location: str) -> pd.DataFrame:
    acute_sequelae = [
        sequelae.acute_myocardial_infarction_first_2_days,
        sequelae.acute_myocardial_infarction_3_to_28_days,
    ]

    if key == project_globals.IHD.ACUTE_MI_PREVALENCE:
        seq = acute_sequelae
    else:
        seq = [s for s in causes.ischemic_heart_disease.sequelae if s not in acute_sequelae]

    prevalence = sum(interface.get_measure(s, 'prevalence', location) for s in seq)
    return prevalence


def load_ischemic_stroke_prevalence(key: str, location: str) -> pd.DataFrame:
    acute_sequelae = [
        sequelae.acute_ischemic_stroke_severity_level_1,
        sequelae.acute_ischemic_stroke_severity_level_2,
        sequelae.acute_ischemic_stroke_severity_level_3,
        sequelae.acute_ischemic_stroke_severity_level_4,
        sequelae.acute_ischemic_stroke_severity_level_5,
    ]

    if key == project_globals.ISCHEMIC_STROKE.ACUTE_STROKE_PREVALENCE:
        seq = acute_sequelae
    else:
        seq = [s for s in causes.ischemic_stroke.sequelae if s not in acute_sequelae]

    prevalence = sum(interface.get_measure(s, 'prevalence', location) for s in seq)
    return prevalence


def load_diabetes_mellitus_prevalence(key: str, location: str) -> pd.DataFrame:
    moderate_sequelae = [
        sequelae.uncomplicated_diabetes_mellitus_type_1,
        sequelae.uncomplicated_diabetes_mellitus_type_2,
    ]

    if key == project_globals.DIABETES_MELLITUS.MODERATE_DIABETES_PREVALENCE:
        seq = moderate_sequelae
    else:
        seq = [s for sc in causes.diabetes_mellitus.sub_causes for s in sc.sequelae if s not in moderate_sequelae]

    prevalence = sum(interface.get_measure(s, 'prevalence', location) for s in seq)
    return prevalence


def load_diabetes_mellitus_incidence_proportion(key: str, location: str) -> pd.DataFrame:
    moderate_sequelae = [
        sequelae.uncomplicated_diabetes_mellitus_type_1,
        sequelae.uncomplicated_diabetes_mellitus_type_2,
    ]

    if key == project_globals.DIABETES_MELLITUS.MODERATE_DIABETES_PREVALENCE:
        seq = moderate_sequelae
    else:
        seq = [s for sc in causes.diabetes_mellitus.sub_causes for s in sc.sequelae if s not in moderate_sequelae]

    # TODO this	is temporarily set to prevalence because of missing incidence rate data
    all_diabetes_incidence_rate = load_standard_data('cause.diabetes_mellitus.prevalence', location)
    sequelae_incidence_rates = []
    for s in seq:
        try:
            # TODO this is temporarily set to prevalence because of missing incidence rate data
            sequelae_incidence_rates.append(interface.get_measure(s, 'prevalence', location))
        except vi_globals.DataDoesNotExistError as e:
            logger.debug(f'There is no incidence data for sequela {s.name}')

    import pdb; pdb.set_trace()
    incidence_proportion = (sum(sequelae_incidence_rates) / all_diabetes_incidence_rate).fillna(0)
    return incidence_proportion


def load_ihd_disability_weight(key: str, location: str) -> pd.DataFrame:
    acute_sequelae = [
        sequelae.acute_myocardial_infarction_first_2_days,
        sequelae.acute_myocardial_infarction_3_to_28_days,
    ]

    if key == project_globals.IHD.ACUTE_MI_PREVALENCE:
        seq = acute_sequelae
    else:
        seq = [s for s in causes.ischemic_heart_disease.sequelae if s not in acute_sequelae]

    prevalence_disability_weights = []
    for s in seq:
        prevalence = interface.get_measure(s, 'prevalence', location)
        disability_weight = interface.get_measure(s, 'disability_weight', location)
        prevalence_disability_weights.append(prevalence * disability_weight)

    ihd_prevalence = interface.get_measure(causes.ischemic_heart_disease, 'prevalence', location)
    ihd_disability_weight = (sum(prevalence_disability_weights) / ihd_prevalence).fillna(0)
    return ihd_disability_weight


def load_ischemic_stroke_disability_weight(key: str, location: str) -> pd.DataFrame:
    acute_sequelae = [
        sequelae.acute_ischemic_stroke_severity_level_1,
        sequelae.acute_ischemic_stroke_severity_level_2,
        sequelae.acute_ischemic_stroke_severity_level_3,
        sequelae.acute_ischemic_stroke_severity_level_4,
        sequelae.acute_ischemic_stroke_severity_level_5,
    ]

    if key == project_globals.IHD.ACUTE_MI_PREVALENCE:
        seq = acute_sequelae
    else:
        seq = [s for s in causes.ischemic_heart_disease.sequelae if s not in acute_sequelae]

    prevalence_disability_weights = []
    for s in seq:
        prevalence = interface.get_measure(s, 'prevalence', location)
        disability_weight = interface.get_measure(s, 'disability_weight', location)
        prevalence_disability_weights.append(prevalence * disability_weight)

    ischemic_stroke_prevalence = interface.get_measure(causes.ischemic_stroke, 'prevalence', location)
    ischemic_stroke_disability_weight = (sum(prevalence_disability_weights) / ischemic_stroke_prevalence).fillna(0)
    return ischemic_stroke_disability_weight


def load_diabetes_mellitus_disability_weight(key: str, location: str) -> pd.DataFrame:
    moderate_sequelae = [
        sequelae.uncomplicated_diabetes_mellitus_type_1,
        sequelae.uncomplicated_diabetes_mellitus_type_2,
    ]

    if key == project_globals.DIABETES_MELLITUS.MODERATE_DIABETES_PREVALENCE:
        seq = moderate_sequelae
    else:
        seq = [s for sc in causes.diabetes_mellitus.sub_causes for s in sc.sequelae if s not in moderate_sequelae]

    prevalence_disability_weights = []
    for s in seq:
        prevalence = interface.get_measure(s, 'prevalence', location)
        disability_weight = interface.get_measure(s, 'disability_weight', location)
        prevalence_disability_weights.append(prevalence * disability_weight)

    diabetes_prevalence = interface.get_measure(causes.diabetes_mellitus, 'prevalence', location)
    diabetes_disability_weight = (sum(prevalence_disability_weights) / diabetes_prevalence).fillna(0)
    return diabetes_disability_weight


def load_diabetes_mellitus_excess_mortality_rate(key: str, location: str) -> pd.DataFrame:
    if key == project_globals.DIABETES_MELLITUS.MODERATE_DIABETES_EMR:
        diabetes_emr = get_data(project_globals.POPULATION.DEMOGRAPHY, location)
        diabetes_emr['value'] = 0
    else:
        raw_diabetes_emr = get_data(project_globals.DIABETES_MELLITUS.CSMR, location)
        prevalence_severe = get_data(project_globals.DIABETES_MELLITUS.SEVERE_DIABETES_PREVALENCE, location)
        diabetes_emr = (raw_diabetes_emr / prevalence_severe).fillna(0)
    return diabetes_emr


def load_excess_mortality_rate(key: str, location: str) -> pd.DataFrame:
    meids = {
        project_globals.IHD.ACUTE_MI_EMR: 1814,
        project_globals.IHD.POST_MI_EMR: 15755,
        project_globals.ISCHEMIC_STROKE.ACUTE_STROKE_EMR: 9310,
        project_globals.ISCHEMIC_STROKE.POST_STROKE_EMR: 10837,
    }

    return _load_em_from_meid(meids[key], location)


def _load_em_from_meid(meid, location):
    location_id = utility_data.get_location_id(location)
    data = gbd.get_modelable_entity_draws(meid, location_id)
    data = data[data.measure_id == vi_globals.MEASURES['Excess mortality rate']]
    data = utilities.normalize(data, fill_value=0)
    data = data.filter(vi_globals.DEMOGRAPHIC_COLUMNS + vi_globals.DRAW_COLUMNS)
    data = utilities.reshape(data)
    data = utilities.scrub_gbd_conventions(data, location)
    data = utilities.split_interval(data, interval_column='age', split_column_prefix='age')
    data = utilities.split_interval(data, interval_column='year', split_column_prefix='year')
    return utilities.sort_hierarchical_data(data)


def get_entity(key: str):
    # Map of entity types to their gbd mappings.
    type_map = {
        'cause': causes,
        'covariate': covariates,
        'risk_factor': risk_factors,
        'alternative_risk_factor': alternative_risk_factors
    }
    key = EntityKey(key)
    return type_map[key.type][key.name]
