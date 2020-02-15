"""Loads, standardizes and validates input data for the simulation.

Abstract the extract and transform pieces of the artifact ETL.
The intent here is to provide a uniform interface around this portion
of artifact creation. The value of this interface shows up when more
complicated data needs are part of the project. See the BEP project
for an example.

`BEP <https://github.com/ihmeuw/vivarium_gates_bep/blob/master/src/vivarium_gates_bep/data/loader.py>`_

.. admonition::

   No logging is done here. Logging is done in vivarium inputs itself and forwarded.
"""
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
        project_globals.IHD.ACUTE_MI_INCIDENCE: load_standard_data,
        project_globals.IHD.ACUTE_MI_DISABILITY_WEIGHT: load_ihd_disability_weight,
        project_globals.IHD.POST_MI_DISABILITY_WEIGHT: load_ihd_disability_weight,
        project_globals.IHD.ACUTE_MI_EMR: load_ihd_excess_mortality_rate,
        project_globals.IHD.POST_MI_EMR: load_ihd_excess_mortality_rate,
        project_globals.IHD.CSMR: load_standard_data,
        project_globals.IHD.RESTRICTIONS: load_metadata,
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
        ihd_disability_weight = interface.get_measure(s, 'disability_weight', location)
        prevalence_disability_weights.append(prevalence * ihd_disability_weight)

    ihd_prevalence = interface.get_measure(causes.ischemic_heart_disease, 'prevalence', location)
    ihd_disability_weight = (sum(prevalence_disability_weights) / ihd_prevalence).fillna(0)
    return ihd_disability_weight


def load_ihd_excess_mortality_rate(key: str, location: str) -> pd.DataFrame:
    meids = {
        project_globals.IHD.ACUTE_MI_EMR: 1814,
        project_globals.IHD.POST_MI_EMR: 15755,
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
