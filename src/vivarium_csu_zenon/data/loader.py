"""Loads, standardizes and validates input data for the simulation.

Abstract the extract and transform pieces of the artifact ETL.
The intent here is to provide a uniform interface around this portion
of artifact creation. The value of this interface shows up when more
complicated data needs are part of the project. See the BEP project
for an example.

`BEP <https://github.com/ihmeuw/vivarium_gates_bep/blob/master/src/vivarium_gates_bep/data/loader.py>`_

.. admonition::
"""
import pandas as pd
import numpy as np
from vivarium_gbd_access import gbd
from gbd_mapping import causes, risk_factors, covariates, sequelae
from vivarium.framework.artifact import EntityKey
from vivarium_inputs import core, extract, interface, utilities, utility_data, globals as vi_globals
from vivarium_inputs.mapping_extension import alternative_risk_factors
import vivarium_inputs.validation.sim as validation

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
        project_globals.IHD.ACUTE_MI_EMR: load_standard_excess_mortality_rate,
        project_globals.IHD.POST_MI_EMR: load_standard_excess_mortality_rate,
        project_globals.IHD.CSMR: load_standard_data,
        project_globals.IHD.RESTRICTIONS: load_metadata,

        project_globals.ISCHEMIC_STROKE.ACUTE_STROKE_PREVALENCE: load_ischemic_stroke_prevalence,
        project_globals.ISCHEMIC_STROKE.POST_STROKE_PREVALENCE: load_ischemic_stroke_prevalence,
        project_globals.ISCHEMIC_STROKE.ACUTE_STROKE_INCIDENCE_RATE: load_standard_data,
        project_globals.ISCHEMIC_STROKE.ACUTE_STROKE_DISABILITY_WEIGHT: load_ischemic_stroke_disability_weight,
        project_globals.ISCHEMIC_STROKE.POST_STROKE_DISABILITY_WEIGHT: load_ischemic_stroke_disability_weight,
        project_globals.ISCHEMIC_STROKE.ACUTE_STROKE_EMR: load_standard_excess_mortality_rate,
        project_globals.ISCHEMIC_STROKE.POST_STROKE_EMR: load_standard_excess_mortality_rate,
        project_globals.ISCHEMIC_STROKE.CSMR: load_standard_data,
        project_globals.ISCHEMIC_STROKE.RESTRICTIONS: load_metadata,

        project_globals.DIABETES_MELLITUS.MODERATE_DIABETES_PREVALENCE: load_diabetes_mellitus_prevalence,
        project_globals.DIABETES_MELLITUS.SEVERE_DIABETES_PREVALENCE: load_diabetes_mellitus_prevalence,
        project_globals.DIABETES_MELLITUS.MODERATE_DIABETES_DISABILITY_WEIGHT: load_diabetes_mellitus_disability_weight,
        project_globals.DIABETES_MELLITUS.SEVERE_DIABETES_DISABILITY_WEIGHT: load_diabetes_mellitus_disability_weight,
        project_globals.DIABETES_MELLITUS.MODERATE_DIABETES_EMR: load_diabetes_mellitus_excess_mortality_rate,
        project_globals.DIABETES_MELLITUS.SEVERE_DIABETES_EMR: load_diabetes_mellitus_excess_mortality_rate,
        project_globals.DIABETES_MELLITUS.CSMR: load_standard_data,
        project_globals.DIABETES_MELLITUS.RESTRICTIONS: load_metadata,

        project_globals.CKD.ALBUMINURIA_PREVALENCE: load_ckd_prevalence,
        project_globals.CKD.STAGE_III_CKD_PREVALENCE: load_ckd_prevalence,
        project_globals.CKD.STAGE_IV_CKD_PREVALENCE: load_ckd_prevalence,
        project_globals.CKD.STAGE_V_CKD_PREVALENCE: load_ckd_prevalence,
        project_globals.CKD.ALBUMINURIA_DISABILITY_WEIGHT: load_ckd_disability_weight,
        project_globals.CKD.STAGE_III_CKD_DISABILITY_WEIGHT: load_ckd_disability_weight,
        project_globals.CKD.STAGE_IV_CKD_DISABILITY_WEIGHT: load_ckd_disability_weight,
        project_globals.CKD.STAGE_V_CKD_DISABILITY_WEIGHT: load_ckd_disability_weight,
        project_globals.CKD.ALBUMINURIA_EMR: load_ckd_excess_mortality_rate,
        project_globals.CKD.STAGE_III_CKD_EMR: load_ckd_excess_mortality_rate,
        project_globals.CKD.STAGE_IV_CKD_EMR: load_ckd_excess_mortality_rate,
        project_globals.CKD.STAGE_V_CKD_EMR: load_ckd_excess_mortality_rate,
        project_globals.CKD.CSMR: load_standard_data,
        project_globals.CKD.RESTRICTIONS: load_metadata,
        
        project_globals.LDL_C.DISTRIBUTION: load_metadata,
        project_globals.LDL_C.EXPOSURE_MEAN: load_standard_data,
        project_globals.LDL_C.EXPOSURE_SD: load_standard_data,
        project_globals.LDL_C.EXPOSURE_WEIGHTS: load_standard_data,
        project_globals.LDL_C.RELATIVE_RISK: load_standard_data,
        project_globals.LDL_C.PAF: load_standard_data,
        project_globals.LDL_C.TMRED: load_metadata,
        project_globals.LDL_C.RELATIVE_RISK_SCALAR: load_metadata,
        
        project_globals.SBP.DISTRIBUTION: load_metadata,
        project_globals.SBP.EXPOSURE_MEAN: load_standard_data,
        project_globals.SBP.EXPOSURE_SD: load_standard_data,
        project_globals.SBP.EXPOSURE_WEIGHTS: load_standard_data,
        project_globals.SBP.RELATIVE_RISK: load_standard_data,
        project_globals.SBP.PAF: load_standard_data,
        project_globals.SBP.TMRED: load_metadata,
        project_globals.SBP.RELATIVE_RISK_SCALAR: load_metadata,

        project_globals.FPG.DISTRIBUTION: load_metadata,
        project_globals.FPG.EXPOSURE_MEAN: load_standard_data,
        project_globals.FPG.EXPOSURE_SD: load_standard_data,
        project_globals.FPG.EXPOSURE_WEIGHTS: load_standard_data,
        project_globals.FPG.RELATIVE_RISK: load_standard_data,
        project_globals.FPG.PAF: load_standard_data,
        project_globals.FPG.TMRED: load_metadata,
        project_globals.FPG.RELATIVE_RISK_SCALAR: load_metadata,

        project_globals.IKF.DISTRIBUTION: load_metadata,
        project_globals.IKF.RELATIVE_RISK: load_ikf_relative_risk,
        project_globals.IKF.PAF: load_ikf_paf,
        project_globals.IKF.CATEGORIES: load_standard_data,
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


def load_ckd_prevalence(key: str, location: str) -> pd.DataFrame:
    ckd_sequelae = {
        project_globals.CKD.ALBUMINURIA_PREVALENCE: [
            sequelae.albuminuria_with_preserved_gfr_due_to_glomerulonephritis,
            sequelae.albuminuria_with_preserved_gfr_due_to_hypertension,
            sequelae.albuminuria_with_preserved_gfr_due_to_other_and_unspecified_causes,
            sequelae.albuminuria_with_preserved_gfr_due_to_type_1_diabetes_mellitus,
            sequelae.albuminuria_with_preserved_gfr_due_to_type_2_diabetes_mellitus,
        ],
        project_globals.CKD.STAGE_III_CKD_PREVALENCE: [
            sequelae.stage_iii_chronic_kidney_disease_and_mild_anemia_due_to_hypertension,
            sequelae.stage_iii_chronic_kidney_disease_and_moderate_anemia_due_to_hypertension,
            sequelae.stage_iii_chronic_kidney_disease_and_severe_anemia_due_to_hypertension,
            sequelae.stage_iii_chronic_kidney_disease_without_anemia_due_to_hypertension,
            sequelae.stage_iii_chronic_kidney_disease_and_mild_anemia_due_to_glomerulonephritis,
            sequelae.stage_iii_chronic_kidney_disease_and_moderate_anemia_due_to_glomerulonephritis,
            sequelae.stage_iii_chronic_kidney_disease_and_severe_anemia_due_to_glomerulonephritis,
            sequelae.stage_iii_chronic_kidney_disease_without_anemia_due_to_glomerulonephritis,
            sequelae.stage_iii_chronic_kidney_disease_and_mild_anemia_due_to_other_and_unspecified_causes,
            sequelae.stage_iii_chronic_kidney_disease_and_moderate_anemia_due_to_other_and_unspecified_causes,
            sequelae.stage_iii_chronic_kidney_disease_and_severe_anemia_due_to_other_and_unspecified_causes,
            sequelae.stage_iii_chronic_kidney_disease_without_anemia_due_to_other_and_unspecified_causes,
            sequelae.stage_iii_chronic_kidney_disease_and_severe_anemia_due_to_type_1_diabetes_mellitus,
            sequelae.stage_iii_chronic_kidney_disease_and_severe_anemia_due_to_type_2_diabetes_mellitus,
            sequelae.stage_iii_chronic_kidney_disease_and_moderate_anemia_due_to_type_1_diabetes_mellitus,
            sequelae.stage_iii_chronic_kidney_disease_and_moderate_anemia_due_to_type_2_diabetes_mellitus,
            sequelae.stage_iii_chronic_kidney_disease_and_mild_anemia_due_to_type_1_diabetes_mellitus,
            sequelae.stage_iii_chronic_kidney_disease_and_mild_anemia_due_to_type_2_diabetes_mellitus,
            sequelae.stage_iii_chronic_kidney_disease_without_anemia_due_to_type_1_diabetes_mellitus,
            sequelae.stage_iii_chronic_kidney_disease_without_anemia_due_to_type_2_diabetes_mellitus,
        ],
        project_globals.CKD.STAGE_IV_CKD_PREVALENCE: [
            sequelae.stage_iv_chronic_kidney_disease_untreated_without_anemia_due_to_hypertension,
            sequelae.stage_iv_chronic_kidney_disease_untreated_and_mild_anemia_due_to_hypertension,
            sequelae.stage_iv_chronic_kidney_disease_untreated_and_moderate_anemia_due_to_hypertension,
            sequelae.stage_iv_chronic_kidney_disease_untreated_and_severe_anemia_due_to_hypertension,
            sequelae.stage_iv_chronic_kidney_disease_untreated_without_anemia_due_to_glomerulonephritis,
            sequelae.stage_iv_chronic_kidney_disease_untreated_and_mild_anemia_due_to_glomerulonephritis,
            sequelae.stage_iv_chronic_kidney_disease_untreated_and_moderate_anemia_due_to_glomerulonephritis,
            sequelae.stage_iv_chronic_kidney_disease_untreated_and_severe_anemia_due_to_glomerulonephritis,
            sequelae.stage_iv_chronic_kidney_disease_untreated_without_anemia_due_to_other_and_unspecified_causes,
            sequelae.stage_iv_chronic_kidney_disease_untreated_and_mild_anemia_due_to_other_and_unspecified_causes,
            sequelae.stage_iv_chronic_kidney_disease_untreated_and_moderate_anemia_due_to_other_and_unspecified_causes,
            sequelae.stage_iv_chronic_kidney_disease_untreated_and_severe_anemia_due_to_other_and_unspecified_causes,
            sequelae.stage_iv_chronic_kidney_disease_untreated_and_severe_anemia_due_to_type_1_diabetes_mellitus,
            sequelae.stage_iv_chronic_kidney_disease_untreated_and_severe_anemia_due_to_type_2_diabetes_mellitus,
            sequelae.stage_iv_chronic_kidney_disease_untreated_and_moderate_anemia_due_to_type_1_diabetes_mellitus,
            sequelae.stage_iv_chronic_kidney_disease_untreated_and_moderate_anemia_due_to_type_2_diabetes_mellitus,
            sequelae.stage_iv_chronic_kidney_disease_untreated_and_mild_anemia_due_to_type_1_diabetes_mellitus,
            sequelae.stage_iv_chronic_kidney_disease_untreated_and_mild_anemia_due_to_type_2_diabetes_mellitus,
            sequelae.stage_iv_chronic_kidney_disease_untreated_without_anemia_due_to_type_1_diabetes_mellitus,
            sequelae.stage_iv_chronic_kidney_disease_untreated_without_anemia_due_to_type_2_diabetes_mellitus,
        ],
        project_globals.CKD.STAGE_V_CKD_PREVALENCE: [
            sequelae.end_stage_renal_disease_after_transplant_due_to_hypertension,
            sequelae.end_stage_renal_disease_on_dialysis_due_to_hypertension,
            sequelae.end_stage_renal_disease_after_transplant_due_to_glomerulonephritis,
            sequelae.end_stage_renal_disease_on_dialysis_due_to_glomerulonephritis,
            sequelae.end_stage_renal_disease_after_transplant_due_to_other_and_unspecified_causes,
            sequelae.end_stage_renal_disease_on_dialysis_due_to_other_and_unspecified_causes,
            sequelae.stage_v_chronic_kidney_disease_untreated_without_anemia_due_to_hypertension,
            sequelae.stage_v_chronic_kidney_disease_untreated_and_mild_anemia_due_to_hypertension,
            sequelae.stage_v_chronic_kidney_disease_untreated_and_moderate_anemia_due_to_hypertension,
            sequelae.stage_v_chronic_kidney_disease_untreated_and_severe_anemia_due_to_hypertension,
            sequelae.stage_v_chronic_kidney_disease_untreated_without_anemia_due_to_glomerulonephritis,
            sequelae.stage_v_chronic_kidney_disease_untreated_and_mild_anemia_due_to_glomerulonephritis,
            sequelae.stage_v_chronic_kidney_disease_untreated_and_moderate_anemia_due_to_glomerulonephritis,
            sequelae.stage_v_chronic_kidney_disease_untreated_and_severe_anemia_due_to_glomerulonephritis,
            sequelae.stage_v_chronic_kidney_disease_untreated_without_anemia_due_to_other_and_unspecified_causes,
            sequelae.stage_v_chronic_kidney_disease_untreated_and_mild_anemia_due_to_other_and_unspecified_causes,
            sequelae.stage_v_chronic_kidney_disease_untreated_and_moderate_anemia_due_to_other_and_unspecified_causes,
            sequelae.stage_v_chronic_kidney_disease_untreated_and_severe_anemia_due_to_other_and_unspecified_causes,
            sequelae.end_stage_renal_disease_after_transplant_due_to_type_1_diabetes_mellitus,
            sequelae.end_stage_renal_disease_after_transplant_due_to_type_2_diabetes_mellitus,
            sequelae.end_stage_renal_disease_on_dialysis_due_to_type_1_diabetes_mellitus,
            sequelae.end_stage_renal_disease_on_dialysis_due_to_type_2_diabetes_mellitus,
            sequelae.stage_v_chronic_kidney_disease_untreated_and_severe_anemia_due_to_type_1_diabetes_mellitus,
            sequelae.stage_v_chronic_kidney_disease_untreated_and_severe_anemia_due_to_type_2_diabetes_mellitus,
            sequelae.stage_v_chronic_kidney_disease_untreated_and_moderate_anemia_due_to_type_1_diabetes_mellitus,
            sequelae.stage_v_chronic_kidney_disease_untreated_and_moderate_anemia_due_to_type_2_diabetes_mellitus,
            sequelae.stage_v_chronic_kidney_disease_untreated_and_mild_anemia_due_to_type_1_diabetes_mellitus,
            sequelae.stage_v_chronic_kidney_disease_untreated_and_mild_anemia_due_to_type_2_diabetes_mellitus,
            sequelae.stage_v_chronic_kidney_disease_untreated_without_anemia_due_to_type_1_diabetes_mellitus,
            sequelae.stage_v_chronic_kidney_disease_untreated_without_anemia_due_to_type_2_diabetes_mellitus,
        ]
    }

    seq = ckd_sequelae[key]

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


def load_ckd_disability_weight(key: str, location: str) -> pd.DataFrame:
    ckd_sequelae = {
        project_globals.CKD.ALBUMINURIA_DISABILITY_WEIGHT: [
            sequelae.albuminuria_with_preserved_gfr_due_to_glomerulonephritis,
            sequelae.albuminuria_with_preserved_gfr_due_to_hypertension,
            sequelae.albuminuria_with_preserved_gfr_due_to_other_and_unspecified_causes,
            sequelae.albuminuria_with_preserved_gfr_due_to_type_1_diabetes_mellitus,
            sequelae.albuminuria_with_preserved_gfr_due_to_type_2_diabetes_mellitus,
        ],
        project_globals.CKD.STAGE_III_CKD_DISABILITY_WEIGHT: [
            sequelae.stage_iii_chronic_kidney_disease_and_mild_anemia_due_to_hypertension,
            sequelae.stage_iii_chronic_kidney_disease_and_moderate_anemia_due_to_hypertension,
            sequelae.stage_iii_chronic_kidney_disease_and_severe_anemia_due_to_hypertension,
            sequelae.stage_iii_chronic_kidney_disease_without_anemia_due_to_hypertension,
            sequelae.stage_iii_chronic_kidney_disease_and_mild_anemia_due_to_glomerulonephritis,
            sequelae.stage_iii_chronic_kidney_disease_and_moderate_anemia_due_to_glomerulonephritis,
            sequelae.stage_iii_chronic_kidney_disease_and_severe_anemia_due_to_glomerulonephritis,
            sequelae.stage_iii_chronic_kidney_disease_without_anemia_due_to_glomerulonephritis,
            sequelae.stage_iii_chronic_kidney_disease_and_mild_anemia_due_to_other_and_unspecified_causes,
            sequelae.stage_iii_chronic_kidney_disease_and_moderate_anemia_due_to_other_and_unspecified_causes,
            sequelae.stage_iii_chronic_kidney_disease_and_severe_anemia_due_to_other_and_unspecified_causes,
            sequelae.stage_iii_chronic_kidney_disease_without_anemia_due_to_other_and_unspecified_causes,
            sequelae.stage_iii_chronic_kidney_disease_and_severe_anemia_due_to_type_1_diabetes_mellitus,
            sequelae.stage_iii_chronic_kidney_disease_and_severe_anemia_due_to_type_2_diabetes_mellitus,
            sequelae.stage_iii_chronic_kidney_disease_and_moderate_anemia_due_to_type_1_diabetes_mellitus,
            sequelae.stage_iii_chronic_kidney_disease_and_moderate_anemia_due_to_type_2_diabetes_mellitus,
            sequelae.stage_iii_chronic_kidney_disease_and_mild_anemia_due_to_type_1_diabetes_mellitus,
            sequelae.stage_iii_chronic_kidney_disease_and_mild_anemia_due_to_type_2_diabetes_mellitus,
            sequelae.stage_iii_chronic_kidney_disease_without_anemia_due_to_type_1_diabetes_mellitus,
            sequelae.stage_iii_chronic_kidney_disease_without_anemia_due_to_type_2_diabetes_mellitus,
        ],
        project_globals.CKD.STAGE_IV_CKD_DISABILITY_WEIGHT: [
            sequelae.stage_iv_chronic_kidney_disease_untreated_without_anemia_due_to_hypertension,
            sequelae.stage_iv_chronic_kidney_disease_untreated_and_mild_anemia_due_to_hypertension,
            sequelae.stage_iv_chronic_kidney_disease_untreated_and_moderate_anemia_due_to_hypertension,
            sequelae.stage_iv_chronic_kidney_disease_untreated_and_severe_anemia_due_to_hypertension,
            sequelae.stage_iv_chronic_kidney_disease_untreated_without_anemia_due_to_glomerulonephritis,
            sequelae.stage_iv_chronic_kidney_disease_untreated_and_mild_anemia_due_to_glomerulonephritis,
            sequelae.stage_iv_chronic_kidney_disease_untreated_and_moderate_anemia_due_to_glomerulonephritis,
            sequelae.stage_iv_chronic_kidney_disease_untreated_and_severe_anemia_due_to_glomerulonephritis,
            sequelae.stage_iv_chronic_kidney_disease_untreated_without_anemia_due_to_other_and_unspecified_causes,
            sequelae.stage_iv_chronic_kidney_disease_untreated_and_mild_anemia_due_to_other_and_unspecified_causes,
            sequelae.stage_iv_chronic_kidney_disease_untreated_and_moderate_anemia_due_to_other_and_unspecified_causes,
            sequelae.stage_iv_chronic_kidney_disease_untreated_and_severe_anemia_due_to_other_and_unspecified_causes,
            sequelae.stage_iv_chronic_kidney_disease_untreated_and_severe_anemia_due_to_type_1_diabetes_mellitus,
            sequelae.stage_iv_chronic_kidney_disease_untreated_and_severe_anemia_due_to_type_2_diabetes_mellitus,
            sequelae.stage_iv_chronic_kidney_disease_untreated_and_moderate_anemia_due_to_type_1_diabetes_mellitus,
            sequelae.stage_iv_chronic_kidney_disease_untreated_and_moderate_anemia_due_to_type_2_diabetes_mellitus,
            sequelae.stage_iv_chronic_kidney_disease_untreated_and_mild_anemia_due_to_type_1_diabetes_mellitus,
            sequelae.stage_iv_chronic_kidney_disease_untreated_and_mild_anemia_due_to_type_2_diabetes_mellitus,
            sequelae.stage_iv_chronic_kidney_disease_untreated_without_anemia_due_to_type_1_diabetes_mellitus,
            sequelae.stage_iv_chronic_kidney_disease_untreated_without_anemia_due_to_type_2_diabetes_mellitus,
        ],
        project_globals.CKD.STAGE_V_CKD_DISABILITY_WEIGHT: [
            sequelae.end_stage_renal_disease_after_transplant_due_to_hypertension,
            sequelae.end_stage_renal_disease_on_dialysis_due_to_hypertension,
            sequelae.end_stage_renal_disease_after_transplant_due_to_glomerulonephritis,
            sequelae.end_stage_renal_disease_on_dialysis_due_to_glomerulonephritis,
            sequelae.end_stage_renal_disease_after_transplant_due_to_other_and_unspecified_causes,
            sequelae.end_stage_renal_disease_on_dialysis_due_to_other_and_unspecified_causes,
            sequelae.stage_v_chronic_kidney_disease_untreated_without_anemia_due_to_hypertension,
            sequelae.stage_v_chronic_kidney_disease_untreated_and_mild_anemia_due_to_hypertension,
            sequelae.stage_v_chronic_kidney_disease_untreated_and_moderate_anemia_due_to_hypertension,
            sequelae.stage_v_chronic_kidney_disease_untreated_and_severe_anemia_due_to_hypertension,
            sequelae.stage_v_chronic_kidney_disease_untreated_without_anemia_due_to_glomerulonephritis,
            sequelae.stage_v_chronic_kidney_disease_untreated_and_mild_anemia_due_to_glomerulonephritis,
            sequelae.stage_v_chronic_kidney_disease_untreated_and_moderate_anemia_due_to_glomerulonephritis,
            sequelae.stage_v_chronic_kidney_disease_untreated_and_severe_anemia_due_to_glomerulonephritis,
            sequelae.stage_v_chronic_kidney_disease_untreated_without_anemia_due_to_other_and_unspecified_causes,
            sequelae.stage_v_chronic_kidney_disease_untreated_and_mild_anemia_due_to_other_and_unspecified_causes,
            sequelae.stage_v_chronic_kidney_disease_untreated_and_moderate_anemia_due_to_other_and_unspecified_causes,
            sequelae.stage_v_chronic_kidney_disease_untreated_and_severe_anemia_due_to_other_and_unspecified_causes,
            sequelae.end_stage_renal_disease_after_transplant_due_to_type_1_diabetes_mellitus,
            sequelae.end_stage_renal_disease_after_transplant_due_to_type_2_diabetes_mellitus,
            sequelae.end_stage_renal_disease_on_dialysis_due_to_type_1_diabetes_mellitus,
            sequelae.end_stage_renal_disease_on_dialysis_due_to_type_2_diabetes_mellitus,
            sequelae.stage_v_chronic_kidney_disease_untreated_and_severe_anemia_due_to_type_1_diabetes_mellitus,
            sequelae.stage_v_chronic_kidney_disease_untreated_and_severe_anemia_due_to_type_2_diabetes_mellitus,
            sequelae.stage_v_chronic_kidney_disease_untreated_and_moderate_anemia_due_to_type_1_diabetes_mellitus,
            sequelae.stage_v_chronic_kidney_disease_untreated_and_moderate_anemia_due_to_type_2_diabetes_mellitus,
            sequelae.stage_v_chronic_kidney_disease_untreated_and_mild_anemia_due_to_type_1_diabetes_mellitus,
            sequelae.stage_v_chronic_kidney_disease_untreated_and_mild_anemia_due_to_type_2_diabetes_mellitus,
            sequelae.stage_v_chronic_kidney_disease_untreated_without_anemia_due_to_type_1_diabetes_mellitus,
            sequelae.stage_v_chronic_kidney_disease_untreated_without_anemia_due_to_type_2_diabetes_mellitus,
        ]
    }

    seq = ckd_sequelae[key]

    prevalence_disability_weights = []
    for s in seq:
        prevalence = interface.get_measure(s, 'prevalence', location)
        disability_weight = interface.get_measure(s, 'disability_weight', location)
        prevalence_disability_weights.append(prevalence * disability_weight)

    diabetes_prevalence = interface.get_measure(causes.chronic_kidney_disease, 'prevalence', location)
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


def load_ckd_excess_mortality_rate(key: str, location: str) -> pd.DataFrame:
    if key == project_globals.CKD.STAGE_V_CKD_EMR:
        raw_ckd_emr = get_data(project_globals.CKD.CSMR, location)
        prevalence_stage_v = get_data(project_globals.CKD.STAGE_V_CKD_PREVALENCE, location)
        ckd_emr = (raw_ckd_emr / prevalence_stage_v).fillna(0)
    else:
        ckd_emr = get_data(project_globals.POPULATION.DEMOGRAPHY, location)
        ckd_emr['value'] = 0
    return ckd_emr


def load_standard_excess_mortality_rate(key: str, location: str) -> pd.DataFrame:
    meids = {
        project_globals.IHD.ACUTE_MI_EMR: 1814,
        project_globals.IHD.POST_MI_EMR: 15755,
        project_globals.ISCHEMIC_STROKE.ACUTE_STROKE_EMR: 9310,
        project_globals.ISCHEMIC_STROKE.POST_STROKE_EMR: 10837,
    }

    return _load_em_from_meid(meids[key], location)


def load_ikf_relative_risk(key: str, location: str) -> pd.DataFrame:
    key = EntityKey(key)
    entity = get_entity(key)

    value_cols = vi_globals.DRAW_COLUMNS
    location_id = utility_data.get_location_id(location)

    data = extract.extract_data(entity, 'relative_risk', location_id, validate=False)
    yll_only_causes = set([c.gbd_id for c in causes if c.restrictions.yll_only])
    data = data[~data.cause_id.isin(yll_only_causes)]

    data = utilities.convert_affected_entity(data, 'cause_id')
    morbidity = data.morbidity == 1
    mortality = data.mortality == 1
    data.loc[morbidity & mortality, 'affected_measure'] = 'incidence_rate'
    data.loc[morbidity & ~mortality, 'affected_measure'] = 'incidence_rate'
    data.loc[~morbidity & mortality, 'affected_measure'] = 'excess_mortality_rate'
    data = core.filter_relative_risk_to_cause_restrictions(data)

    data = (data.groupby(['affected_entity', 'parameter'])
            .apply(utilities.normalize, fill_value=1)
            .reset_index(drop=True))
    data = data.filter(vi_globals.DEMOGRAPHIC_COLUMNS + ['affected_entity', 'affected_measure', 'parameter']
                       + vi_globals.DRAW_COLUMNS)

    tmrel_cat = utility_data.get_tmrel_category(entity)
    tmrel_mask = data.parameter == tmrel_cat
    data.loc[tmrel_mask, value_cols] = (
        data.loc[tmrel_mask, value_cols].mask(np.isclose(data.loc[tmrel_mask, value_cols], 1.0), 1.0)
    )

    data = utilities.reshape(data, value_cols=value_cols)
    data = utilities.scrub_gbd_conventions(data, location)
    validation.validate_for_simulation(data, entity, key.measure, location)
    data = utilities.split_interval(data, interval_column='age', split_column_prefix='age')
    data = utilities.split_interval(data, interval_column='year', split_column_prefix='year')
    return utilities.sort_hierarchical_data(data)


def load_ikf_paf(key: str, location: str) -> pd.DataFrame:
    key = EntityKey(key)
    entity = get_entity(key)

    value_cols = vi_globals.DRAW_COLUMNS
    location_id = utility_data.get_location_id(location)

    data = extract.extract_data(entity, 'population_attributable_fraction', location_id, validate=False)
    relative_risk = extract.extract_data(entity, 'relative_risk', location_id, validate=False)

    yll_only_causes = set([c.gbd_id for c in causes if c.restrictions.yll_only])
    data = data[~data.cause_id.isin(yll_only_causes)]
    relative_risk = relative_risk[~relative_risk.cause_id.isin(yll_only_causes)]

    data = (data.groupby('cause_id', as_index=False)
            .apply(core.filter_by_relative_risk, relative_risk)
            .reset_index(drop=True))

    causes_map = {c.gbd_id: c for c in causes}
    temp = []
    # We filter paf age groups by cause level restrictions.
    for (c_id, measure), df in data.groupby(['cause_id', 'measure_id']):
        cause = causes_map[c_id]
        measure = 'yll' if measure == vi_globals.MEASURES['YLLs'] else 'yld'
        df = utilities.filter_data_by_restrictions(df, cause, measure, utility_data.get_age_group_ids())
        temp.append(df)
    data = pd.concat(temp, ignore_index=True)

    data = utilities.convert_affected_entity(data, 'cause_id')
    data.loc[data['measure_id'] == vi_globals.MEASURES['YLLs'], 'affected_measure'] = 'excess_mortality_rate'
    data.loc[data['measure_id'] == vi_globals.MEASURES['YLDs'], 'affected_measure'] = 'incidence_rate'
    data = (data.groupby(['affected_entity', 'affected_measure'])
            .apply(utilities.normalize, fill_value=0)
            .reset_index(drop=True))
    data = data.filter(vi_globals.DEMOGRAPHIC_COLUMNS + ['affected_entity', 'affected_measure']
                       + vi_globals.DRAW_COLUMNS)

    data = utilities.reshape(data, value_cols=value_cols)
    data = utilities.scrub_gbd_conventions(data, location)
    validation.validate_for_simulation(data, entity, key.measure, location)
    data = utilities.split_interval(data, interval_column='age', split_column_prefix='age')
    data = utilities.split_interval(data, interval_column='year', split_column_prefix='year')
    return utilities.sort_hierarchical_data(data)


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
