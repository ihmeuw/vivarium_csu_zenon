from typing import NamedTuple

import itertools


####################
# Project metadata #
####################

PROJECT_NAME = 'vivarium_csu_zenon'
CLUSTER_PROJECT = 'proj_csu'

CLUSTER_QUEUE = 'all.q'
MAKE_ARTIFACT_MEM = '3G'
MAKE_ARTIFACT_CPU = '1'
MAKE_ARTIFACT_RUNTIME = '3:00:00'
MAKE_ARTIFACT_SLEEP = 10

LOCATIONS = [
    'Brazil',
    'China',
    'France',
    'Italy',
    'Spain',
    'Russian Federation',
]


#############
# Data Keys #
#############

METADATA_LOCATIONS = 'metadata.locations'


class __Population(NamedTuple):
    STRUCTURE: str = 'population.structure'
    AGE_BINS: str = 'population.age_bins'
    DEMOGRAPHY: str = 'population.demographic_dimensions'
    TMRLE: str = 'population.theoretical_minimum_risk_life_expectancy'
    ACMR: str = 'cause.all_causes.cause_specific_mortality_rate'

    @property
    def name(self):
        return 'population'

    @property
    def log_name(self):
        return 'population'


POPULATION = __Population()


class __IHD(NamedTuple):
    ACUTE_MI_PREVALENCE: str = 'sequela.acute_myocardial_infarction.prevalence'
    POST_MI_PREVALENCE: str = 'sequela.post_myocardial_infarction.prevalence'
    ACUTE_MI_INCIDENCE_RATE: str = 'cause.ischemic_heart_disease.incidence_rate'
    ACUTE_MI_DISABILITY_WEIGHT: str = 'sequela.acute_myocardial_infarction.disability_weight'
    POST_MI_DISABILITY_WEIGHT: str = 'sequela.post_myocardial_infarction.disability_weight'
    ACUTE_MI_EMR: str = 'sequela.acute_myocardial_infarction.excess_mortality_rate'
    POST_MI_EMR: str = 'sequela.post_myocardial_infarction.excess_mortality_rate'
    CSMR: str = 'cause.ischemic_heart_disease.cause_specific_mortality_rate'
    RESTRICTIONS: str = 'cause.ischemic_heart_disease.restrictions'

    @property
    def name(self):
        return 'ischemic_heart_disease'

    @property
    def log_name(self):
        return 'ischemic heart disease'


IHD = __IHD()


class __IschemicStroke(NamedTuple):
    ACUTE_STROKE_PREVALENCE: str = 'sequela.acute_ischemic_stroke.prevalence'
    POST_STROKE_PREVALENCE: str = 'sequela.post_ischemic_stroke.prevalence'
    ACUTE_STROKE_INCIDENCE_RATE: str = 'cause.ischemic_stroke.incidence_rate'
    ACUTE_STROKE_DISABILITY_WEIGHT: str = 'sequela.acute_ischemic_stroke.disability_weight'
    POST_STROKE_DISABILITY_WEIGHT: str = 'sequela.post_ischemic_stroke.disability_weight'
    ACUTE_STROKE_EMR: str = 'sequela.acute_ischemic_stroke.excess_mortality_rate'
    POST_STROKE_EMR: str = 'sequela.post_ischemic_stroke.excess_mortality_rate'
    CSMR: str = 'cause.ischemic_stroke.cause_specific_mortality_rate'
    RESTRICTIONS: str = 'cause.ischemic_stroke.restrictions'

    @property
    def name(self):
        return 'ischemic_stroke'

    @property
    def log_name(self):
        return 'ischemic stroke'


ISCHEMIC_STROKE = __IschemicStroke()


class __DiabetesMellitus(NamedTuple):
    MODERATE_DIABETES_PREVALENCE: str = 'sequela.moderate_diabetes_mellitus.prevalence'
    SEVERE_DIABETES_PREVALENCE: str = 'sequela.severe_diabetes_mellitus.prevalence'
    MODERATE_DIABETES_DISABILITY_WEIGHT: str = 'sequela.moderate_diabetes_mellitus.disability_weight'
    SEVERE_DIABETES_DISABILITY_WEIGHT: str = 'sequela.severe_diabetes_mellitus.disability_weight'
    MODERATE_DIABETES_EMR: str = 'sequela.moderate_diabetes_mellitus.excess_mortality_rate'
    SEVERE_DIABETES_EMR: str = 'sequela.severe_diabetes_mellitus.excess_mortality_rate'
    CSMR: str = 'cause.diabetes_mellitus.cause_specific_mortality_rate'
    RESTRICTIONS: str = 'cause.diabetes_mellitus.restrictions'

    @property
    def name(self):
        return 'diabetes_mellitus'

    @property
    def log_name(self):
        return 'diabetes mellitus'


DIABETES_MELLITUS = __DiabetesMellitus()


class __ChronicKidneyDisease(NamedTuple):
    ALBUMINURIA_PREVALENCE: str = 'sequela.albuminuria.prevalence'
    STAGE_III_CKD_PREVALENCE: str = 'sequela.stage_iii_chronic_kidney_disease.prevalence'
    STAGE_IV_CKD_PREVALENCE: str = 'sequela.stage_iv_chronic_kidney_disease.prevalence'
    STAGE_V_CKD_PREVALENCE: str = 'sequela.stage_v_chronic_kidney_disease.prevalence'
    ALBUMINURIA_DISABILITY_WEIGHT: str = 'sequela.albuminuria.disability_weight'
    STAGE_III_CKD_DISABILITY_WEIGHT: str = 'sequela.stage_iii_chronic_kidney_disease.disability_weight'
    STAGE_IV_CKD_DISABILITY_WEIGHT: str = 'sequela.stage_iv_chronic_kidney_disease.disability_weight'
    STAGE_V_CKD_DISABILITY_WEIGHT: str = 'sequela.stage_v_chronic_kidney_disease.disability_weight'
    ALBUMINURIA_EMR: str = 'sequela.albuminuria.excess_mortality_rate'
    STAGE_III_CKD_EMR: str = 'sequela.stage_iii_chronic_kidney_disease.excess_mortality_rate'
    STAGE_IV_CKD_EMR: str = 'sequela.stage_iv_chronic_kidney_disease.excess_mortality_rate'
    STAGE_V_CKD_EMR: str = 'sequela.stage_v_chronic_kidney_disease.excess_mortality_rate'
    CSMR: str = 'cause.chronic_kidney_disease.cause_specific_mortality_rate'
    RESTRICTIONS: str = 'cause.chronic_kidney_disease.restrictions'

    @property
    def name(self):
        return 'chronic_kidney_disease'

    @property
    def log_name(self):
        return 'chronic kidney disease'


CKD = __ChronicKidneyDisease()


class __HighLDLCholesterol(NamedTuple):
    DISTRIBUTION: str = 'risk_factor.high_ldl_cholesterol.distribution'
    EXPOSURE_MEAN: str = 'risk_factor.high_ldl_cholesterol.exposure'
    EXPOSURE_SD: str = 'risk_factor.high_ldl_cholesterol.exposure_standard_deviation'
    EXPOSURE_WEIGHTS: str = 'risk_factor.high_ldl_cholesterol.exposure_distribution_weights'
    RELATIVE_RISK: str = 'risk_factor.high_ldl_cholesterol.relative_risk'
    PAF: str = 'risk_factor.high_ldl_cholesterol.population_attributable_fraction'
    TMRED: str = 'risk_factor.high_ldl_cholesterol.tmred'
    RELATIVE_RISK_SCALAR: str = 'risk_factor.high_ldl_cholesterol.relative_risk_scalar'

    @property
    def name(self):
        return 'high_ldl_cholesterol'

    @property
    def log_name(self):
        return 'high ldl cholesterol'


LDL_C = __HighLDLCholesterol()


class __HighSystolicBloodPressure(NamedTuple):
    DISTRIBUTION: str = 'risk_factor.high_systolic_blood_pressure.distribution'
    EXPOSURE_MEAN: str = 'risk_factor.high_systolic_blood_pressure.exposure'
    EXPOSURE_SD = 'risk_factor.high_systolic_blood_pressure.exposure_standard_deviation'
    EXPOSURE_WEIGHTS: str = 'risk_factor.high_systolic_blood_pressure.exposure_distribution_weights'
    RELATIVE_RISK: str = 'risk_factor.high_systolic_blood_pressure.relative_risk'
    PAF: str = 'risk_factor.high_systolic_blood_pressure.population_attributable_fraction'
    TMRED: str = 'risk_factor.high_systolic_blood_pressure.tmred'
    RELATIVE_RISK_SCALAR: str = 'risk_factor.high_systolic_blood_pressure.relative_risk_scalar'

    @property
    def name(self):
        return 'high_systolic_blood_pressure'

    @property
    def log_name(self):
        return 'high systolic blood pressure'


SBP = __HighSystolicBloodPressure()

MAKE_ARTIFACT_KEY_GROUPS = [
    POPULATION,
    IHD,
    ISCHEMIC_STROKE,
    DIABETES_MELLITUS,
    CKD,
    LDL_C,
    SBP,
]

###########################
# Disease Model variables #
###########################

IHD_MODEL_NAME = 'ischemic_heart_disease'
IHD_SUSCEPTIBLE_STATE_NAME = f'susceptible_to_{IHD_MODEL_NAME}'
ACUTE_MI_STATE_NAME = 'acute_myocardial_infarction'
POST_MI_STATE_NAME = 'post_myocardial_infarction'
IHD_MODEL_STATES = (IHD_SUSCEPTIBLE_STATE_NAME, ACUTE_MI_STATE_NAME, POST_MI_STATE_NAME)
IHD_MODEL_TRANSITIONS = (
    f'{IHD_SUSCEPTIBLE_STATE_NAME}_TO_{ACUTE_MI_STATE_NAME}',
    f'{ACUTE_MI_STATE_NAME}_TO_{POST_MI_STATE_NAME}',
    f'{POST_MI_STATE_NAME}_TO_{ACUTE_MI_STATE_NAME}'
)

ISCHEMIC_STROKE_MODEL_NAME = 'ischemic_stroke'
ISCHEMIC_STROKE_SUSCEPTIBLE_STATE_NAME = f'susceptible_to_{ISCHEMIC_STROKE_MODEL_NAME}'
ACUTE_ISCHEMIC_STROKE_STATE_NAME = 'acute_ischemic_stroke'
POST_ISCHEMIC_STROKE_STATE_NAME = 'post_ischemic_stroke'
ISCHEMIC_STROKE_MODEL_STATES = (
    ISCHEMIC_STROKE_SUSCEPTIBLE_STATE_NAME,
    ACUTE_ISCHEMIC_STROKE_STATE_NAME,
    POST_ISCHEMIC_STROKE_STATE_NAME
)
ISCHEMIC_STROKE_MODEL_TRANSITIONS = (
    f'{ISCHEMIC_STROKE_SUSCEPTIBLE_STATE_NAME}_TO_{ACUTE_ISCHEMIC_STROKE_STATE_NAME}',
    f'{ACUTE_ISCHEMIC_STROKE_STATE_NAME}_TO_{POST_ISCHEMIC_STROKE_STATE_NAME}',
    f'{POST_ISCHEMIC_STROKE_STATE_NAME}_TO_{ACUTE_ISCHEMIC_STROKE_STATE_NAME}'
)

DIABETES_MELLITUS_MODEL_NAME = 'diabetes_mellitus'
DIABETES_MELLITUS_SUSCEPTIBLE_STATE_NAME = f'susceptible_to_{DIABETES_MELLITUS_MODEL_NAME}'
MODERATE_DIABETES_MELLITUS_STATE_NAME = 'moderate_diabetes_mellitus'
SEVERE_DIABETES_MELLITUS_STATE_NAME = 'severe_diabetes_mellitus'
DIABETES_MELLITUS_MODEL_STATES = (
    DIABETES_MELLITUS_SUSCEPTIBLE_STATE_NAME,
    MODERATE_DIABETES_MELLITUS_STATE_NAME,
    SEVERE_DIABETES_MELLITUS_STATE_NAME
)
DIABETES_MELLITUS_MODEL_TRANSITIONS = ()

CKD_MODEL_NAME = 'chronic_kidney_disease'
CKD_SUSCEPTIBLE_STATE_NAME = f'susceptible_to_{CKD_MODEL_NAME}'
ALBUMINURIA_STATE_NAME = 'albuminuria'
STAGE_III_CKD_STATE_NAME = f'stage_iii_{CKD_MODEL_NAME}'
STAGE_IV_CKD_STATE_NAME = f'stage_iv_{CKD_MODEL_NAME}'
STAGE_V_CKD_STATE_NAME = f'stage_v_{CKD_MODEL_NAME}'
CKD_MODEL_STATES = (
    CKD_SUSCEPTIBLE_STATE_NAME,
    ALBUMINURIA_STATE_NAME,
    STAGE_III_CKD_STATE_NAME,
    STAGE_IV_CKD_STATE_NAME,
    STAGE_V_CKD_STATE_NAME
)
CKD_MODEL_TRANSITIONS = ()

DISEASE_MODELS = (IHD_MODEL_NAME, ISCHEMIC_STROKE_MODEL_NAME, DIABETES_MELLITUS_MODEL_NAME, CKD_MODEL_NAME)
DISEASE_MODEL_MAP = {
    IHD_MODEL_NAME: {
        'states': IHD_MODEL_STATES,
        'transitions': IHD_MODEL_TRANSITIONS,
    },
    ISCHEMIC_STROKE_MODEL_NAME: {
        'states': ISCHEMIC_STROKE_MODEL_STATES,
        'transitions': ISCHEMIC_STROKE_MODEL_TRANSITIONS,
    },
    DIABETES_MELLITUS_MODEL_NAME: {
        'states': DIABETES_MELLITUS_MODEL_STATES,
        'transitions': DIABETES_MELLITUS_MODEL_TRANSITIONS,
    },
    CKD_MODEL_NAME: {
        'states': CKD_MODEL_STATES,
        'transitions': CKD_MODEL_TRANSITIONS,
    },
}

STATES = tuple(state for model in DISEASE_MODELS for state in DISEASE_MODEL_MAP[model]['states'])
TRANSITIONS = tuple(transition for model in DISEASE_MODELS for transition in DISEASE_MODEL_MAP[model]['transitions'])

#################################
# Results columns and variables #
#################################

TOTAL_POPULATION_COLUMN = 'total_population'
TOTAL_YLDS_COLUMN = 'years_lived_with_disability'
TOTAL_YLLS_COLUMN = 'years_of_life_lost'

# Columns from parallel runs
INPUT_DRAW_COLUMN = 'input_draw'
RANDOM_SEED_COLUMN = 'random_seed'

STANDARD_COLUMNS = {
    'total_population': TOTAL_POPULATION_COLUMN,
    'total_ylls': TOTAL_YLLS_COLUMN,
    'total_ylds': TOTAL_YLDS_COLUMN,
}

THROWAWAY_COLUMNS = ([f'{state}_event_count' for state in STATES]
                     + [f'{state}_prevalent_cases_at_sim_end' for state in STATES])

TOTAL_POPULATION_COLUMN_TEMPLATE = 'total_population_{POP_STATE}'
PERSON_TIME_COLUMN_TEMPLATE = 'person_time_in_{YEAR}_among_{SEX}_in_age_group_{AGE_GROUP}'
DEATH_COLUMN_TEMPLATE = 'death_due_to_{CAUSE_OF_DEATH}_in_{YEAR}_among_{SEX}_in_age_group_{AGE_GROUP}'
YLLS_COLUMN_TEMPLATE = 'ylls_due_to_{CAUSE_OF_DEATH}_in_{YEAR}_among_{SEX}_in_age_group_{AGE_GROUP}'
YLDS_COLUMN_TEMPLATE = 'ylds_due_to_{CAUSE_OF_DISABILITY}_in_{YEAR}_among_{SEX}_in_age_group_{AGE_GROUP}'
STATE_PERSON_TIME_COLUMN_TEMPLATE = '{STATE}_person_time_in_{YEAR}_among_{SEX}_in_age_group_{AGE_GROUP}'
TRANSITION_COUNT_COLUMN_TEMPLATE = '{TRANSITION}_event_count_in_{YEAR}_among_{SEX}_in_age_group_{AGE_GROUP}'

COLUMN_TEMPLATES = {
    'population': TOTAL_POPULATION_COLUMN_TEMPLATE,
    'person_time': PERSON_TIME_COLUMN_TEMPLATE,
    'deaths': DEATH_COLUMN_TEMPLATE,
    'ylls': YLLS_COLUMN_TEMPLATE,
    'ylds': YLDS_COLUMN_TEMPLATE,
    'state_person_time': STATE_PERSON_TIME_COLUMN_TEMPLATE,
    'transition_count': TRANSITION_COUNT_COLUMN_TEMPLATE,
}

NON_COUNT_TEMPLATES = [
]

POP_STATES = ('living', 'dead', 'tracked', 'untracked')
SEXES = ('male', 'female')
YEARS = tuple(range(2020, 2025))
AGE_GROUPS = (
    '30_to_34',
    '35_to_39',
    '40_to_44',
    '45_to_49',
    '50_to_54',
    '55_to_59',
    '60_to_64',
    '65_to_69',
    '70_to_74',
    '75_to_79',
    '80_to_84',
    '85_to_89',
    '95_plus',
)
# TODO: do we include states with no mortality (e.g. MODERATE_DIABETES_MELLITUS_STATE_NAME)?
CAUSES_OF_DEATH = (
    'other_causes',
    ACUTE_MI_STATE_NAME,
    POST_MI_STATE_NAME,
    ACUTE_ISCHEMIC_STROKE_STATE_NAME,
    POST_ISCHEMIC_STROKE_STATE_NAME,
    SEVERE_DIABETES_MELLITUS_STATE_NAME,
    STAGE_V_CKD_STATE_NAME,
)
CAUSES_OF_DISABILITY = (
    ACUTE_MI_STATE_NAME,
    POST_MI_STATE_NAME,
    ACUTE_ISCHEMIC_STROKE_STATE_NAME,
    POST_ISCHEMIC_STROKE_STATE_NAME,
    MODERATE_DIABETES_MELLITUS_STATE_NAME,
    SEVERE_DIABETES_MELLITUS_STATE_NAME,
    ALBUMINURIA_STATE_NAME,
    STAGE_III_CKD_STATE_NAME,
    STAGE_IV_CKD_STATE_NAME,
    STAGE_V_CKD_STATE_NAME,
)

TEMPLATE_FIELD_MAP = {
    'POP_STATE': POP_STATES,
    'YEAR': YEARS,
    'SEX': SEXES,
    'AGE_GROUP': AGE_GROUPS,
    'CAUSE_OF_DEATH': CAUSES_OF_DEATH,
    'CAUSE_OF_DISABILITY': CAUSES_OF_DISABILITY,
    'STATE': STATES,
    'TRANSITION': TRANSITIONS,
}


def RESULT_COLUMNS(kind='all'):
    if kind not in COLUMN_TEMPLATES and kind != 'all':
        raise ValueError(f'Unknown result column type {kind}')
    columns = []
    if kind == 'all':
        for k in COLUMN_TEMPLATES:
            columns += RESULT_COLUMNS(k)
        columns = list(STANDARD_COLUMNS.values()) + columns
    else:
        template = COLUMN_TEMPLATES[kind]
        filtered_field_map = {field: values
                              for field, values in TEMPLATE_FIELD_MAP.items() if f'{{{field}}}' in template}
        fields, value_groups = filtered_field_map.keys(), itertools.product(*filtered_field_map.values())
        for value_group in value_groups:
            columns.append(template.format(**{field: value for field, value in zip(fields, value_group)}).lower())
    return columns

