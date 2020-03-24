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
    PREVALENCE: str = 'cause.diabetes_mellitus.prevalence'
    MODERATE_DIABETES_PREVALENCE: str = 'sequela.moderate_diabetes_mellitus.prevalence'
    SEVERE_DIABETES_PREVALENCE: str = 'sequela.severe_diabetes_mellitus.prevalence'
    INCIDENCE_RATE: str = 'cause.diabetes_mellitus.incidence_rate'
    REMISSION_RATE: str = 'cause.diabetes_mellitus.remission_rate'
    MODERATE_DIABETES_PROPORTION: str = 'sequela.moderate_diabetes_mellitus.proportion'
    SEVERE_DIABETES_PROPORTION: str = 'sequela.severe_diabetes_mellitus.proportion'
    MODERATE_DIABETES_DISABILITY_WEIGHT: str = 'sequela.moderate_diabetes_mellitus.disability_weight'
    SEVERE_DIABETES_DISABILITY_WEIGHT: str = 'sequela.severe_diabetes_mellitus.disability_weight'
    CSMR: str = 'cause.diabetes_mellitus.cause_specific_mortality_rate'
    EMR: str = 'cause.diabetes_mellitus.excess_mortality_rate'
    MODERATE_DIABETES_EMR: str = 'sequela.moderate_diabetes_mellitus.excess_mortality_rate'
    SEVERE_DIABETES_EMR: str = 'sequela.severe_diabetes_mellitus.excess_mortality_rate'
    RESTRICTIONS: str = 'cause.diabetes_mellitus.restrictions'

    @property
    def name(self):
        return 'diabetes_mellitus'

    @property
    def log_name(self):
        return 'diabetes mellitus'


DIABETES_MELLITUS = __DiabetesMellitus()


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
    EXPOSURE_SD: str = 'risk_factor.high_systolic_blood_pressure.exposure_standard_deviation'
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


class __FastingPlasmaGlucose(NamedTuple):
    DISTRIBUTION: str = 'risk_factor.high_fasting_plasma_glucose_continuous.distribution'
    EXPOSURE_MEAN: str = 'risk_factor.high_fasting_plasma_glucose_continuous.exposure'
    EXPOSURE_SD: str = 'risk_factor.high_fasting_plasma_glucose_continuous.exposure_standard_deviation'
    EXPOSURE_WEIGHTS: str = 'risk_factor.high_fasting_plasma_glucose_continuous.exposure_distribution_weights'
    DIABETES_MELLITUS_THRESHOLD: str = 'risk_factor.high_fasting_plasma_glucose_continuous.diabetes_mellitus_threshold'
    RELATIVE_RISK: str = 'risk_factor.high_fasting_plasma_glucose_continuous.relative_risk'
    PAF: str = 'risk_factor.high_fasting_plasma_glucose_continuous.population_attributable_fraction'
    TMRED: str = 'risk_factor.high_fasting_plasma_glucose_continuous.tmred'
    RELATIVE_RISK_SCALAR: str = 'risk_factor.high_fasting_plasma_glucose_continuous.relative_risk_scalar'

    @property
    def name(self):
        return 'high_fasting_plasma_glucose_continuous'

    @property
    def log_name(self):
        return 'high fasting plasma glucose'


FPG = __FastingPlasmaGlucose()


class __ImpairedKidneyFunction(NamedTuple):
    CATEGORIES: str = 'risk_factor.impaired_kidney_function.categories'
    DISTRIBUTION: str = 'risk_factor.impaired_kidney_function.distribution'
    EXPOSURE: str = 'risk_factor.impaired_kidney_function.exposure'
    RELATIVE_RISK: str = 'risk_factor.impaired_kidney_function.relative_risk'
    CAT_5_DISABILITY_WEIGHT: str = 'risk_factor.cat_5_impaired_kidney_function.disability_weight'
    CAT_4_DISABILITY_WEIGHT: str = 'risk_factor.cat_4_impaired_kidney_function.disability_weight'
    CAT_3_DISABILITY_WEIGHT: str = 'risk_factor.cat_3_impaired_kidney_function.disability_weight'
    CAT_2_DISABILITY_WEIGHT: str = 'risk_factor.cat_2_impaired_kidney_function.disability_weight'
    CAT_1_DISABILITY_WEIGHT: str = 'risk_factor.cat_1_impaired_kidney_function.disability_weight'
    EMR: str = 'risk_factor.impaired_kidney_function.excess_mortality_rate'
    CSMR: str = 'risk_factor.impaired_kidney_function.cause_specific_mortality_rate'
    PAF: str = 'risk_factor.impaired_kidney_function.population_attributable_fraction'

    @property
    def name(self):
        return 'impaired_kidney_function'

    @property
    def log_name(self):
        return 'impaired kidney function'

    @property
    def disability_weights(self):
        return [
            self.CAT_1_DISABILITY_WEIGHT,
            self.CAT_2_DISABILITY_WEIGHT,
            self.CAT_3_DISABILITY_WEIGHT,
            self.CAT_4_DISABILITY_WEIGHT,
            self.CAT_5_DISABILITY_WEIGHT,
        ]


IKF = __ImpairedKidneyFunction()

MAKE_ARTIFACT_KEY_GROUPS = [
    POPULATION,
    IHD,
    ISCHEMIC_STROKE,
    DIABETES_MELLITUS,
    LDL_C,
    SBP,
    FPG,
    IKF,
]

###########################
# Disease Model variables #
###########################


class TransitionString(str):

    def __new__(cls, value):
        # noinspection PyArgumentList
        obj = str.__new__(cls, value.lower())
        obj.from_state, obj.to_state = value.split('_TO_')
        return obj


IHD_MODEL_NAME = 'ischemic_heart_disease'
IHD_SUSCEPTIBLE_STATE_NAME = f'susceptible_to_{IHD_MODEL_NAME}'
ACUTE_MI_STATE_NAME = 'acute_myocardial_infarction'
POST_MI_STATE_NAME = 'post_myocardial_infarction'
IHD_MODEL_STATES = (IHD_SUSCEPTIBLE_STATE_NAME, ACUTE_MI_STATE_NAME, POST_MI_STATE_NAME)
IHD_MODEL_TRANSITIONS = (
    TransitionString(f'{IHD_SUSCEPTIBLE_STATE_NAME}_TO_{ACUTE_MI_STATE_NAME}'),
    TransitionString(f'{ACUTE_MI_STATE_NAME}_TO_{POST_MI_STATE_NAME}'),
    TransitionString(f'{POST_MI_STATE_NAME}_TO_{ACUTE_MI_STATE_NAME}')
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
    TransitionString(f'{ISCHEMIC_STROKE_SUSCEPTIBLE_STATE_NAME}_TO_{ACUTE_ISCHEMIC_STROKE_STATE_NAME}'),
    TransitionString(f'{ACUTE_ISCHEMIC_STROKE_STATE_NAME}_TO_{POST_ISCHEMIC_STROKE_STATE_NAME}'),
    TransitionString(f'{POST_ISCHEMIC_STROKE_STATE_NAME}_TO_{ACUTE_ISCHEMIC_STROKE_STATE_NAME}')
)

DIABETES_MELLITUS_MODEL_NAME = 'diabetes_mellitus'
DIABETES_MELLITUS_SUSCEPTIBLE_STATE_NAME = f'susceptible_to_{DIABETES_MELLITUS_MODEL_NAME}'
MODERATE_DIABETES_MELLITUS_STATE_NAME = f'moderate_{DIABETES_MELLITUS_MODEL_NAME}'
SEVERE_DIABETES_MELLITUS_STATE_NAME = f'severe_{DIABETES_MELLITUS_MODEL_NAME}'
DIABETES_MELLITUS_MODEL_STATES = (
    DIABETES_MELLITUS_SUSCEPTIBLE_STATE_NAME,
    MODERATE_DIABETES_MELLITUS_STATE_NAME,
    SEVERE_DIABETES_MELLITUS_STATE_NAME
)
DIABETES_MELLITUS_MODEL_TRANSITIONS = (
    TransitionString(f'{DIABETES_MELLITUS_SUSCEPTIBLE_STATE_NAME}_TO_{MODERATE_DIABETES_MELLITUS_STATE_NAME}'),
    TransitionString(f'{DIABETES_MELLITUS_SUSCEPTIBLE_STATE_NAME}_TO_{SEVERE_DIABETES_MELLITUS_STATE_NAME}'),
    TransitionString(f'{MODERATE_DIABETES_MELLITUS_STATE_NAME}_TO_{DIABETES_MELLITUS_SUSCEPTIBLE_STATE_NAME}'),
    TransitionString(f'{SEVERE_DIABETES_MELLITUS_STATE_NAME}_TO_{DIABETES_MELLITUS_SUSCEPTIBLE_STATE_NAME}'),
)

CKD_MODEL_NAME = 'chronic_kidney_disease'
CKD_SUSCEPTIBLE_STATE_NAME = f'susceptible_to_{CKD_MODEL_NAME}'
ALBUMINURIA_STATE_NAME = 'albuminuria'
STAGE_III_CKD_STATE_NAME = f'stage_iii_{CKD_MODEL_NAME}'
STAGE_IV_CKD_STATE_NAME = f'stage_iv_{CKD_MODEL_NAME}'
STAGE_V_CKD_STATE_NAME = f'stage_v_{CKD_MODEL_NAME}'
CKD_MODEL_STATES = (
    CKD_SUSCEPTIBLE_STATE_NAME,
    CKD_MODEL_NAME
)
CKD_MODEL_TRANSITIONS = (
    TransitionString(f'{CKD_SUSCEPTIBLE_STATE_NAME}_TO_{CKD_MODEL_NAME}'),
    TransitionString(f'{CKD_MODEL_NAME}_TO_{CKD_SUSCEPTIBLE_STATE_NAME}'),
)
IKF_TMREL_CATEGORY = 'cat5'
CKD_IKF_MAP = {
    CKD_SUSCEPTIBLE_STATE_NAME: 'cat5',
    ALBUMINURIA_STATE_NAME: 'cat4',
    STAGE_III_CKD_STATE_NAME: 'cat3',
    STAGE_IV_CKD_STATE_NAME: 'cat2',
    STAGE_V_CKD_STATE_NAME: 'cat1',
}

DISEASE_MODELS = (
    IHD_MODEL_NAME,
    ISCHEMIC_STROKE_MODEL_NAME,
    DIABETES_MELLITUS_MODEL_NAME,
    CKD_MODEL_NAME
)
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

# CVD Risk Categories
CVD_VERY_HIGH_RISK = 'very_high_risk'
CVD_HIGH_RISK = 'high_risk'
CVD_MODERATE_RISK = 'moderate_risk'
CVD_LOW_RISK = 'low_risk'

# Correlated propensity columns
SBP_PROPENSITY_COLUMN = f'{SBP.name}_propensity'
FPG_PROPENSITY_COLUMN = f'{FPG.name}_propensity'
IKF_PROPENSITY_COLUMN = f'{IKF.name}_propensity'
LDL_C_PROPENSITY_COLUMN = f'{LDL_C.name}_propensity'
DIABETES_PROPENSITY_COLUMN = f'{DIABETES_MELLITUS.name}_propensity'
CORRELATED_PROPENSITY_COLUMNS = [
    SBP_PROPENSITY_COLUMN,
    FPG_PROPENSITY_COLUMN,
    IKF_PROPENSITY_COLUMN,
    LDL_C_PROPENSITY_COLUMN,
    DIABETES_PROPENSITY_COLUMN,
]



########################
# Stratification Constants #
########################

CVD_RISK_CATEGORIES = [CVD_VERY_HIGH_RISK, CVD_HIGH_RISK, CVD_MODERATE_RISK, CVD_LOW_RISK]

#################################
# Results columns and variables #
#################################

TOTAL_POPULATION_COLUMN = 'total_population'
TOTAL_YLDS_COLUMN = 'years_lived_with_disability'
TOTAL_YLLS_COLUMN = 'years_of_life_lost'

# Columns from parallel runs
INPUT_DRAW_COLUMN = 'input_draw'
RANDOM_SEED_COLUMN = 'random_seed'
OUTPUT_SCENARIO_COLUMN = 'scenario'

STANDARD_COLUMNS = {
    'total_population': TOTAL_POPULATION_COLUMN,
    'total_ylls': TOTAL_YLLS_COLUMN,
    'total_ylds': TOTAL_YLDS_COLUMN,
}

THROWAWAY_COLUMNS = ([f'{state}_event_count' for state in STATES]
                     + [f'{DIABETES_MELLITUS_MODEL_NAME}_event_count']  # From transient state
                     + [f'{state}_prevalent_cases_at_sim_end' for state in STATES])

TOTAL_POPULATION_COLUMN_TEMPLATE = 'total_population_{POP_STATE}'
PERSON_TIME_COLUMN_TEMPLATE = 'person_time_in_{YEAR}_among_{SEX}_in_age_group_{AGE_GROUP}_cvd_{CVD_RISK}'
DEATH_COLUMN_TEMPLATE = 'death_due_to_{CAUSE_OF_DEATH}_in_{YEAR}_among_{SEX}_in_age_group_{AGE_GROUP}_cvd_{CVD_RISK}'
YLLS_COLUMN_TEMPLATE = 'ylls_due_to_{CAUSE_OF_DEATH}_in_{YEAR}_among_{SEX}_in_age_group_{AGE_GROUP}_cvd_{CVD_RISK}'
YLDS_COLUMN_TEMPLATE = 'ylds_due_to_{CAUSE_OF_DISABILITY}_in_{YEAR}_among_{SEX}_in_age_group_{AGE_GROUP}_cvd_{CVD_RISK}'
STATE_PERSON_TIME_COLUMN_TEMPLATE = '{STATE}_person_time_in_{YEAR}_among_{SEX}_in_age_group_{AGE_GROUP}_cvd_{CVD_RISK}'
TRANSITION_COUNT_COLUMN_TEMPLATE = '{TRANSITION}_event_count_in_{YEAR}_among_{SEX}_in_age_group_{AGE_GROUP}_cvd_{CVD_RISK}'

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
    '90_to_94',
    '95_plus',
)

CAUSES_OF_DISABILITY = (
    ACUTE_MI_STATE_NAME,
    POST_MI_STATE_NAME,
    ACUTE_ISCHEMIC_STROKE_STATE_NAME,
    POST_ISCHEMIC_STROKE_STATE_NAME,
    MODERATE_DIABETES_MELLITUS_STATE_NAME,
    SEVERE_DIABETES_MELLITUS_STATE_NAME,
    CKD_MODEL_NAME,
)
CAUSES_OF_DEATH = CAUSES_OF_DISABILITY + ('other_causes',)

TEMPLATE_FIELD_MAP = {
    'POP_STATE': POP_STATES,
    'YEAR': YEARS,
    'SEX': SEXES,
    'AGE_GROUP': AGE_GROUPS,
    'CAUSE_OF_DEATH': CAUSES_OF_DEATH,
    'CAUSE_OF_DISABILITY': CAUSES_OF_DISABILITY,
    'STATE': STATES,
    'TRANSITION': TRANSITIONS,
    'CVD_RISK': CVD_RISK_CATEGORIES,
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

