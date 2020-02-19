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
MAKE_ARTIFACT_SLEEP = '10'

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
    ALL_DIABETES_INCIDENCE_RATE: str = 'cause.diabetes_mellitus.incidence_rate'
    MODERATE_DIABETES_PROPORTION: str = 'sequela.moderate_diabetes_mellitus.incidence_proportion'
    SEVERE_DIABETES_PROPORTION: str = 'sequela.severe_diabetes_mellitus.incidence_proportion'
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
    f'{POST_MI_STATE_NAME} _TO_{ACUTE_MI_STATE_NAME}'
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
    f'{POST_ISCHEMIC_STROKE_STATE_NAME} _TO_{ACUTE_ISCHEMIC_STROKE_STATE_NAME}'
)

DIABETES_MELLITUS_MODEL_NAME = 'diabetes_mellitus'
DIABETES_MELLITUS_SUSCEPTIBLE_STATE_NAME = f'susceptible_to_{DIABETES_MELLITUS_MODEL_NAME}'
TRANSIENT_DIABETES_MELLITUS_STATE_NAME = 'transient_diabetes_mellitus'
MODERATE_DIABETES_MELLITUS_STATE_NAME = 'moderate_diabetes_mellitus'
SEVERE_DIABETES_MELLITUS_STATE_NAME = 'severe_diabetes_mellitus'
DIABETES_MELLITUS_MODEL_STATES = (
    DIABETES_MELLITUS_SUSCEPTIBLE_STATE_NAME,
    MODERATE_DIABETES_MELLITUS_STATE_NAME,
    SEVERE_DIABETES_MELLITUS_STATE_NAME
)
DIABETES_MELLITUS_MODEL_TRANSITIONS = (
    f'{DIABETES_MELLITUS_SUSCEPTIBLE_STATE_NAME}_TO_{TRANSIENT_DIABETES_MELLITUS_STATE_NAME}',
    f'{TRANSIENT_DIABETES_MELLITUS_STATE_NAME} _TO_{MODERATE_DIABETES_MELLITUS_STATE_NAME}'
    f'{TRANSIENT_DIABETES_MELLITUS_STATE_NAME}_TO_{SEVERE_DIABETES_MELLITUS_STATE_NAME}',
)

DISEASE_MODELS = (IHD_MODEL_NAME, ISCHEMIC_STROKE_MODEL_NAME, DIABETES_MELLITUS_MODEL_NAME)
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
}


#################################
# Results columns and variables #
#################################

TOTAL_POPULATION_COLUMN = 'total_population'
TOTAL_YLDS_COLUMN = 'years_lived_with_disability'
TOTAL_YLLS_COLUMN = 'years_of_life_lost'

STANDARD_COLUMNS = {
    'total_population': TOTAL_POPULATION_COLUMN,
    'total_ylls': TOTAL_YLLS_COLUMN,
    'total_ylds': TOTAL_YLDS_COLUMN,
}

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

POP_STATES = ('living', 'dead', 'tracked', 'untracked')
SEXES = ('male', 'female')
# TODO - add literals for years in the model
YEARS = ()
# TODO - add literals for ages in the model
AGE_GROUPS = ()
# TODO - add causes of death
CAUSES_OF_DEATH = (
    'other_causes',
    ACUTE_MI_STATE_NAME,
)
# TODO - add causes of disability
CAUSES_OF_DISABILITY = (
    ACUTE_MI_STATE_NAME,
)
STATES = (state for model in DISEASE_MODELS for state in DISEASE_MODEL_MAP[model]['states'])
TRANSITIONS = (transition for model in DISEASE_MODELS for transition in DISEASE_MODEL_MAP[model]['transitions'])

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
            columns.append(template.format(**{field: value for field, value in zip(fields, value_group)}))
    return columns

