"""Healthcare utilization and treatment model."""
import typing
from typing import Dict, List, NamedTuple

import pandas as pd

from vivarium_csu_zenon import globals as project_globals, paths
from vivarium_csu_zenon.components.treatment import parameters


if typing.TYPE_CHECKING:
    from vivarium.framework.engine import Builder
    from vivarium.framework.event import Event
    from vivarium.framework.population import SimulantData


# Days to follow up.
FOLLOW_UP_MIN = 3 * 30
FOLLOW_UP_MAX = 6 * 30
LDLC_AT_TREATMENT_START = 'ldlc_at_treatment_start'
PROPORTION_REDUCTION = 'initial_treatment_proportion_reduction'
LDLC_TREATMENT_ADHERENCE = 'ldlc_treatment_adherence'
LDLC_TREATMENT_ADHERENCE_PROPENSITY = 'ldlc_treatment_adherence_propensity'
LDL_CHOLESTEROL_EXPOSURE = 'high_ldl_cholesterol.exposure'
PATIENT_PROFILE = 'patient_profile'
BACKGROUND_VISIT = 'background_visit'
FOLLOW_UP_DATE = 'follow_up_date'
FOLLOW_UP_SCHEDULING = 'follow_up_scheduling'
LDLC_TREATMENT_ALGORITHM = 'ldlc_treatment_algorithm'
LOW_DOSE_IF_STATIN = 'low_dose_if_statin'
TREATMENT_TYPE = 'treatment_type'


class __Scenarios(NamedTuple):
    baseline: str = 'baseline'
    guideline: str = 'guideline'
    guideline_and_new_treatment: str = 'guideline_and_new_treatment'


SCENARIOS = __Scenarios()


class TreatmentAlgorithm:
    """Manages healthcare utilization and treatment."""

    configuration_defaults = {
        LDLC_TREATMENT_ALGORITHM: {
            'scenario': SCENARIOS.baseline
        }
    }

    def __init__(self):
        self.patient_profile = PatientProfile()

    @property
    def name(self) -> str:
        """The name of this component."""
        return LDLC_TREATMENT_ALGORITHM

    @property
    def sub_components(self) -> List:
        """The patient profile."""
        return [self.patient_profile]

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: 'Builder'):
        """Select an algorithm based on the current scenario and get
        healthcare utilization organized and set up.

        Parameters
        ----------
        builder
            The simulation builder object.

        """
        treatment_params = self.load_treatment_parameters(builder)
        scenario = builder.configuration.ldlc_treatment_algorithm.scenario
        if scenario == SCENARIOS.baseline:
            self.visit_doctor = CurrentPractice(self.patient_profile, treatment_params)
        elif scenario in SCENARIOS:
            self.visit_doctor = GuidelineTreatment(self.patient_profile, treatment_params)
        else:
            raise ValueError(f'Invalid scenario {scenario}')

        self.clock = builder.time.clock()
        self.randomness = builder.randomness.get_stream(self.name)
        utilization_data = builder.data.load(project_globals.POPULATION.HEALTHCARE_UTILIZATION)
        background_utilization_rate = builder.lookup.build_table(utilization_data,
                                                                 parameter_columns=['age', 'year'],
                                                                 key_columns=['sex'])
        self.background_utilization_rate = builder.value.register_rate_producer('utilization_rate',
                                                                                background_utilization_rate,
                                                                                requires_columns=['age', 'sex'])

        self.population_view = builder.population.get_view([project_globals.IHD_MODEL_NAME,
                                                            project_globals.ISCHEMIC_STROKE_MODEL_NAME,
                                                            FOLLOW_UP_DATE])
        builder.population.initializes_simulants(self.on_initialize_simulants,
                                                 creates_columns=[FOLLOW_UP_DATE],
                                                 requires_columns=[project_globals.TREATMENT.name])

        builder.event.register_listener('time_step__cleanup', self.on_time_step_cleanup)
        builder.event.register_listener('time_step', self.on_time_step)

    def on_initialize_simulants(self, pop_data: 'SimulantData'):
        """For patients currently treated assign them a follow up date."""
        follow_up_date = pd.Series(pd.NaT, index=pop_data.index, name=FOLLOW_UP_DATE)
        currently_treated = self.patient_profile.currently_treated(pop_data.index)
        # noinspection PyTypeChecker
        follow_up_date.loc[currently_treated] = pd.Series(
            self.clock() + self.random_time_delta(pd.Series(28, index=pop_data.index),
                                                  pd.Series(FOLLOW_UP_MAX, index=pop_data.index)), index=pop_data.index)
        self.population_view.update(follow_up_date)

    def on_time_step_cleanup(self, event: 'Event'):
        """Send patients who will have a cardiact event at the end of the step
        to the doctor as well.

        """
        pop = self.population_view.get(event.index, query='alive == "alive"')
        # State table adjusts at the end of each event handler, so we already
        # this info even though it doesn't occur until the start of the
        # next time step.
        acute_mi = pop[project_globals.IHD_MODEL_NAME] == project_globals.ACUTE_MI_STATE_NAME
        acute_is = pop[project_globals.ISCHEMIC_STROKE_MODEL_NAME] == project_globals.ACUTE_ISCHEMIC_STROKE_STATE_NAME
        acute_cve = pop[acute_mi | acute_is].index
        follow_up_start, follow_up_end = self.visit_doctor.for_acute_cardiovascular_event(acute_cve)
        self.population_view.update(self.schedule_follow_up(follow_up_start, follow_up_end))

    def on_time_step(self, event: 'Event'):
        """Determine if someone will go for a background or follow up visit."""
        follow_up_date = self.population_view.subview([FOLLOW_UP_DATE]).get(event.index,
                                                                            query='alive == "alive"').follow_up_date
        to_follow_up = follow_up_date[(self.clock() < follow_up_date) & (follow_up_date <= event.time)].index
        follow_up_start, follow_up_end = self.visit_doctor.for_follow_up(to_follow_up)
        new_follow_up = self.schedule_follow_up(follow_up_start, follow_up_end)
        maybe_follow_up = event.index.difference(to_follow_up)
        # noinspection PyTypeChecker
        utilization_rate = self.background_utilization_rate(maybe_follow_up)  # type: pd.Series
        to_background_visit = self.randomness.filter_for_rate(maybe_follow_up,
                                                              utilization_rate,
                                                              additional_key=BACKGROUND_VISIT)
        follow_up_start, follow_up_end = self.visit_doctor.for_background_visit(to_background_visit)
        new_follow_up_background = self.schedule_follow_up(follow_up_start, follow_up_end)
        new_follow_up = new_follow_up.append(new_follow_up_background)
        self.population_view.update(new_follow_up)

    def schedule_follow_up(self, start: pd.Series, end: pd.Series) -> pd.Series:
        """Schedules follow up visits."""
        current_time = self.clock()
        # noinspection PyTypeChecker
        return pd.Series(current_time + self.random_time_delta(start, end),
                         index=start.index, name=FOLLOW_UP_DATE)

    def random_time_delta(self, start: pd.Series, end: pd.Series) -> pd.Series:
        """Generate a random time delta for each individual in the start
        and end series."""
        return pd.to_timedelta(start + (end - start) * self.randomness.get_draw(start.index), unit='day')

    @staticmethod
    def load_treatment_parameters(builder: 'Builder'):
        """Load treatment transition matrices for the scenario."""
        scenario = builder.configuration.ldlc_treatment_algorithm.scenario
        location = builder.configuration.input_data.location
        treatment_parameters = {}
        for transition_name, path in paths.TRANSITION_PARAMETERS.items():
            treatment_param = (pd.read_csv(path)
                               .set_index(['location', 'scenario'])
                               .loc[(location, scenario)]
                               .reset_index(drop=True)
                               .set_index(['current_treatment', 'next_treatment'])
                               .unstack())
            treatment_param.columns = treatment_param.columns.droplevel()
            treatment_param.columns.name = None
            treatment_param.index.name = None
            treatment_parameters[transition_name] = treatment_param
        return treatment_parameters


class PatientProfile:
    """Manager for patient information and doctor actions."""

    @property
    def name(self) -> str:
        """The name of this component."""
        return PATIENT_PROFILE

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: 'Builder'):
        """Gather all patient and doctor behavior information."""
        self.ldlc = builder.value.get_value(LDL_CHOLESTEROL_EXPOSURE)
        self.is_adherent = builder.value.get_value(LDLC_TREATMENT_ADHERENCE)
        self.p_measurement = self.load_measurement_p(builder)
        self.p_treatment_given_high = self.load_treatment_p(builder)
        self.p_treatment_type = self.load_treatment_type_p(builder)
        self.cvd_risk = builder.value.get_value('cvd_risk_category')
        self.randomness = builder.randomness.get_stream(self.name)
        self.population_view = builder.population.get_view([project_globals.TREATMENT.name,
                                                            LDLC_AT_TREATMENT_START, 'had_adverse_event',
                                                            LDLC_TREATMENT_ADHERENCE_PROPENSITY])
        builder.population.initializes_simulants(self.on_initialize_simulants,
                                                 creates_columns=[LDLC_AT_TREATMENT_START],
                                                 requires_columns=[PROPORTION_REDUCTION],
                                                 requires_values=[LDL_CHOLESTEROL_EXPOSURE])

    def on_initialize_simulants(self, pop_data: 'SimulantData'):
        """Record the ldlc at treatment start for new simulants.

        Assumes we can just back this value out by scaling up their ldlc
        value, which is questionable.

        """
        pop = self.population_view.subview([PROPORTION_REDUCTION]).get(pop_data.index)
        proportion_reduction = pop[PROPORTION_REDUCTION]  # coerce to series
        ldlc = self.ldlc(pop.index)
        ldlc_at_start = pd.Series(ldlc / (1 - proportion_reduction),
                                  index=pop_data.index, name=LDLC_AT_TREATMENT_START)
        self.population_view.update(ldlc_at_start)

    def had_adverse_event(self, index: pd.Index) -> pd.Series:
        """Determine whether a patient has had an adverse event since they
        were last put on treatment."""
        pop = self.population_view.subview(['had_adverse_event']).get(index)
        return pop['had_adverse_event']

    def at_target(self, index: pd.Index) -> pd.Series:
        """Returns whether each individual is at their ldlc target."""
        pop = self.population_view.subview([LDLC_AT_TREATMENT_START]).get(index)
        ldlc_at_start = pop[LDLC_AT_TREATMENT_START]  # coerce to series
        ldlc = self.ldlc(index)
        return ldlc / ldlc_at_start < 0.5

    def currently_treated(self, index: pd.Index) -> pd.Series:
        """Returns whether each individual is currently treated."""
        current_treatments = self.population_view.get(index)
        # noinspection PyTypeChecker
        return current_treatments[project_globals.TREATMENT.name] != project_globals.TREATMENT.none

    def sbp_is_measured_and_above_threshold(self, index: pd.Index, threshold) -> pd.Series:
        """Determine if the doctor measures sbp and finds it above threshold."""
        # noinspection PyTypeChecker
        measured = self.randomness.get_draw(index, additional_key='sbp_is_measured') < self.p_measurement
        above_threshold = self.ldlc(index) > threshold
        # noinspection PyTypeChecker
        return measured & above_threshold

    def will_treat_if_bad(self, index: pd.Index) -> pd.Series:
        """Determine whether the doctor treats if ldlc is measured too high."""
        # noinspection PyTypeChecker
        return self.randomness.get_draw(index, additional_key='will_treat_if_bad') < self.p_treatment_given_high

    def update_treatment(self, index: pd.Index, transition_params):
        current_treatment = self.population_view.subview([project_globals.TREATMENT.name]).get(index)
        current_treatment = current_treatment[project_globals.TREATMENT.name]
        new_treatment = self.randomness.choice(index, transition_params.columns,
                                               transition_params.loc[current_treatment],
                                               additional_key='update_treatment')
        new_treatment.name = project_globals.TREATMENT.name
        pop_update = pd.DataFrame({
            project_globals.TREATMENT.name: new_treatment,
            LDLC_TREATMENT_ADHERENCE_PROPENSITY: self.randomness.get_draw(index, additional_key='adherence_update')
        })
        self.population_view.update(pop_update)

    def start_guideline_treatment(self, index: pd.Index):
        new_treatment = pd.Series(project_globals.TREATMENT.none, index=index, name=project_globals.TREATMENT.name)
        adherence = pd.Series(0, index=index, name=LDLC_TREATMENT_ADHERENCE_PROPENSITY)
        cvd_risk_category = self.cvd_risk(index)
        ldlc = self.ldlc(index)

        lifestyle = (
                ((cvd_risk_category == project_globals.CVD_VERY_HIGH_RISK) & (ldlc < 1.8))

                | ((cvd_risk_category == project_globals.CVD_HIGH_RISK) & (ldlc < 2.6))

                | (((cvd_risk_category == project_globals.CVD_MODERATE_RISK)
                    | (cvd_risk_category == project_globals.CVD_LOW_RISK))
                   & ((3 < ldlc) & (ldlc < 4.9)))
        )

        high_potency_high_dose_statin = (
            ((cvd_risk_category == project_globals.CVD_VERY_HIGH_RISK) & (ldlc >= 1.8))

            | ((cvd_risk_category == project_globals.CVD_HIGH_RISK) & (ldlc >= 2.6))

            | (((cvd_risk_category == project_globals.CVD_MODERATE_RISK)
                | (cvd_risk_category == project_globals.CVD_LOW_RISK))
               & (ldlc >= 4.9))
        )

        new_treatment.loc[lifestyle] = project_globals.TREATMENT.lifestyle
        new_treatment.loc[high_potency_high_dose_statin] = project_globals.TREATMENT.high_statin_high_dose
        adherence.loc[high_potency_high_dose_statin] = self.randomness.get_draw(index[high_potency_high_dose_statin],
                                                                                additional_key='new_guideline_tx')
        pop_update = pd.DataFrame({
            project_globals.TREATMENT.name: new_treatment,
            LDLC_TREATMENT_ADHERENCE_PROPENSITY: adherence
        })
        self.population_view.update(pop_update)
        return index[high_potency_high_dose_statin], index[lifestyle]

    @staticmethod
    def load_measurement_p(builder: 'Builder') -> float:
        """Load probability that ldlc will be measured."""
        location = builder.configuration.input_data.location
        draw = builder.configuration.input_data.input_draw_number
        return parameters.sample_probability_testing_ldl_c(location, draw)

    @staticmethod
    def load_treatment_p(builder: 'Builder') -> float:
        """Load probability of treatment after high ldl measurements."""
        location = builder.configuration.input_data.location
        draw = builder.configuration.input_data.input_draw_number
        return parameters.sample_probability_rx_given_high_ldl_c(location, draw)

    @staticmethod
    def load_treatment_type_p(builder: 'Builder') -> Dict[str, float]:
        """Load probabilities of particular treatments given treated."""
        location = builder.configuration.input_data.location
        draw = builder.configuration.input_data.input_draw_number
        p_treatment_type = {treatment_type: parameters.sample_raw_drug_prescription(location, draw, treatment_type)
                            for treatment_type in [project_globals.TREATMENT.ezetimibe, project_globals.TREATMENT.fibrates,
                                                   parameters.STATIN_HIGH, parameters.STATIN_LOW]}
        p_treatment_type = dict(zip([k for k in p_treatment_type.keys()],
                                    parameters.get_adjusted_probabilities(*p_treatment_type.values())))
        return p_treatment_type


class CurrentPractice:
    """Business as usual treatment scenario."""

    ldlc_threshold = parameters.HIGH_LDL_BASELINE

    def __init__(self, patient_profile: PatientProfile, transition_parameters: Dict[str, pd.DataFrame]):
        self.patient_profile = patient_profile
        self.transition_parameters = transition_parameters

    def for_acute_cardiovascular_event(self, index: pd.Index):
        """Treat for an stroke or myocardial infarction."""
        self.patient_profile.update_treatment(index, self.transition_parameters['post_cve'])
        return pd.Series(FOLLOW_UP_MIN, index=index), pd.Series(FOLLOW_UP_MAX, index=index)

    def for_follow_up(self, index: pd.Index):
        """Treat for a follow up visit."""
        had_adverse_event = self.patient_profile.had_adverse_event(index)
        to_switch = index[had_adverse_event]
        self.patient_profile.update_treatment(to_switch, self.transition_parameters['adverse_event'])
        not_at_target = ~self.patient_profile.at_target(index)
        is_adherent = self.patient_profile.is_adherent(index)
        to_ramp = index[not_at_target & is_adherent]
        self.patient_profile.update_treatment(to_ramp, self.transition_parameters['ramp_up'])
        return pd.Series(FOLLOW_UP_MIN, index=index), pd.Series(FOLLOW_UP_MAX, index=index)

    def for_background_visit(self, index: pd.Index):
        """Treat for a background visit"""
        currently_treated = self.patient_profile.currently_treated(index)
        measured_and_bad = self.patient_profile.sbp_is_measured_and_above_threshold(index, self.ldlc_threshold)
        will_treat_if_bad = self.patient_profile.will_treat_if_bad(index)
        to_treat = index[~currently_treated & measured_and_bad & will_treat_if_bad]
        self.patient_profile.update_treatment(to_treat, self.transition_parameters['treatment_start'])
        return pd.Series(FOLLOW_UP_MIN, index=index), pd.Series(FOLLOW_UP_MAX, index=index)


class GuidelineTreatment:

    follow_up_min = 4 * 7  # 4 weeks
    follow_up_max = 6 * 7  # 6 weeks
    ldlc_threshold = 3

    def __init__(self, patient_profile: PatientProfile, transition_parameters: Dict[str, pd.DataFrame]):
        self.patient_profile = patient_profile
        self.transition_parameters = transition_parameters

    def for_acute_cardiovascular_event(self, index: pd.Index):
        """Treat for an stroke or myocardial infarction."""
        self.patient_profile.update_treatment(index, self.transition_parameters['post_cve'])
        return pd.Series(self.follow_up_min, index=index), pd.Series(self.follow_up_max, index=index)

    def for_follow_up(self, index: pd.Index):
        """Treat for a follow up visit."""
        had_adverse_event = self.patient_profile.had_adverse_event(index)
        to_switch = index[had_adverse_event]
        self.patient_profile.update_treatment(to_switch, self.transition_parameters['adverse_event'])
        not_at_target = ~self.patient_profile.at_target(index)
        is_adherent = self.patient_profile.is_adherent(index)
        to_ramp = index[not_at_target & is_adherent]
        self.patient_profile.update_treatment(to_ramp, self.transition_parameters['ramp_up'])
        return pd.Series(self.follow_up_min, index=index), pd.Series(self.follow_up_max, index=index)

    def for_background_visit(self, index: pd.Index):
        """Treat for a background visit"""
        currently_treated = self.patient_profile.currently_treated(index)
        measured_and_bad = self.patient_profile.sbp_is_measured_and_above_threshold(index, self.ldlc_threshold)
        will_treat_if_bad = self.patient_profile.will_treat_if_bad(index)
        to_treat = index[~currently_treated & measured_and_bad & will_treat_if_bad]
        follow_up_short, follow_up_long = self.patient_profile.start_guideline_treatment(to_treat)
        follow_up_min = pd.Series(self.follow_up_min, index=follow_up_short.union(follow_up_long))
        follow_up_min[follow_up_long] = FOLLOW_UP_MIN
        follow_up_max = pd.Series(self.follow_up_max, index=follow_up_short.union(follow_up_long))
        follow_up_max[follow_up_long] = FOLLOW_UP_MAX
        return follow_up_min, follow_up_max
