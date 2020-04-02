"""Healthcare utilization and treatment model."""
import typing
from typing import Dict, List, NamedTuple

import pandas as pd

from vivarium_csu_zenon import globals as project_globals
from vivarium_csu_zenon.components.treatment import parameters


if typing.TYPE_CHECKING:
    from vivarium.framework.engine import Builder
    from vivarium.framework.event import Event
    from vivarium.framework.population import SimulantData


TREATMENT_COLUMNS = [parameters.STATIN_LOW, parameters.STATIN_HIGH,
                     parameters.FIBRATES, parameters.EZETIMIBE,
                     parameters.LIFESTYLE, parameters.FDC]

# Days to follow up.
FOLLOW_UP_MIN = 3 * 30
FOLLOW_UP_MAX = 6 * 30
LDLC_AT_TREATMENT_START = 'ldlc_at_treatment_start'
PROPORTION_REDUCTION = 'initial_treatment_proportion_reduction'
LDLC_TREATMENT_ADHERENCE = 'ldlc_treatment_adherence'
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
        scenario = builder.configuration.ldlc_treatment_algorithm.scenario
        if scenario != SCENARIOS.baseline:
            raise NotImplementedError
        self.visit_doctor = CurrentPractice(self.patient_profile)

        self.clock = builder.time.clock()
        self.randomness = builder.randomness.get_stream(self.name)
        utilization_data = builder.data.load(project_globals.POPULATION.HEALTHCARE_UTILIZATION)
        self.background_utilization_rate = builder.lookup.build_table(utilization_data)

        self.population_view = builder.population.get_view([project_globals.IHD_MODEL_NAME,
                                                            project_globals.ISCHEMIC_STROKE_MODEL_NAME,
                                                            FOLLOW_UP_DATE])
        builder.population.initializes_simulants(self.on_initialize_simulants,
                                                 creates_columns=[FOLLOW_UP_DATE],
                                                 requires_columns=TREATMENT_COLUMNS)

        builder.event.register_listener('time_step__cleanup', self.on_time_step_cleanup)
        builder.event.register_listener('time_step', self.on_time_step)

    def on_initialize_simulants(self, pop_data: 'SimulantData'):
        """For patients currently treated assign them a follow up date."""
        follow_up_date = pd.Series(pd.NaT, index=pop_data.index)
        currently_treated = self.patient_profile.currently_treated(pop_data.index)
        # noinspection PyTypeChecker
        follow_up_date.loc[currently_treated] = pd.Series(
            self.clock() + self.random_time_delta(pd.Series(0, index=pop_data.index),
                                                  pd.Series(FOLLOW_UP_MAX, index=pop_data.index)),
            index=pop_data.index, name=FOLLOW_UP_DATE)
        self.population_view.update(follow_up_date)

    def on_time_step_cleanup(self, event: 'Event'):
        """Send patients who will have a cardiact event at the end of the step
        to the doctor as well.

        """
        pop = self.population_view.get(event.index)
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
        follow_up_date = self.population_view.subview([FOLLOW_UP_DATE]).get(event.index).follow_up_date
        to_follow_up = follow_up_date[(self.clock() < follow_up_date) & (follow_up_date <= event.time)].index
        follow_up_start, follow_up_end = self.visit_doctor.for_follow_up(to_follow_up)
        new_follow_up = self.schedule_follow_up(follow_up_start, follow_up_end)
        maybe_follow_up = event.index.difference(to_follow_up)
        utilization_rate = self.background_utilization_rate(maybe_follow_up).value
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
        return pd.to_timedelta(start + (end - start) * self.randomness.get_draw(start.index))


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
        self.randomness = builder.randomness.get_stream(self.name)
        self.population_view = builder.population.get_view([parameters.STATIN_LOW, parameters.STATIN_HIGH,
                                                            parameters.FIBRATES, parameters.EZETIMIBE,
                                                            parameters.LIFESTYLE, parameters.FDC,
                                                            LDLC_AT_TREATMENT_START])
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

    def at_target(self, index: pd.Index) -> pd.Series:
        """Returns whether each individual is at their ldlc target."""
        pop = self.population_view.subview([LDLC_AT_TREATMENT_START]).get(index)
        ldlc_at_start = pop[LDLC_AT_TREATMENT_START]  # coerce to series
        ldlc = self.ldlc(index)
        return ldlc / ldlc_at_start < 0.5

    def currently_treated(self, index: pd.Index) -> pd.Series:
        """Returns whether each individual is currently treated."""
        current_treatments = self.population_view.get(index)
        high_statin = current_treatments[parameters.STATIN_HIGH] != parameters.STATIN_DOSES.none
        low_statin = current_treatments[parameters.STATIN_LOW] != parameters.STATIN_DOSES.none
        other = current_treatments[parameters.FIBRATES] | current_treatments[parameters.EZETIMIBE]
        return high_statin | low_statin | other

    def start_new_monotherapy(self, index: pd.Index, p_low_statin: float):
        """Starts a group of people on a new monotherapy.

        Parameters
        ----------
        index
            The people to start on therapy.
        p_low_statin
            Probability that an individual will be assigned low potency
            statin if they're assigned statins as a monotherapy.

        """
        new_treatment = pd.DataFrame({parameters.STATIN_LOW: parameters.STATIN_DOSES.none,
                                      parameters.STATIN_HIGH: parameters.STATIN_DOSES.none,
                                      parameters.EZETIMIBE: False,
                                      parameters.FIBRATES: False}, index=index)
        treatment_type = self.randomness.choice(index,
                                                list(self.p_treatment_type.keys()),
                                                list(self.p_treatment_type.values()),
                                                additional_key=TREATMENT_TYPE)
        low_statin = treatment_type == parameters.STATIN_LOW
        high_statin = treatment_type == parameters.STATIN_HIGH
        ezetimibe = treatment_type == parameters.EZETIMIBE
        fibrates = treatment_type == parameters.FIBRATES

        # noinspection PyTypeChecker
        low_dose_if_statin = self.randomness.get_draw(index, additional_key=LOW_DOSE_IF_STATIN) < p_low_statin
        new_treatment.loc[low_statin & low_dose_if_statin, parameters.STATIN_LOW] = parameters.STATIN_DOSES.low
        new_treatment.loc[low_statin & ~low_dose_if_statin, parameters.STATIN_LOW] = parameters.STATIN_DOSES.high
        new_treatment.loc[high_statin & low_dose_if_statin, parameters.STATIN_HIGH] = parameters.STATIN_DOSES.low
        new_treatment.loc[high_statin & ~low_dose_if_statin, parameters.STATIN_HIGH] = parameters.STATIN_DOSES.high
        new_treatment.loc[ezetimibe, parameters.EZETIMIBE] = True
        new_treatment.loc[fibrates, parameters.FIBRATES] = True
        self.population_view.update(new_treatment)

    def simple_ramp(self, index: pd.Index):
        """Change treatment by adding or switching drugs or increasing dose."""
        # TODO
        pass

    @staticmethod
    def load_measurement_p(builder: 'Builder') -> float:
        """Load probability that ldlc will be measured."""
        location = builder.configuration.location.input_data.location
        draw = builder.configuration.location.input_data.input_draw_number
        return parameters.sample_probability_testing_ldl_c(location, draw)

    @staticmethod
    def load_treatment_p(builder: 'Builder') -> float:
        """Load probability of treatment after high ldl measurements."""
        location = builder.configuration.location.input_data.location
        draw = builder.configuration.location.input_data.input_draw_number
        return parameters.sample_probability_rx_given_high_ldl_c(location, draw)

    @staticmethod
    def load_treatment_type_p(builder: 'Builder') -> Dict[str, float]:
        """Load probability of a treatment type if treated."""
        location = builder.configuration.input_data.location
        draw = builder.configuration.input_data.input_draw_number
        p_treatment_type = {treatment_type: parameters.sample_raw_drug_prescription(location, draw, treatment_type)
                            for treatment_type in [parameters.EZETIMIBE, parameters.FIBRATES,
                                                   parameters.STATIN_HIGH, parameters.STATIN_LOW]}
        p_treatment_type = dict(zip([k for k in p_treatment_type.keys()],
                                    parameters.get_adjusted_probabilities(*p_treatment_type.values())))
        return p_treatment_type


class CurrentPractice:
    """Business as usual treatment scenario."""

    # Days til follow up
    follow_up_low = 3 * 30
    follow_up_high = 6 * 30
    p_low_statin = parameters.LOW_DOSE_THRESHOLD

    def __init__(self, patient_profile: PatientProfile):
        self.patient_profile = patient_profile

    def for_acute_cardiovascular_event(self, index: pd.Index):
        """Treat for an stroke or myocardial infarction."""
        not_treated = index[~self.patient_profile.currently_treated(index)]
        self.patient_profile.start_new_monotherapy(not_treated, self.p_low_statin)
        return pd.Series(self.follow_up_low, index=not_treated), pd.Series(self.follow_up_high, index=not_treated)

    def for_follow_up(self, index: pd.Index):
        """Treat for a follow up visit."""
        # TODO: Adverse event
        not_at_target = ~self.patient_profile.at_target(index)
        is_adherent = self.patient_profile.is_adherent(index)
        to_ramp = index[not_at_target & is_adherent]
        self.patient_profile.simple_ramp(to_ramp)
        return pd.Series(self.follow_up_low, index=index), pd.Series(self.follow_up_high, index=index)

    def for_background_visit(self, index: pd.Index):
        """Treat for a background visit"""
        # TODO:
        return pd.Series(self.follow_up_low, index=index), pd.Series(self.follow_up_high, index=index)
