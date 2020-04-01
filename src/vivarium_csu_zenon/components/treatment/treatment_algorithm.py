import typing

import numpy as np
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


class TreatmentAlgorithm:

    configuration_defaults = {
        'ldlc_treatment_algorithm': {
            'scenario': 'baseline'  # ['guideline', 'guideline_and_new_treatment']
        }
    }

    def __init__(self):
        self.patient_profile = PatientProfile()

    @property
    def name(self):
        return 'ldlc_treatment_algorithm'

    @property
    def sub_components(self):
        return [self.patient_profile]

    def setup(self, builder: 'Builder'):
        scenario = builder.configuration.ldlc_treatment_algorithm.scenario
        if scenario != 'baseline':
            raise NotImplementedError
        self.visit_doctor = CurrentPractice(self.patient_profile)

        self.clock = builder.time.clock()
        self.randomness = builder.randomness.get_stream('follow_up_scheduling')
        utilization_data = builder.data.load(project_globals.POPULATION.HEALTHCARE_UTILIZATION)
        self.background_utilization_rate = builder.lookup.build_table(utilization_data)

        self.population_view = builder.population.get_view([project_globals.IHD_MODEL_NAME,
                                                            project_globals.ISCHEMIC_STROKE_MODEL_NAME,
                                                            'follow_up_date'])
        builder.population.initializes_simulants(self.on_initialize_simulants,
                                                 creates_columns=['follow_up_date'],
                                                 requires_columns=TREATMENT_COLUMNS)

        builder.event.register_listener('time_step__prepare', self.on_time_step_prepare)
        builder.event.register_listener('time_step', self.on_time_step)

    def on_initialize_simulants(self, pop_data: 'SimulantData'):
        follow_up_date = pd.Series(pd.NaT, index=pop_data.index)
        currently_treated = self.patient_profile.currently_treated(pop_data.index)
        follow_up_date.loc[currently_treated] = pd.Series(self.clock() + self.random_time_delta(pop_data.index, 0, 180),
                                                          index=pop_data.index,
                                                          name='follow_up_date')
        self.population_view.update(follow_up_date)

    def on_time_step_prepare(self, event: 'Event'):
        pop = self.population_view.get(event.index)
        acute_mi = pop[project_globals.IHD_MODEL_NAME] == project_globals.ACUTE_MI_STATE_NAME
        acute_is = pop[project_globals.ISCHEMIC_STROKE_MODEL_NAME] == project_globals.ACUTE_ISCHEMIC_STROKE_STATE_NAME
        acute_cve = pop[acute_mi | acute_is].index
        follow_up_start, follow_up_end = self.visit_doctor.for_acute_cardiovascular_event(acute_cve)
        self.population_view.update(self.schedule_follow_up(follow_up_start, follow_up_end))

    def on_time_step(self, event: 'Event'):
        follow_up_date = self.population_view.subview(['follow_up_date']).get(event.index).follow_up_date
        to_follow_up = follow_up_date[(self.clock() < follow_up_date) & (follow_up_date <= event.time)].index
        follow_up_start, follow_up_end = self.visit_doctor.for_follow_up(to_follow_up)
        new_follow_up = self.schedule_follow_up(follow_up_start, follow_up_end)
        maybe_follow_up = event.index.difference(to_follow_up)
        utilization_rate = self.background_utilization_rate(maybe_follow_up).value
        to_background_visit = self.randomness.filter_for_rate(maybe_follow_up,
                                                              utilization_rate,
                                                              additional_key='background_visit')
        follow_up_start, follow_up_end = self.visit_doctor.for_background_visit(to_background_visit)
        new_follow_up_background = self.schedule_follow_up(follow_up_start, follow_up_end)
        new_follow_up = new_follow_up.append(new_follow_up_background)
        self.population_view.update(new_follow_up)

    def schedule_follow_up(self, start: pd.Series, end: pd.Series) -> pd.Series:
        current_time = self.clock()
        return pd.Series(current_time + self.random_time_delta(start.index, start, end),
                         index=start.index, name='follow_up_date')

    def random_time_delta(self, index: pd.Index, start, end):
        return pd.to_timedelta(start + (end - start) * self.randomness.get_draw(start.index))






class PatientProfile:

    @property
    def name(self):
        return 'patient_profile'

    def setup(self, builder: 'Builder'):
        self.ldlc = builder.value.get_value('high_ldl_cholesterol.exposure')
        self.is_adherent = builder.value.get_value('ldlc_treatment_adherence')
        self.p_measurement = self.load_measurement_p(builder)
        self.p_treatment_given_high = self.load_treatment_p(builder)
        self.p_treatment_type = self.load_treatment_type_p(builder)
        self.randomness = builder.randomness.get_stream(self.name)
        self.population_view = builder.population.get_view([parameters.STATIN_LOW, parameters.STATIN_HIGH,
                                                            parameters.FIBRATES, parameters.EZETIMIBE,
                                                            parameters.LIFESTYLE, parameters.FDC,
                                                            'ldlc_at_treatment_start'])
        builder.population.initializes_simulants(self.on_initialize_simulants,
                                                 creates_columns=['ldlc_at_treatment_start'],
                                                 requires_columns=['initial_treatment_proportion_reduction'],
                                                 requires_values=['high_ldl_cholesterol.exposure'])

    def on_initialize_simulants(self, pop_data: 'SimulantData'):
        pop = self.population_view.subview(['initial_treatment_proportion_reduction']).get(pop_data.index)
        proportion_reduction = pop['initial_treatment_proportion_reduction']  # coerce to series
        ldlc = self.ldlc(pop.index)
        ldlc_at_start = pd.Series(ldlc / (1 - proportion_reduction),
                                  index=pop_data.index, name='ldlc_at_treatment_start')
        self.population_view.update(ldlc_at_start)

    def at_target(self, index: pd.Index):
        pop = self.population_view.subview(['ldlc_at_treatment_start']).get(index)
        ldlc_at_start = pop['ldlc_at_treatment_start']  # coerce to series
        ldlc = self.ldlc(index)
        return ldlc / ldlc_at_start < 0.5

    def currently_treated(self, index: pd.Index) -> pd.Series:
        current_treatments = self.population_view.get(index)
        high_statin = current_treatments[parameters.STATIN_HIGH] != 'none'
        low_statin = current_treatments[parameters.STATIN_LOW] != 'none'
        other = current_treatments[parameters.FIBRATES] | current_treatments[parameters.EZETIMIBE]
        return high_statin | low_statin | other

    def start_new_monotherapy(self, index: pd.Index, p_low_statin: float):
        new_treatment = pd.DataFrame({parameters.STATIN_LOW: 'none',
                                      parameters.STATIN_HIGH: 'none',
                                      parameters.EZETIMIBE: False,
                                      parameters.FIBRATES: False}, index=index)
        treatment_type = self.randomness.choice(index,
                                                list(self.p_treatment_type.keys()),
                                                list(self.p_treatment_type.values()),
                                                additional_key='treatment_type')
        low_statin = treatment_type == parameters.STATIN_LOW
        high_statin = treatment_type == parameters.STATIN_HIGH
        ezetimibe = treatment_type == parameters.EZETIMIBE
        fibrates = treatment_type == parameters.FIBRATES

        # noinspection PyTypeChecker
        low_dose_if_statin = self.randomness.get_draw(index, additional_key='low_dose_if_statin') < p_low_statin
        new_treatment.loc[low_statin & low_dose_if_statin, parameters.STATIN_LOW] = 'low'
        new_treatment.loc[low_statin & ~low_dose_if_statin, parameters.STATIN_LOW] = 'high'
        new_treatment.loc[high_statin & low_dose_if_statin, parameters.STATIN_HIGH] = 'low'
        new_treatment.loc[high_statin & ~low_dose_if_statin, parameters.STATIN_HIGH] = 'high'
        new_treatment.loc[ezetimibe, parameters.EZETIMIBE] = True
        new_treatment.loc[fibrates, parameters.FIBRATES] = True
        self.population_view.update(new_treatment)

    def simple_ramp(self, index: pd.Index):
        # TODO
        pass


    @staticmethod
    def load_measurement_p(builder: 'Builder') -> float:
        location = builder.configuration.location.input_data.location
        draw = builder.configuration.location.input_data.input_draw_number
        return parameters.sample_probability_testing_ldl_c(location, draw)

    @staticmethod
    def load_treatment_p(builder: 'Builder') -> float:
        location = builder.configuration.location.input_data.location
        draw = builder.configuration.location.input_data.input_draw_number
        return parameters.sample_probability_rx_given_high_ldl_c(location, draw)

    @staticmethod
    def load_treatment_type_p(builder):
        location = builder.configuration.input_data.location
        draw = builder.configuration.input_data.input_draw_number
        p_treatment_type = {treatment_type: parameters.sample_raw_drug_prescription(location, draw, treatment_type)
                            for treatment_type in [parameters.EZETIMIBE, parameters.FIBRATES,
                                                   parameters.STATIN_HIGH, parameters.STATIN_LOW]}
        p_treatment_type = dict(zip([k for k in p_treatment_type.keys()],
                                    parameters.get_adjusted_probabilities(*p_treatment_type.values())))
        return p_treatment_type


class CurrentPractice:

    # Days til follow up
    follow_up_low = 3 * 30
    follow_up_high = 6 * 30
    p_low_statin = parameters.LOW_DOSE_THRESHOLD

    def __init__(self, patient_profile: PatientProfile):
        self.patient_profile = patient_profile

    def for_acute_cardiovascular_event(self, index: pd.Index):
        not_treated = index[~self.patient_profile.currently_treated(index)]
        self.patient_profile.start_new_monotherapy(not_treated, self.p_low_statin)
        return pd.Series(self.follow_up_low, index=not_treated), pd.Series(self.follow_up_high, index=not_treated)

    def for_follow_up(self, index: pd.Index):
        # TODO: Adverse event
        not_at_target = ~self.patient_profile.at_target(index)
        is_adherent = self.patient_profile.is_adherent(index)
        to_ramp = index[not_at_target & is_adherent]
        self.patient_profile.simple_ramp(to_ramp)
        return pd.Series(self.follow_up_low, index=index), pd.Series(self.follow_up_high, index=index)

    def for_background_visit(self, index: pd.Index):
        # TODO:
        return pd.Series(self.follow_up_low, index=index), pd.Series(self.follow_up_high, index=index)
