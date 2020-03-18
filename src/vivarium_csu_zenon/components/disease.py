import typing

import pandas as pd
from vivarium_public_health.disease import (DiseaseState as DiseaseState_, DiseaseModel, SusceptibleState,
                                            TransientDiseaseState, RateTransition as RateTransition_)

from vivarium_csu_zenon import globals as project_globals

if typing.TYPE_CHECKING:
    from vivarium.framework.engine import Builder
    from vivarium.framework.population import SimulantData
    from vivarium.framework.event import Event


class RateTransition(RateTransition_):
    def load_transition_rate_data(self, builder):
        if 'transition_rate' in self._get_data_functions:
            rate_data = self._get_data_functions['transition_rate'](builder, self.input_state.cause,
                                                                    self.output_state.cause)
            pipeline_name = f'{self.input_state.cause}_to_{self.output_state.cause}.transition_rate'
        else:
            raise ValueError("No valid data functions supplied.")
        return rate_data, pipeline_name


class DiseaseState(DiseaseState_):

    # I really need to rewrite the state machine code.  It's super inflexible
    def add_transition(self, output, source_data_type=None, get_data_functions=None, **kwargs):
        if source_data_type == 'rate':
            if get_data_functions is None or 'transition_rate' not in get_data_functions:
                raise ValueError('Must supply get data functions for transition_rate.')
            t = RateTransition(self, output, get_data_functions, **kwargs)
            self.transition_set.append(t)
        else:
            t = super().add_transition(output, source_data_type, get_data_functions, **kwargs)
        return t


def IschemicHeartDisease():
    susceptible = SusceptibleState('ischemic_heart_disease')
    data_funcs = {'dwell_time': lambda *args: pd.Timedelta(days=28)}
    acute_mi = DiseaseState('acute_myocardial_infarction', cause_type='sequela', get_data_functions=data_funcs)
    post_mi = DiseaseState('post_myocardial_infarction', cause_type='sequela',)

    susceptible.allow_self_transitions()
    data_funcs = {
        'incidence_rate': lambda _, builder: builder.data.load('cause.ischemic_heart_disease.incidence_rate')
    }
    susceptible.add_transition(acute_mi, source_data_type='rate', get_data_functions=data_funcs)
    acute_mi.allow_self_transitions()
    acute_mi.add_transition(post_mi)
    post_mi.allow_self_transitions()
    data_funcs = {
        'transition_rate': lambda builder, *_: builder.data.load('cause.ischemic_heart_disease.incidence_rate')
    }
    post_mi.add_transition(acute_mi, source_data_type='rate', get_data_functions=data_funcs)

    return DiseaseModel('ischemic_heart_disease', states=[susceptible, acute_mi, post_mi])


def IschemicStroke():
    susceptible = SusceptibleState('ischemic_stroke')
    data_funcs = {'dwell_time': lambda *args: pd.Timedelta(days=28)}
    acute_stroke = DiseaseState('acute_ischemic_stroke', cause_type='sequela', get_data_functions=data_funcs)
    post_stroke = DiseaseState('post_ischemic_stroke', cause_type='sequela',)

    susceptible.allow_self_transitions()
    data_funcs = {
        'incidence_rate': lambda _, builder: builder.data.load('cause.ischemic_stroke.incidence_rate')
    }
    susceptible.add_transition(acute_stroke, source_data_type='rate', get_data_functions=data_funcs)
    acute_stroke.allow_self_transitions()
    acute_stroke.add_transition(post_stroke)
    post_stroke.allow_self_transitions()
    data_funcs = {
        'transition_rate': lambda builder, *_: builder.data.load('cause.ischemic_stroke.incidence_rate')
    }
    post_stroke.add_transition(acute_stroke, source_data_type='rate', get_data_functions=data_funcs)

    return DiseaseModel('ischemic_stroke', states=[susceptible, acute_stroke, post_stroke])


def DiabetesMellitus():
    susceptible = SusceptibleState(project_globals.DIABETES_MELLITUS.name)
    transient = TransientDiseaseState(project_globals.DIABETES_MELLITUS.name)
    moderate = DiseaseState(f'moderate_{project_globals.DIABETES_MELLITUS.name}', cause_type='sequela',)
    severe = DiseaseState(f'severe_{project_globals.DIABETES_MELLITUS.name}', cause_type='sequela', )

    # Self transitions
    susceptible.allow_self_transitions()
    moderate.allow_self_transitions()
    severe.allow_self_transitions()

    # Transitions from Susceptible
    susceptible.add_transition(transient, source_data_type='rate')
    
    # Transitions from Transient
    data_funcs = {
        'proportion': lambda _, builder: builder.data.load(
            project_globals.DIABETES_MELLITUS.MODERATE_DIABETES_PROPORTION
        )
    }
    transient.add_transition(moderate, source_data_type='proportion', get_data_functions=data_funcs)
    data_funcs = {
        'proportion': lambda _, builder: builder.data.load(
            project_globals.DIABETES_MELLITUS.SEVERE_DIABETES_PROPORTION
        )
    }
    transient.add_transition(severe, source_data_type='proportion', get_data_functions=data_funcs)
    
    # Remission transitions
    data_funcs = {
        'transition_rate': lambda builder, *_: builder.data.load(
            project_globals.DIABETES_MELLITUS.REMISSION_RATE
        )
    }
    moderate.add_transition(susceptible, source_data_type='rate', get_data_functions=data_funcs)
    severe.add_transition(susceptible, source_data_type='rate', get_data_functions=data_funcs)

    return DiseaseModel(project_globals.DIABETES_MELLITUS.name, states=[susceptible, transient, moderate, severe])


class ChronicKidneyDisease:
    @property
    def name(self):
        return 'chronic_kidney_disease'

    def setup(self, builder: 'Builder'):
        self.population_view = builder.population.get_view([self.name])

        cause_specific_mortality_rate = builder.data.load(project_globals.IKF.CSMR)
        self.cause_specific_mortality_rate = builder.lookup.build_table(cause_specific_mortality_rate,
                                                                        key_columns=['sex'],
                                                                        parameter_columns=['age', 'year'])
        builder.value.register_value_modifier('cause_specific_mortality_rate',
                                              self.adjust_cause_specific_mortality_rate,
                                              requires_columns=['age', 'sex'])

        builder.population.initializes_simulants(self.on_initialize_simulants,
                                                 creates_columns=[self.name],
                                                 requires_values=[f'{project_globals.IKF.name}.exposure'])

        # TODO uncomment and aggregate states once disaggregation not needed
        # disability_weight_data = self.load_disability_weight_data(builder)
        self.base_disability_weight = {}
        self.disability_weight = {}
        for i, disability_weight_key in enumerate(project_globals.IKF.disability_weights):
            ikf_category = f'cat{i + 1}'
            disability_weight_data = builder.data.load(disability_weight_key)
            self.base_disability_weight[ikf_category] = builder.lookup.build_table(disability_weight_data,
                                                                                   key_columns=['sex'],
                                                                                   parameter_columns=['age', 'year'])
            self.disability_weight[ikf_category] = builder.value.register_value_producer(
                f'{project_globals.IKF_TO_CKD_MAP[ikf_category]}.disability_weight',
                source=lambda index: self.compute_disability_weight(index, ikf_category),
                requires_values=[f'{project_globals.IKF.name}.exposure']
            )

            builder.value.register_value_modifier('disability_weight', modifier=self.disability_weight[ikf_category])

        excess_mortality_data = builder.data.load(project_globals.IKF.EMR)
        self.base_excess_mortality_rate = {}
        self.excess_mortality_rate = {}
        for ikf_category, ckd_state in project_globals.IKF_TO_CKD_MAP.items():
            # Calculate EMR
            self.base_excess_mortality_rate[ikf_category] = builder.lookup.build_table(
                excess_mortality_data, key_columns=['sex'], parameter_columns=['age', 'year'])

            self.excess_mortality_rate[ikf_category] = builder.value.register_rate_producer(
                f'{ckd_state}.excess_mortality_rate',
                source=lambda index: self.compute_excess_mortality_rate(index, ikf_category),
                requires_values=[f'{project_globals.IKF.name}.exposure']
            )

            # Calculate mortality rate adjustment
            builder.value.register_value_modifier(
                'mortality_rate',
                modifier=lambda index, rates: self.adjust_mortality_rate(index, rates, ikf_category),
                requires_values=[f'{ckd_state}.excess_mortality_rate']
            )

        self.ikf_exposure = builder.value.get_value(f'{project_globals.IKF.name}.exposure')

    def on_initialize_simulants(self, pop_data: 'SimulantData'):
        exposure = self.ikf_exposure(pop_data.index)
        states = exposure.map(project_globals.IKF_TO_CKD_MAP)
        states.name = self.name
        self.population_view.update(states)

    def adjust_cause_specific_mortality_rate(self, index, rate):
        return rate + self.cause_specific_mortality_rate(index)

    # leave this here
    def load_disability_weight_data(self, builder: 'Builder'):
        dfs = []
        for i, disability_weight_key in enumerate(project_globals.IKF.disability_weights):
            col_name = f'cat{i+1}'
            df = builder.data.load(disability_weight_key)
            df = df.rename(columns={'value': col_name})
            df = df.set_index([c for c in df.columns if c != col_name])
            dfs.append(df)

        return pd.concat(dfs, axis=1).reset_index()

    def compute_disability_weight(self, index, category):
        """Gets the disability weight associated with this state.

        Parameters
        ----------
        index : iterable of ints
            An iterable of integer labels for the simulants.
        category:
            IKF category

        Returns
        -------
        `pandas.Series`
            An iterable of disability weights indexed by the provided `index`.
        """
        disability_weight = pd.Series(0, index=index)
        with_condition = index[self.ikf_exposure(index) == category]
        disability_weight.loc[with_condition] = self.base_disability_weight[category](with_condition)
        return disability_weight

    def compute_excess_mortality_rate(self, index, category):
        excess_mortality_rate = pd.Series(0, index=index)
        with_condition = index[self.ikf_exposure(index) == category]
        excess_mortality_rate.loc[with_condition] = self.base_excess_mortality_rate[category](with_condition)
        return excess_mortality_rate

    def adjust_mortality_rate(self, index, rates_df, category):
        """Modifies the baseline mortality rate for a simulant if they are in this state.

        Parameters
        ----------
        index : iterable of ints
            An iterable of integer labels for the simulants.
        category
            IKF category
        rates_df : `pandas.DataFrame`

        """
        rate = self.excess_mortality_rate[category](index, skip_post_processor=True)
        rates_df[project_globals.IKF_TO_CKD_MAP[category]] = rate
        return rates_df
