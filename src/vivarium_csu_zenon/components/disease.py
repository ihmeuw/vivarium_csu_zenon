import pandas as pd
from vivarium_public_health.disease import (DiseaseState as DiseaseState_, DiseaseModel, SusceptibleState,
                                            TransientDiseaseState, RateTransition as RateTransition_)

from vivarium_csu_zenon import globals as project_globals


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

    susceptible.allow_self_transitions()
    data_funcs = {
        'incidence_rate': lambda _, builder: builder.data.load(
            project_globals.DIABETES_MELLITUS.INCIDENCE_RATE
        )
    }
    susceptible.add_transition(transient, source_data_type='rate', get_data_functions=data_funcs)
    data_funcs = {
        'proportion': lambda _, builder: builder.data.load(
            project_globals.DIABETES_MELLITUS.MODERATE_DIABETES_PROPORTION
        )
    }
    transient.add_transition(moderate, source_data_type='proportion', get_data_functions=data_funcs)
    moderate.allow_self_transitions()
    data_funcs = {
        'proportion': lambda _, builder: builder.data.load(
            project_globals.DIABETES_MELLITUS.SEVERE_DIABETES_PROPORTION
        )
    }
    transient.add_transition(severe, source_data_type='proportion', get_data_functions=data_funcs)
    severe.allow_self_transitions()

    return DiseaseModel(project_globals.DIABETES_MELLITUS.name, states=[susceptible, transient, moderate, severe])


def ChronicKidneyDisease():
    # States
    susceptible = SusceptibleState(project_globals.CKD.name)
    transient = TransientDiseaseState(project_globals.CKD.name)
    albuminuria = DiseaseState('albuminuria', cause_type='sequela',)
    stage_iii = DiseaseState(f'stage_iii_{project_globals.CKD.name}', cause_type='sequela', )
    stage_iv = DiseaseState(f'stage_iv_{project_globals.CKD.name}', cause_type='sequela', )
    stage_v = DiseaseState(f'stage_v_{project_globals.CKD.name}', cause_type='sequela', )

    # Susceptible transitions
    susceptible.allow_self_transitions()
    data_funcs = {'incidence_rate': lambda _, builder: builder.data.load(project_globals.CKD.INCIDENCE_RATE)}
    susceptible.add_transition(transient, source_data_type='rate', get_data_functions=data_funcs)

    # Transient transitions
    data_funcs = {'proportion': lambda _, builder: builder.data.load(project_globals.CKD.ALBUMINURIA_PROPORTION)}
    transient.add_transition(albuminuria, source_data_type='proportion', get_data_functions=data_funcs)
    data_funcs = {'proportion': lambda _, builder: builder.data.load(project_globals.CKD.STAGE_III_CKD_PROPORTION)}
    transient.add_transition(stage_iii, source_data_type='proportion', get_data_functions=data_funcs)
    data_funcs = {'proportion': lambda _, builder: builder.data.load(project_globals.CKD.STAGE_IV_CKD_PROPORTION)}
    transient.add_transition(stage_iv, source_data_type='proportion', get_data_functions=data_funcs)
    data_funcs = {'proportion': lambda _, builder: builder.data.load(project_globals.CKD.STAGE_V_CKD_PROPORTION)}
    transient.add_transition(stage_v, source_data_type='proportion', get_data_functions=data_funcs)

    # Other transitions
    albuminuria.allow_self_transitions()
    stage_iii.allow_self_transitions()
    stage_iv.allow_self_transitions()
    stage_v.allow_self_transitions()

    return DiseaseModel(
        project_globals.CKD.name,
        states=[susceptible, transient, albuminuria, stage_iii, stage_iv, stage_v]
    )
