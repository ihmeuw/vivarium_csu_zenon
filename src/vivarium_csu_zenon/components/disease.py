import pandas as pd
from vivarium_public_health.disease import (DiseaseState as DiseaseState_, DiseaseModel, SusceptibleState,
                                            RateTransition as RateTransition_)


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
    acute_mi = DiseaseState('acute_myocardial_infarction',
                            cause_type='sequela',
                            get_data_functions=data_funcs)
    post_mi = DiseaseState('post_myocardial_infarction',
                           cause_type='sequela',)

    susceptible.allow_self_transitions()
    data_funcs = {
        'incidence_rate': lambda _, builder: builder.data.load('cause.ischemic_heart_disease.incidence_rate')
    }
    susceptible.add_transition(acute_mi,
                               source_data_type='rate',
                               get_data_functions=data_funcs)
    acute_mi.allow_self_transitions()
    acute_mi.add_transition(post_mi)
    post_mi.allow_self_transitions()
    data_funcs = {
        'transition_rate': lambda builder, *_: builder.data.load('cause.ischemic_heart_disease.incidence_rate')
    }
    post_mi.add_transition(acute_mi, source_data_type='rate', get_data_functions=data_funcs)

    return DiseaseModel('ischemic_heart_disease', states=[susceptible, acute_mi, post_mi])


# def Stroke(stroke_name):
#     stroke_types = ['ischemic_stroke', 'subarachnoid_hemorrhage', 'intracerebral_hemorrhage']
#     if stroke_name not in stroke_types:
#         raise ValueError(f'Stroke name must be one of {stroke_types}.  You supplied {stroke_name}')
#
#     susceptible = SusceptibleState(stroke_name)
#     data_funcs = {'dwell_time': lambda *args: pd.Timedelta(days=28)}
#     acute = DiseaseState(f'acute_{stroke_name}',
#                          cause_type='sequela',
#                          get_data_functions=data_funcs)
#     #data_funcs = {'dwell_time': lambda *args: pd.Timedelta(days=365.25)}
#     post = RelapseState(f'post_{stroke_name}',
#                         cause_type='sequela')
#
#     susceptible.allow_self_transitions()
#     data_funcs = {
#         'incidence_rate': lambda _, builder: builder.data.load(f'cause.{stroke_name}.incidence_rate')
#     }
#     susceptible.add_transition(acute,
#                                source_data_type='rate',
#                                get_data_functions=data_funcs)
#     acute.allow_self_transitions()
#     acute.add_transition(post)
#     post.allow_self_transitions()
#     data_funcs = {
#         'relapse_rate': lambda _, builder: builder.data.load(f'cause.{stroke_name}.incidence_rate')
#     }
#     post.add_transition(acute, source_data_type='rate', get_data_functions=data_funcs)
#     #post.add_transition(susceptible, source_data_type='time')
#
#     return DiseaseModel(stroke_name, states=[susceptible, acute, post])