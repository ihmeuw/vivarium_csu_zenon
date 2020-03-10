import pandas as pd

from gbd_mapping import risk_factors
from vivarium_public_health.risks import Risk, RiskEffect as RiskEffect_
from vivarium_public_health.risks.data_transformations import get_distribution_type, pivot_categorical
from vivarium_public_health.utilities import TargetString

from vivarium_csu_zenon import globals as project_globals

TARGET_MAP = {
    'sequela.acute_myocardial_infarction.incidence_rate': TargetString(project_globals.IHD.ACUTE_MI_INCIDENCE_RATE),
    'sequela.post_myocardial_infarction_to_acute_myocardial_infarction.transition_rate': TargetString(project_globals.IHD.ACUTE_MI_INCIDENCE_RATE),
    'sequela.acute_ischemic_stroke.incidence_rate': TargetString(project_globals.ISCHEMIC_STROKE.ACUTE_STROKE_INCIDENCE_RATE),
    'sequela.post_ischemic_stroke_to_acute_ischemic_stroke.transition_rate': TargetString(project_globals.ISCHEMIC_STROKE.ACUTE_STROKE_INCIDENCE_RATE),
}


class RiskEffect(RiskEffect_):
    def load_relative_risk_data(self, builder):
        relative_risk_data = builder.data.load(f'{self.risk}.relative_risk')
        correct_target = ((relative_risk_data['affected_entity'] == TARGET_MAP[self.target].name)
                          & (relative_risk_data['affected_measure'] == TARGET_MAP[self.target].measure))
        relative_risk_data = (relative_risk_data[correct_target]
                              .drop(['affected_entity', 'affected_measure'], 'columns'))

        if get_distribution_type(builder, self.risk) in ['dichotomous', 'ordered_polytomous', 'unordered_polytomous']:
            relative_risk_data = pivot_categorical(relative_risk_data)

        else:
            relative_risk_data = relative_risk_data.drop(['parameter'], 'columns')
        return relative_risk_data

    def load_population_attributable_fraction_data(self, builder):
        paf_data = builder.data.load(f'{self.risk}.population_attributable_fraction')
        correct_target = ((paf_data['affected_entity'] == TARGET_MAP[self.target].name)
                          & (paf_data['affected_measure'] == TARGET_MAP[self.target].measure))
        paf_data = (paf_data[correct_target]
                    .drop(['affected_entity', 'affected_measure'], 'columns'))
        return paf_data
