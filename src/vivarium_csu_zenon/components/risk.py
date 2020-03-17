import typing

import pandas as pd

from vivarium_public_health.risks import Risk, RiskEffect as RiskEffect_
from vivarium_public_health.risks.data_transformations import (get_distribution_type, pivot_categorical,
                                                               get_exposure_post_processor)
from vivarium_public_health.utilities import TargetString

from vivarium_csu_zenon import globals as project_globals

if typing.TYPE_CHECKING:
    from vivarium.framework.engine import Builder

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


class FastingPlasmaGlucose(Risk):
    @property
    def name(self):
        return 'fasting_plasma_glucose'

    def setup(self, builder: 'Builder'):
        propensity_col = f'{self.risk.name}_propensity'
        diabetes_state_col = project_globals.DIABETES_MELLITUS.name

        self.randomness = builder.randomness.get_stream(f'initial_{self.risk.name}_propensity')

        self.propensity = builder.value.register_value_producer(
            f'{self.risk.name}.propensity',
            source=lambda index: self.population_view.get(index)[propensity_col],
            requires_columns=[propensity_col]
        )

        self.exposure = builder.value.register_value_producer(
            f'{self.risk.name}.exposure',
            source=self.get_current_exposure,
            requires_columns=['age', 'sex', diabetes_state_col],
            requires_values=[f'{self.risk.name}.propensity'],
            preferred_post_processor=get_exposure_post_processor(builder, self.risk)
        )

        threshold_data = builder.data.load(project_globals.FPG.DIABETES_MELLITUS_THRESHOLD)
        self.fpg_threshold = builder.lookup.build_table(threshold_data, key_columns=['sex'],
                                                        parameter_columns=['age', 'year'])

        builder.population.initializes_simulants(self.on_initialize_simulants, creates_columns=[propensity_col],
                                                 requires_streams=[f'initial_{self.risk.name}_propensity'])

        self.population_view = builder.population.get_view([propensity_col, diabetes_state_col])

    def get_current_exposure(self, index):
        pop = self.population_view.get(index)
        diabetes_state = pop.loc[:, project_globals.DIABETES_MELLITUS.name]
        diabetic_mask = diabetes_state != project_globals.DIABETES_MELLITUS_SUSCEPTIBLE_STATE_NAME
        raw_propensity = self.propensity(index)
        thresholds = self.fpg_threshold(index)

        propensity = pd.Series(0, index=index)
        propensity.loc[~diabetic_mask] = raw_propensity.loc[~diabetic_mask] * thresholds.loc[~diabetic_mask]
        propensity.loc[diabetic_mask] = (thresholds.loc[diabetic_mask]
                                         + raw_propensity.loc[diabetic_mask] * (1 - thresholds.loc[diabetic_mask]))

        return pd.Series(self.exposure_distribution.ppf(propensity), index=index)
