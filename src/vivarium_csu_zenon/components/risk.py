import typing

import pandas as pd

from vivarium_public_health.risks import Risk as Risk_, RiskEffect as RiskEffect_
from vivarium_public_health.risks.data_transformations import (get_distribution_type, pivot_categorical,
                                                               get_exposure_post_processor)

from vivarium_csu_zenon import globals as project_globals

if typing.TYPE_CHECKING:
    from vivarium.framework.engine import Builder


class RiskEffect(RiskEffect_):

    def setup(self, builder):
        self.randomness = builder.randomness.get_stream(
            f'effect_of_{self.risk.name}_on_{self.target.name}.{self.target.measure}'
        )

        relative_risk_data = self.load_relative_risk_data(builder)
        self.relative_risk = builder.lookup.build_table(relative_risk_data, key_columns=['sex'],
                                                        parameter_columns=['age', 'year'])
        self.exposure_effect = self.load_exposure_effect(builder)

        builder.value.register_value_modifier(f'{self.target.name}.{self.target.measure}',
                                              modifier=self.adjust_target,
                                              requires_values=[f'{self.risk.name}.exposure'],
                                              requires_columns=['age', 'sex'])

    def load_relative_risk_data(self, builder):
        relative_risk_data = builder.data.load(f'{self.risk}.relative_risk')
        correct_target = ((relative_risk_data['affected_entity']
                           == project_globals.RATE_TARGET_MAP[self.target].name)
                          & (relative_risk_data['affected_measure']
                             == project_globals.RATE_TARGET_MAP[self.target].measure))
        relative_risk_data = (relative_risk_data[correct_target]
                              .drop(['affected_entity', 'affected_measure'], 'columns'))

        if get_distribution_type(builder, self.risk) in ['dichotomous', 'ordered_polytomous', 'unordered_polytomous']:
            relative_risk_data = pivot_categorical(relative_risk_data)

        else:
            relative_risk_data = relative_risk_data.drop(['parameter'], 'columns')
        return relative_risk_data


class Risk(Risk_):

    def setup(self, builder: 'Builder'):
        propensity_col = f'{self.risk.name}_propensity'
        self.population_view = builder.population.get_view([propensity_col])
        self.propensity = builder.value.register_value_producer(
            f'{self.risk.name}.propensity',
            source=lambda index: self.population_view.get(index)[propensity_col],
            requires_columns=[propensity_col])
        self.exposure = builder.value.register_value_producer(
            f'{self.risk.name}.exposure',
            source=self.get_current_exposure,
            requires_columns=['age', 'sex'],
            requires_values=[f'{self.risk.name}.propensity'],
            preferred_post_processor=get_exposure_post_processor(builder, self.risk)
        )


class LDLCholesterolRisk(Risk_):

    def setup(self, builder: 'Builder'):
        propensity_col = f'{self.risk.name}_propensity'
        self.population_view = builder.population.get_view([propensity_col])
        self.propensity = builder.value.register_value_producer(
            f'{self.risk.name}.propensity',
            source=lambda index: self.population_view.get(index)[propensity_col],
            requires_columns=[propensity_col])
        # Need a separate hook to avoid a cyclic dependency at initialization.
        self.base_exposure = builder.value.register_value_producer(
            f'{self.risk.name}.base_exposure',
            source=self.get_current_exposure,
            requires_columns=['age', 'sex'],
            requires_values=[f'{self.risk.name}.propensity'],
            preferred_post_processor=get_exposure_post_processor(builder, self.risk)
        )
        self.exposure = builder.value.register_value_producer(f'{self.risk.name}.exposure', source=self.base_exposure)


class FastingPlasmaGlucose(Risk):
    @property
    def name(self):
        return f'risk_factor.{project_globals.FPG.name}'

    def __init__(self):
        super().__init__(self.name)

    def setup(self, builder: 'Builder'):
        super().setup(builder)
        diabetes_state_col = project_globals.DIABETES_MELLITUS.name

        threshold_data = builder.data.load(project_globals.FPG.DIABETES_MELLITUS_THRESHOLD)
        self.fpg_threshold = builder.lookup.build_table(threshold_data, key_columns=['sex'],
                                                        parameter_columns=['age', 'year'])
        propensity_col = f'{self.risk.name}_propensity'
        self.population_view = builder.population.get_view([diabetes_state_col, propensity_col])

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
