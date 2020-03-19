import typing

import pandas as pd
from vivarium_public_health.disease import (DiseaseState as DiseaseState_, DiseaseModel, SusceptibleState,
                                            TransientDiseaseState, RateTransition as RateTransition_)

from vivarium_csu_zenon import globals as project_globals

if typing.TYPE_CHECKING:
    from vivarium.framework.engine import Builder
    from vivarium.framework.population import SimulantData
    from vivarium.framework.event import Event


class CVDRiskAttribute:
    CVD_VERY_HIGH_RISK = 'very_high_risk'
    CVD_HIGH_RISK = 'high_risk'
    CVD_MODERATE_RISK = 'moderate_risk'
    CVD_LOW_RISK = 'low_risk'

    DIABETES_STATE_COL = project_globals.DIABETES_MELLITUS.name
    CKD_STATE_COL = project_globals.CKD_MODEL_NAME

    @property
    def name(self):
        return 'cvd_risk_attribute'

    def setup(self, builder: 'Builder'):

        self.systolic_blood_pressure = builder.value.get_value('high_systolic_blood_pressure.exposure')

        self.cvd_risk_score = builder.value.register_value_producer(
            'cvd_risk_score',
            source=self.get_cvd_risk_score,
            requires_columns=['age', 'sex'],
            requires_values=['high_systolic_blood_pressure.exposure']
        )

        self.cvd_risk_category = builder.value.register_value_producer(
            'cvd_risk_category',
            source=self.get_cvd_risk_category,
            requires_columns=['age', 'sex', self.DIABETES_STATE_COL, self.CKD_STATE_COL],
            requires_values=['cvd_risk_score', 'high_systolic_blood_pressure.exposure']
        )

        self.population_view = builder.population.get_view(['age', 'sex', self.DIABETES_STATE_COL, self.CKD_STATE_COL])

    def get_cvd_risk_score(self, index):
        pop = self.population_view.get(index)
        sbp = self.systolic_blood_pressure(index)
        age = pop.loc[:, 'age']
        sex = pop.loc[:, 'sex']
        score = -16.5 + 0.043 * sbp + 0.266 * age + 2.32 * sex
        return score

    def get_cvd_risk_category(self, index):
        pop = self.population_view.get(index)
        diabetes_state = pop.loc[:, self.DIABETES_STATE_COL]
        ckd_state = pop.loc[:, self.CKD_STATE_COL]
        cvd_risk_score = self.cvd_risk_score(index)
        sbp = self.systolic_blood_pressure(index)

        very_high_risk_mask = (diabetes_state == project_globals.SEVERE_DIABETES_MELLITUS_STATE_NAME
                               or ckd_state == project_globals.STAGE_V_CKD_STATE_NAME
                               or cvd_risk_score >= 10)
        high_risk_mask = (5 <= cvd_risk_score < 10
                          and (diabetes_state == project_globals.MODERATE_DIABETES_MELLITUS_STATE_NAME
                               or ckd_state in [project_globals.ALBUMINURIA_STATE_NAME,
                                                project_globals.STAGE_III_CKD_STATE_NAME,
                                                project_globals.STAGE_IV_CKD_STATE_NAME]
                               or sbp > 180))
        moderate_risk_mask = 1 <= cvd_risk_score < 5
        low_risk_mask = cvd_risk_score < 1

        cvd_risk_categories = pd.Series(0, index=index)
        cvd_risk_categories[very_high_risk_mask] = self.CVD_VERY_HIGH_RISK
        cvd_risk_categories[high_risk_mask] = self.CVD_HIGH_RISK
        cvd_risk_categories[moderate_risk_mask] = self.CVD_MODERATE_RISK
        cvd_risk_categories[low_risk_mask] = self.CVD_LOW_RISK
        return cvd_risk_categories
