import typing

import pandas as pd

from vivarium_csu_zenon import globals as project_globals

if typing.TYPE_CHECKING:
    from vivarium.framework.engine import Builder


class CVDRiskAttribute:
    """This component creates two pipelines that define cardiovascular risk.

    Cardiovascular disease (CVD) risk can be measured as a score that
    incorporates age, sex, and blood pressure.  It can also be measured as a
    categorical label in
    ['low_risk', 'moderate_risk', 'high_risk', and 'very_high_risk']
    that takes into account the CVD risk score as well as a person's current
    condition with respect to diabetes and chronic kidney disease.

    Both measures of CVD risk are computed and exposed as pipelines by this
    component.  Note that this is not a 'risk' in the normal sense.  It has
    no direct effects on other attributes in the model. It is, however,
    a scoring mechanism used by doctors to determine treatment for high
    cholesterol (LDL-C) levels.

    """
    DIABETES_STATE_COL = project_globals.DIABETES_MELLITUS.name

    @property
    def name(self) -> str:
        """The name of this component."""
        return 'cvd_risk_attribute'

    def setup(self, builder: 'Builder'):
        self.systolic_blood_pressure = builder.value.get_value('high_systolic_blood_pressure.exposure')
        self.impaired_kidney_function = builder.value.get_value('impaired_kidney_function.exposure')

        self.cvd_risk_score = builder.value.register_value_producer(
            'cvd_risk_score',
            source=self.get_cvd_risk_score,
            requires_columns=['age', 'sex'],
            requires_values=['high_systolic_blood_pressure.exposure']
        )

        self.cvd_risk_category = builder.value.register_value_producer(
            'cvd_risk_category',
            source=self.get_cvd_risk_category,
            requires_columns=['age', 'sex', self.DIABETES_STATE_COL],
            requires_values=['cvd_risk_score',
                             'high_systolic_blood_pressure.exposure',
                             'impaired_kidney_function.exposure']
        )

        self.population_view = builder.population.get_view(['age', 'sex', self.DIABETES_STATE_COL])

    def get_cvd_risk_score(self, index: pd.Index) -> pd.Series:
        """Source for the cvd risk score pipeline."""
        pop = self.population_view.get(index)
        sbp = self.systolic_blood_pressure(index)
        age = pop.loc[:, 'age']
        sex = pop.loc[:, 'sex'] == 'Male'
        score = -16.5 + 0.043 * sbp + 0.266 * age + 2.32 * sex
        return score

    def get_cvd_risk_category(self, index):
        """Source for the cvd risk category pipeline."""
        pop = self.population_view.get(index)
        diabetes_state = pop.loc[:, self.DIABETES_STATE_COL]
        cvd_risk_score = self.cvd_risk_score(index)
        sbp = self.systolic_blood_pressure(index)
        ikf = self.impaired_kidney_function(index)

        high_sbp_mask = sbp > 180
        moderate_diabetes_mask = diabetes_state == project_globals.MODERATE_DIABETES_MELLITUS_STATE_NAME
        moderate_ckd_mask = ((ikf == project_globals.CKD_IKF_MAP[project_globals.ALBUMINURIA_STATE_NAME])
                             | (ikf == project_globals.CKD_IKF_MAP[project_globals.STAGE_III_CKD_STATE_NAME])
                             | (ikf == project_globals.CKD_IKF_MAP[project_globals.STAGE_IV_CKD_STATE_NAME]))
        low_non_cvd_risk_mask = ((diabetes_state == project_globals.DIABETES_MELLITUS_SUSCEPTIBLE_STATE_NAME)
                                 & (ikf == project_globals.CKD_IKF_MAP[project_globals.CKD_SUSCEPTIBLE_STATE_NAME])
                                 & ~high_sbp_mask)

        very_high_risk_mask = ((diabetes_state == project_globals.SEVERE_DIABETES_MELLITUS_STATE_NAME)
                               | (ikf == project_globals.CKD_IKF_MAP[project_globals.STAGE_V_CKD_STATE_NAME])
                               | (cvd_risk_score >= 10))
        high_risk_mask = (moderate_diabetes_mask
                          | moderate_ckd_mask
                          | high_sbp_mask
                          | (5 <= cvd_risk_score) & (cvd_risk_score < 10))
        moderate_risk_mask = low_non_cvd_risk_mask & (1 <= cvd_risk_score) & (cvd_risk_score < 5)
        low_risk_mask = low_non_cvd_risk_mask & (cvd_risk_score < 1)

        cvd_risk_categories = pd.Series(0, index=index)
        cvd_risk_categories[very_high_risk_mask] = project_globals.CVD_VERY_HIGH_RISK
        cvd_risk_categories[high_risk_mask] = project_globals.CVD_HIGH_RISK
        cvd_risk_categories[moderate_risk_mask] = project_globals.CVD_MODERATE_RISK
        cvd_risk_categories[low_risk_mask] = project_globals.CVD_LOW_RISK
        return cvd_risk_categories
