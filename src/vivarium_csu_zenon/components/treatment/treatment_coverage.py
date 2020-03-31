import typing

import pandas as pd

from vivarium_csu_zenon import globals as project_globals
from vivarium_csu_zenon.components.treatment import parameters

if typing.TYPE_CHECKING:
    from vivarium.framework.engine import Builder
    from vivarium.framework.population import SimulantData


# NOTE:
# `bad_ldlc` is defined here as a person's untreated ldlc is above the
# ldlc threshold, regardless of what their actual ldlc level is.

class LDLCTreatmentCoverage:

    @property
    def name(self):
        """The name of this component."""
        return 'ldlc_treatment_coverage'

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: 'Builder'):
        """Perform this component's setup."""
        self.randomness = builder.randomness.get_stream(self.name)

        self.ldlc = builder.value.get_value('high_ldl_cholesterol.exposure')

        self.p_rx_given_bad_ldlc, self.p_at_target_given_treated = self.load_treatment_and_target_p(builder)
        self.p_therapy_type = self.load_therapy_type_p(builder)
        self.p_treatment_type = self.load_treatment_type_p(builder)

        columns_created = [parameters.STATIN_HIGH, parameters.STATIN_LOW, parameters.EZETIMIBE,
                           parameters.FIBRATES, parameters.LIFESTYLE, parameters.FDC]
        self.population_view = builder.population.get_view(columns_created)
        builder.population.initializes_simulants(self.on_initialize_simulants,
                                                 creates_columns=columns_created,
                                                 requires_columns=[project_globals.ISCHEMIC_STROKE_MODEL_NAME,
                                                                   project_globals.IHD_MODEL_NAME],
                                                 requires_values=['high_ldl_cholesterol.exposure'],
                                                 requires_streams=[self.name])

    def on_initialize_simulants(self, pop_data: 'SimulantData'):
        """Initialize treatment status"""
        pop_update = pd.DataFrame({
            parameters.STATIN_HIGH: 'none',  # 'none', 'low', 'high'
            parameters.STATIN_LOW: 'none',  # 'none', 'low', 'high'
            parameters.EZETIMIBE: False,
            parameters.FIBRATES: False,
            parameters.LIFESTYLE: False,
            parameters.FDC: False,
        })

        ldlc = self.ldlc(pop_data.index)
        treatment_probability = self.get_treatment_probability(ldlc)
        treated = self.randomness.get_draw(pop_data.index) < treatment_probability
        mono_if_treated = (self.randomness.get_draw(pop_data.index, additional_key='therapy_type')
                           < self.p_therapy_type[parameters.MONOTHERAPY])
        treatment_type_if_mono = self.randomness.choice(pop_data.index,
                                                        list(self.p_treatment_type.keys()),
                                                        list(self.p_treatment_type.values()),
                                                        additional_key='treatment_type')
        ezetimibe_if_mono = treatment_type_if_mono == parameters.EZETIMIBE
        fibrates_if_mono = treatment_type_if_mono == parameters.FIBRATES
        low_statin_if_mono = treatment_type_if_mono == parameters.STATIN_LOW
        high_statin_if_mono = treatment_type_if_mono == parameters.STATIN_HIGH
        fdc_if_multi = (self.randomness.get_draw(pop_data.index, additional_key='fdc')
                        < self.p_therapy_type[parameters.FDC])
        p_low = (self.p_treatment_type[parameters.STATIN_LOW]
                 / (self.p_treatment_type[parameters.STATIN_LOW] + self.p_treatment_type[parameters.STATIN_HIGH]))
        # noinspection PyTypeChecker
        low_potency_statin_if_not_fdc = (self.randomness.get_draw(pop_data.index, additional_key='low_potency_statin')
                                         < p_low)

        # potency_statin_dose
        high_statin_not_fdc = (treated & ((mono_if_treated & high_statin_if_mono)
                                          | (~mono_if_treated & ~fdc_if_multi & ~low_potency_statin_if_not_fdc)))
        low_statin_not_fdc = (treated & ((mono_if_treated & low_statin_if_mono)
                                         | (~mono_if_treated & ~fdc_if_multi & low_potency_statin_if_not_fdc)))
        fdc = treated & ~mono_if_treated & fdc_if_multi

        # potency_statin_dose
        # TODO: figure out the appropriate dosing
        # TODO: figure out the statin potency for fdc
        high_statin_low_dose = high_statin_not_fdc | fdc
        low_statin_high_dose = low_statin_not_fdc

        ezetimibe = treated & ~(mono_if_treated & ~ezetimibe_if_mono)
        fibrates = treated & mono_if_treated & fibrates_if_mono

        pop_update.loc[high_statin_low_dose, parameters.STATIN_HIGH] = 'low'
        pop_update.loc[low_statin_high_dose, parameters.STATIN_LOW] = 'high'
        pop_update.loc[ezetimibe, parameters.EZETIMIBE] = True
        pop_update.loc[fibrates, parameters.FIBRATES] = True
        pop_update.loc[fdc, parameters.FDC] = True
        self.population_view.update(pop_update)

    def get_treatment_probability(self, ldlc):
        high_ldlc = ldlc > parameters.HIGH_LDL_BASELINE
        # FIXME: Generate data, this is an awful hack for small sample sizes and not age/sex specific.
        p_high_ldlc = len(ldlc[ldlc > parameters.HIGH_LDL_BASELINE])/len(ldlc)
        p_bad_ldlc = p_high_ldlc / (1 - self.p_at_target_given_treated * self.p_rx_given_bad_ldlc)

        p_treated_low = (self.p_at_target_given_treated * self.p_rx_given_bad_ldlc * p_bad_ldlc
                         / (1 - p_high_ldlc))
        p_treated_high = ((1 - self.p_at_target_given_treated) * self.p_rx_given_bad_ldlc * p_bad_ldlc
                          / p_high_ldlc)
        treatment_probability = pd.Series(p_treated_low, index=ldlc.index)
        treatment_probability.loc[high_ldlc] = p_treated_high
        return treatment_probability

    @staticmethod
    def load_treatment_and_target_p(builder):
        location = builder.configuration.input_data.location
        draw = builder.configuration.input_data.input_draw_number
        p_rx_given_bad_ldlc = parameters.sample_probability_rx_given_high_ldl_c(location, draw)
        p_at_target_given_treated = parameters.sample_probability_target_given_rx(location, draw)
        return p_rx_given_bad_ldlc, p_at_target_given_treated

    @staticmethod
    def load_therapy_type_p(builder):
        location = builder.configuration.input_data.location
        draw = builder.configuration.input_data.input_draw_number
        p_therapy_type = {therapy_type: parameters.sample_therapy_type(location, draw, therapy_type)
                          for therapy_type in [parameters.MONOTHERAPY, parameters.FDC]}
        return p_therapy_type

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



