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

        self.p_tx_given_bad_ldlc, self.p_target_given_treated = self.load_treatment_and_target_p(builder)
        self.p_therapy_type = self.load_therapy_type_p(builder)
        self.p_treatment_type = self.load_treatment_type_p(builder)

        columns_created = [parameters.STATIN_HIGH, parameters.STATIN_LOW, parameters.EZETIMIBE,
                           parameters.FIBRATES, parameters.LIFESTYLE, parameters.FDC]
        self.population_view = builder.population.get_view(columns_created)
        builder.population.initializes_simulants(self.on_initialize_simulants,
                                                 creates_columns=columns_created,
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
        low_dose_if_statin = (self.randomness.get_draw(pop_data.index, additional_key='statin_dose')
                              < parameters.PROB_LOW_DOSE)

        # potency_statin_dose
        high_statin = ((treated & mono_if_treated & high_statin_if_mono)
                       | (treated & ~mono_if_treated & ~fdc_if_multi & ~low_potency_statin_if_not_fdc))
        low_statin = ((treated & mono_if_treated & low_statin_if_mono)
                      | (treated & ~mono_if_treated & ~fdc_if_multi & low_potency_statin_if_not_fdc))
        # potency_statin_dose
        high_statin_high = high_statin & ~low_dose_if_statin
        high_statin_low = high_statin & low_dose_if_statin
        low_statin_high = low_statin & ~low_dose_if_statin
        low_statin_low = low_statin & low_dose_if_statin

        ezetimibe = ((treated & mono_if_treated & ezetimibe_if_mono)
                     | (treated & ~mono_if_treated))

        fibrates = treated & mono_if_treated & fibrates_if_mono

        fdc = treated & ~mono_if_treated & fdc_if_multi

        pop_update.loc[high_statin_high, parameters.STATIN_HIGH] = 'high'
        pop_update.loc[high_statin_low, parameters.STATIN_HIGH] = 'low'
        pop_update.loc[low_statin_high, parameters.STATIN_LOW] = 'high'
        pop_update.loc[low_statin_low, parameters.STATIN_LOW] = 'low'
        pop_update.loc[ezetimibe, parameters.EZETIMIBE] = True
        pop_update.loc[fibrates, parameters.FIBRATES] = True
        pop_update.loc[fdc, parameters.FDC] = True
        self.population_view.update(pop_update)

    def get_treatment_probability(self, ldlc):
        high_ldlc = ldlc > parameters.HIGH_LDL_BASELINE
        # FIXME: Generate data, this is an awful hack for small sample sizes and not age/sex specific.
        p_high_ldlc = len(ldlc[ldlc > parameters.HIGH_LDL_BASELINE])/len(ldlc)
        p_bad_ldlc = p_high_ldlc / (1 - self.p_target_given_treated * self.p_tx_given_bad_ldlc)

        p_treated_low = (self.p_target_given_treated * self.p_tx_given_bad_ldlc * p_bad_ldlc
                         / (1 - p_high_ldlc))
        p_treated_high = ((1 - self.p_target_given_treated) * self.p_tx_given_bad_ldlc * p_bad_ldlc
                          / p_high_ldlc)
        treatment_probability = pd.Series(p_treated_low, index=ldlc.index)
        treatment_probability.loc[high_ldlc] = p_treated_high
        return treatment_probability

    @staticmethod
    def load_treatment_and_target_p(builder):
        location = builder.configuration.input_data.location
        draw = builder.configuration.input_data.input_draw_number
        p_tx_given_bad_ldlc = parameters.sample_probability_rx_given_high_ldl_c(location, draw)
        p_target_among_treated = parameters.sample_probability_target_given_rx(location, draw)
        return p_tx_given_bad_ldlc, p_target_among_treated

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


class LDLCTreatmentAdherence:

    @property
    def name(self):
        return 'ldlc_treatment_adherence'

    def setup(self, builder: 'Builder'):
        self.p_adherent = self.load_adherence_p(builder)

        self.columns_required = [project_globals.IHD_MODEL_NAME,
                                 project_globals.ISCHEMIC_STROKE_MODEL_NAME,
                                 parameters.STATIN_LOW, parameters.STATIN_HIGH,
                                 parameters.FIBRATES, parameters.EZETIMIBE, parameters.FDC]

        self.population_view = builder.population.get_view(self.columns_required
                                                           + [f'{self.name}_propensity'])
        self.randomness = builder.randomness.get_stream(self.name)
        builder.population.initializes_simulants(self.on_initialize_simulants,
                                                 creates_columns=[f'{self.name}_propensity'],
                                                 requires_streams=[self.name])

        builder.value.register_value_producer(self.name, self.is_adherent,
                                              requires_columns=self.columns_required + [f'{self.name}_propensity'])

    def on_initialize_simulants(self, pop_data: 'SimulantData'):
        self.population_view.update(pd.Series(self.randomness.get_draw(pop_data.index),
                                              index=pop_data.index,
                                              name=f'{self.name}_propensity'))

    def is_adherent(self, index: pd.Index):
        propensity = self.population_view.subview([f'{self.name}_propensity']).get(index)
        return self.determine_adherent(propensity)

    def determine_adherent(self, propensity):
        # Wrote as separate function cause I thought I needed for
        # initialization too. Leave for now in case I'm dumb.  - J.C.
        p_adherent = pd.Series(0, index=propensity.index)

        pop_status = self.population_view.subview(self.columns_required).get(propensity.index)
        ihd = pop_status[project_globals.IHD_MODEL_NAME] != project_globals.IHD_SUSCEPTIBLE_STATE_NAME
        stroke = (pop_status[project_globals.ISCHEMIC_STROKE_MODEL_NAME]
                  != project_globals.ISCHEMIC_STROKE_SUSCEPTIBLE_STATE_NAME)
        on_statin = (pop_status[parameters.STATIN_HIGH] != 'none') | (pop_status[parameters.STATIN_LOW] != 'none')
        num_drugs = sum([on_statin, pop_status[parameters.FIBRATES], pop_status[parameters.EZETIMIBE]])
        on_fdc = pop_status[parameters.FDC]

        had_cve = ihd | stroke
        single_med = (num_drugs == 1) | on_fdc
        multi_med = num_drugs > 1

        p_adherent.loc[~had_cve & single_med] = self.p_adherent[parameters.SINGLE_NO_CVE]
        p_adherent.loc[~had_cve & multi_med] = self.p_adherent[parameters.MULTI_NO_CVE]
        p_adherent.loc[had_cve & single_med] = self.p_adherent[parameters.SINGLE_CVE]
        p_adherent.loc[had_cve & multi_med] = self.p_adherent[parameters.MULTI_CVE]
        return propensity < p_adherent

    @staticmethod
    def load_adherence_p(builder):
        location = builder.configuration.input_data.location
        draw = builder.configuration.input_data.input_draw_number
        p_adherent = {group: parameters.sample_adherence(location, draw, *group)
                      for group in [parameters.SINGLE_NO_CVE, parameters.MULTI_NO_CVE,
                                    parameters.SINGLE_CVE, parameters.MULTI_CVE]}
        return p_adherent


