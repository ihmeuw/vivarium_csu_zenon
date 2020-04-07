"""Baseline coverage and effect of ldlc treatment."""
import typing
from typing import Dict

import pandas as pd
from vivarium.framework.randomness import get_hash

from vivarium_csu_zenon import globals as project_globals
from vivarium_csu_zenon.components.treatment import parameters
from vivarium_csu_zenon.utilities import sample_truncnorm_distribution

if typing.TYPE_CHECKING:
    from vivarium.framework.engine import Builder
    from vivarium.framework.population import SimulantData
    from vivarium.framework.event import Event


# NOTE:
# `bad_ldlc` is defined here as a person's untreated ldlc is above the
# ldlc threshold, regardless of what their actual ldlc level is.

class LDLCTreatmentCoverage:
    """Manages the baseline coverage of ldlc meds."""

    @property
    def name(self) -> str:
        """The name of this component."""
        return 'ldlc_treatment_coverage'

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: 'Builder'):
        """Perform this component's setup."""
        self.randomness = builder.randomness.get_stream(self.name)

        self.ldlc = builder.value.get_value('high_ldl_cholesterol.base_exposure')

        self.p_rx_given_bad_ldlc, self.p_at_target_given_treated = self.load_treatment_and_target_p(builder)
        self.p_therapy_type = self.load_therapy_type_p(builder)
        self.p_treatment_type = self.load_treatment_type_p(builder)

        columns_created = [project_globals.TREATMENT.name]
        self.population_view = builder.population.get_view(columns_created)
        builder.population.initializes_simulants(self.on_initialize_simulants,
                                                 creates_columns=columns_created,
                                                 requires_values=['high_ldl_cholesterol.base_exposure'],
                                                 requires_streams=[self.name])

    def on_initialize_simulants(self, pop_data: 'SimulantData'):
        """Initialize treatment status"""
        pop_update = pd.Series(project_globals.TREATMENT.none, index=pop_data.index, name=project_globals.TREATMENT.name)

        ldlc = self.ldlc(pop_data.index)
        treatment_probability = self.get_treatment_probability(ldlc)
        treated = self.randomness.get_draw(pop_data.index) < treatment_probability

        # noinspection PyTypeChecker
        mono_if_treated = (self.randomness.get_draw(pop_data.index, additional_key='therapy_type')
                           < self.p_therapy_type[parameters.MONOTHERAPY])
        treatment_type_if_mono = self.randomness.choice(pop_data.index,
                                                        list(self.p_treatment_type.keys()),
                                                        list(self.p_treatment_type.values()),
                                                        additional_key='treatment_type')
        ezetimibe_if_mono = treatment_type_if_mono == project_globals.TREATMENT.ezetimibe
        fibrates_if_mono = treatment_type_if_mono == project_globals.TREATMENT.fibrates
        low_statin_if_mono = treatment_type_if_mono == parameters.STATIN_LOW
        high_statin_if_mono = treatment_type_if_mono == parameters.STATIN_HIGH

        # noinspection PyTypeChecker
        fdc_if_multi = (self.randomness.get_draw(pop_data.index, additional_key='fdc')
                        < self.p_therapy_type[parameters.FDC])

        p_low = (self.p_treatment_type[parameters.STATIN_LOW]
                 / (self.p_treatment_type[parameters.STATIN_LOW] + self.p_treatment_type[parameters.STATIN_HIGH]))
        # noinspection PyTypeChecker
        low_potency_statin_if_not_fdc = (self.randomness.get_draw(pop_data.index, additional_key='low_potency_statin')
                                         < p_low)
        low_dose_if_low_statin = (self.randomness.get_draw(pop_data.index, additional_key='low_dose')
                                  < parameters.LOW_DOSE_THRESHOLD)

        low_dose_if_fdc = (self.randomness.get_draw(pop_data.index, additional_key='low_dose_fdc')
                           < parameters.PROBABILITY_FDC_LOW_DOSE)

        # Mono doses
        on_mono = treated & mono_if_treated

        pop_update.loc[on_mono & fibrates_if_mono] = project_globals.TREATMENT.fibrates
        pop_update.loc[on_mono & ezetimibe_if_mono] = project_globals.TREATMENT.ezetimibe

        pop_update.loc[(on_mono & low_statin_if_mono
                        & low_dose_if_low_statin)] = project_globals.TREATMENT.low_statin_low_dose
        pop_update.loc[(on_mono & low_statin_if_mono
                        & ~low_dose_if_low_statin)] = project_globals.TREATMENT.low_statin_high_dose
        pop_update.loc[on_mono & high_statin_if_mono] = project_globals.TREATMENT.high_statin_low_dose

        # Multi pill doses
        on_multi = treated & ~mono_if_treated & ~fdc_if_multi

        pop_update.loc[(on_multi & low_potency_statin_if_not_fdc
                        & low_dose_if_low_statin)] = project_globals.TREATMENT.low_statin_low_dose_multi
        pop_update.loc[(on_multi & low_potency_statin_if_not_fdc
                        & ~low_dose_if_low_statin)] = project_globals.TREATMENT.low_statin_high_dose_multi
        pop_update.loc[on_multi & ~low_potency_statin_if_not_fdc] = project_globals.TREATMENT.high_statin_low_dose

        # FDC
        fdc = treated & ~mono_if_treated & fdc_if_multi

        pop_update.loc[fdc & low_dose_if_fdc] = project_globals.TREATMENT.low_statin_low_dose_fdc
        pop_update.loc[fdc & ~low_dose_if_fdc] = project_globals.TREATMENT.low_statin_high_dose_fdc

        self.population_view.update(pop_update)

    def get_treatment_probability(self, ldlc: pd.Series) -> pd.Series:
        """Gets probability of treatment given ldlc level."""
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
    def load_treatment_and_target_p(builder: 'Builder') -> (float, float):
        """Load the probability someone is at target given they are treated."""
        location = builder.configuration.input_data.location
        draw = builder.configuration.input_data.input_draw_number
        p_rx_given_bad_ldlc = parameters.sample_probability_rx_given_high_ldl_c(location, draw)
        p_at_target_given_treated = parameters.sample_probability_target_given_rx(location, draw)
        return p_rx_given_bad_ldlc, p_at_target_given_treated

    @staticmethod
    def load_therapy_type_p(builder: 'Builder') -> Dict[str, float]:
        """Load probability of monotherapy or fdc given treated."""
        location = builder.configuration.input_data.location
        draw = builder.configuration.input_data.input_draw_number
        p_therapy_type = {therapy_type: parameters.sample_therapy_type(location, draw, therapy_type)
                          for therapy_type in [parameters.MONOTHERAPY, parameters.FDC]}
        return p_therapy_type

    @staticmethod
    def load_treatment_type_p(builder: 'Builder') -> Dict[str, float]:
        """Load probabilities of particular treatments given treated."""
        location = builder.configuration.input_data.location
        draw = builder.configuration.input_data.input_draw_number
        p_treatment_type = {treatment_type: parameters.sample_raw_drug_prescription(location, draw, treatment_type)
                            for treatment_type in [project_globals.TREATMENT.ezetimibe,
                                                   project_globals.TREATMENT.fibrates,
                                                   parameters.STATIN_HIGH, parameters.STATIN_LOW]}
        p_treatment_type = dict(zip([k for k in p_treatment_type.keys()],
                                    parameters.get_adjusted_probabilities(*p_treatment_type.values())))
        return p_treatment_type


class LDLCTreatmentAdherence:
    """Manages adherence of patients to their ldlc meds."""

    @property
    def name(self):
        """the name of this component."""
        return 'ldlc_treatment_adherence'

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: 'Builder'):
        """Loads adherence data and sets up propensity."""
        self.p_adherent = self.load_adherence_p(builder)

        self.columns_required = [project_globals.IHD_MODEL_NAME,
                                 project_globals.ISCHEMIC_STROKE_MODEL_NAME,
                                 project_globals.TREATMENT.name]

        self.population_view = builder.population.get_view(self.columns_required
                                                           + [f'{self.name}_propensity'])
        self.randomness = builder.randomness.get_stream(self.name)
        builder.population.initializes_simulants(self.on_initialize_simulants,
                                                 creates_columns=[f'{self.name}_propensity'],
                                                 requires_streams=[self.name])

        builder.value.register_value_producer(self.name, source=self.is_adherent,
                                              requires_columns=self.columns_required + [f'{self.name}_propensity'])

    def on_initialize_simulants(self, pop_data: 'SimulantData'):
        """Samples adherence propensity."""
        self.population_view.update(pd.Series(self.randomness.get_draw(pop_data.index),
                                              index=pop_data.index,
                                              name=f'{self.name}_propensity'))

    def is_adherent(self, index: pd.Index):
        """Pipeline source for boolean value indicating treatment adherence."""
        propensity = self.population_view.subview([f'{self.name}_propensity']).get(index)
        return self.determine_adherent(propensity[f'{self.name}_propensity'])

    def determine_adherent(self, propensity: pd.Series):
        """Determines whether someone is adherent based on propensity."""
        # Wrote as separate function cause I thought I needed for
        # initialization too. Leave for now in case I'm dumb.  - J.C.
        p_adherent = pd.Series(0, index=propensity.index)

        pop_status = self.population_view.subview(self.columns_required).get(propensity.index)
        ihd = pop_status[project_globals.IHD_MODEL_NAME] != project_globals.IHD_SUSCEPTIBLE_STATE_NAME
        stroke = (pop_status[project_globals.ISCHEMIC_STROKE_MODEL_NAME]
                  != project_globals.ISCHEMIC_STROKE_SUSCEPTIBLE_STATE_NAME)
        had_cve = ihd | stroke

        treated = pop_status[project_globals.TREATMENT.name] != 'none'
        multi_pill = pop_status[project_globals.TREATMENT.name].isin([
            project_globals.TREATMENT.low_statin_low_dose_multi,
            project_globals.TREATMENT.low_statin_high_dose_multi,
            project_globals.TREATMENT.high_statin_low_dose_multi,
            project_globals.TREATMENT.high_statin_high_dose_multi
        ])

        p_adherent.loc[treated & ~had_cve & ~multi_pill] = self.p_adherent[parameters.SINGLE_NO_CVE]
        p_adherent.loc[treated & ~had_cve & multi_pill] = self.p_adherent[parameters.MULTI_NO_CVE]
        p_adherent.loc[treated & had_cve & ~multi_pill] = self.p_adherent[parameters.SINGLE_CVE]
        p_adherent.loc[treated & had_cve & multi_pill] = self.p_adherent[parameters.MULTI_CVE]
        return propensity < p_adherent

    @staticmethod
    def load_adherence_p(builder):
        """Load probability of adherence given treatment and cve status."""
        location = builder.configuration.input_data.location
        draw = builder.configuration.input_data.input_draw_number
        p_adherent = {group: parameters.sample_adherence(location, draw, *group)
                      for group in [parameters.SINGLE_NO_CVE, parameters.MULTI_NO_CVE,
                                    parameters.SINGLE_CVE, parameters.MULTI_CVE]}
        return p_adherent


class LDLCTreatmentEffect:
    """Manages the impact of ldlc meds on cholesterol."""

    @property
    def name(self):
        """The name of this component."""
        return 'ldlc_treatment_effect'

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: 'Builder'):
        """Loads treatment effect and registers exposure modifier."""
        self.treatment_effect = self.load_treatment_effect(builder)

        self.is_adherent = builder.value.get_value('ldlc_treatment_adherence')
        self.columns_required = [project_globals.TREATMENT.name]

        # This pipeline is not required.  It's a convenience for reporting later.
        self.proportion_reduction = builder.value.register_value_producer(self.name, self.compute_proportion_reduction,
                                                                          requires_columns=self.columns_required,
                                                                          requires_values=['ldlc_treatment_adherence'])

        builder.value.register_value_modifier('high_ldl_cholesterol.exposure',
                                              self.adjust_exposure,
                                              requires_values=[self.name])
        self.population_view = builder.population.get_view(self.columns_required
                                                           + ['initial_treatment_proportion_reduction'])
        builder.population.initializes_simulants(self.on_initialize_simulants,
                                                 creates_columns=['initial_treatment_proportion_reduction'],
                                                 requires_values=[self.name])

    def on_initialize_simulants(self, pop_data: 'SimulantData'):
        """Creates initial treatment proportion reduction column."""
        self.population_view.update(pd.Series(self.proportion_reduction(pop_data.index),
                                              index=pop_data.index,
                                              name='initial_treatment_proportion_reduction'))

    def adjust_exposure(self, index: pd.Index, exposure: pd.Series) -> pd.Series:
        """Changes ldlc exposure based on treatment."""
        initial_effect = self.population_view.subview(['initial_treatment_proportion_reduction']).get(index)
        initial_effect = initial_effect['initial_treatment_proportion_reduction']  # coerce df to series
        # noinspection PyTypeChecker
        return (exposure / (1 - initial_effect)) * (1 - self.proportion_reduction(index))

    def compute_proportion_reduction(self, index: pd.Index) -> pd.Series:
        """Determines how much current treatment reduces ldlc level."""
        pop_status = self.population_view.subview([project_globals.TREATMENT.name]).get(index)
        adherence = self.is_adherent(index).astype(int)
        effect_size = pop_status[project_globals.TREATMENT.name].map(self.treatment_effect)
        return effect_size * adherence

    @staticmethod
    def load_treatment_effect(builder: 'Builder') -> Dict[str, float]:
        location = builder.configuration.input_data.location
        draw = builder.configuration.input_data.input_draw_number
        treatment_effect = {project_globals.TREATMENT.none: 0}
        for treatment in [project_globals.TREATMENT.lifestyle, project_globals.TREATMENT.fibrates,
                          project_globals.TREATMENT.ezetimibe, project_globals.TREATMENT.low_statin_low_dose,
                          project_globals.TREATMENT.low_statin_high_dose,
                          project_globals.TREATMENT.high_statin_low_dose,
                          project_globals.TREATMENT.high_statin_high_dose]:
            treatment_effect[treatment] = parameters.sample_ldlc_reduction(location, draw, treatment)

        ezetimibe_effect = treatment_effect[project_globals.TREATMENT.ezetimibe]
        for treatment in project_globals.TREATMENT:
            if 'multi' in treatment or 'fdc' in treatment:
                existing_key = treatment.split('_multi')[0].split('_fdc')[0]
                treatment_effect[treatment] = 1 - (1 - treatment_effect[existing_key]) * (1 - ezetimibe_effect)

        return treatment_effect


class AdverseEffects:
    """Manages adverse events related to ldlc treatment."""

    rate_mean = 0.032833
    rate_sd = 0.020103206

    @property
    def name(self):
        """The name of this component."""
        return 'ldlc_adverse_effects'

    def setup(self, builder: 'Builder'):
        """Builds the event rate and effect on adherence."""
        self.event_rate = builder.lookup.build_table(self.load_adverse_event_rate(builder))
        self.randomness = builder.randomness.get_stream(self.name)
        columns_created = [f'had_adverse_event']
        self.population_view = builder.population.get_view(columns_created
                                                           + ['ldlc_treatment_adherence_propensity']
                                                           + [project_globals.TREATMENT.name])
        builder.population.initializes_simulants(self.on_initialize_simulants,
                                                 creates_columns=columns_created)

    def on_initialize_simulants(self, pop_data: 'SimulantData'):
        self.population_view.update(pd.Series(False, index=pop_data.index, name='had_adverse_event'))

    def on_time_step(self, event: 'Event'):
        pop = self.population_view.get(event.index, query='alive == "alive"')
        on_treatment = pop[project_globals.TREATMENT.name] != project_globals.TREATMENT.none
        # Lookup returns a series for scalar tables.  Yuck inconsistent.
        # noinspection PyTypeChecker
        had_adverse_event = self.randomness.filter_for_rate(on_treatment, self.event_rate(on_treatment))
        pop.loc[had_adverse_event, 'ldlc_treatment_adherence_propensity'] = 0
        pop.loc[had_adverse_event, 'had_adverse_event'] = True
        self.population_view.update(pop)

    @staticmethod
    def load_adverse_event_rate(builder: 'Builder'):
        location = builder.configuration.input_data.location
        draw = builder.configuration.input_data.input_draw_number
        seed = get_hash(f'ldlc_adverse_event_rate_location_{location}_draw_{draw}')
        return sample_truncnorm_distribution(seed, AdverseEffects.rate_mean, AdverseEffects.rate_sd)
