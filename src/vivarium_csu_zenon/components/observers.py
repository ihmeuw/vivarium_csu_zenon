from collections import Counter
import typing
from typing import Dict, Iterable, List, Tuple, Union

import pandas as pd
from vivarium_public_health.metrics import (MortalityObserver as MortalityObserver_,
                                            DisabilityObserver as DisabilityObserver_)
from vivarium_public_health.metrics.utilities import (get_output_template, get_group_counts,
                                                      QueryString, to_years, get_person_time,
                                                      get_deaths, get_years_of_life_lost,
                                                      get_years_lived_with_disability, get_age_bins,
                                                      get_age_sex_filter_and_iterables)

from vivarium_csu_zenon import globals as project_globals, paths
from vivarium_csu_zenon.components.disease import ChronicKidneyDisease

if typing.TYPE_CHECKING:
    from vivarium.framework.engine import Builder
    from vivarium.framework.event import Event
    from vivarium.framework.population import SimulantData


class ResultsStratifier:
    """Centralized component for handling results stratification.

    This should be used as a sub-component for observers.  The observers
    can then ask this component for population subgroups and labels during
    results production and have this component manage adjustments to the
    final column labels for the subgroups.

    """

    def __init__(self, observer_name: str):
        self.name = f'{observer_name}_results_stratifier'

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: 'Builder'):
        """Perform this component's setup."""
        # The only thing you should request here are resources necessary for
        # results stratification.
        self.sbp = builder.value.get_value('high_systolic_blood_pressure.exposure')
        self.ldlc = builder.value.get_value('high_ldl_cholesterol.exposure')
        columns_required = [project_globals.IHD_MODEL_NAME,
                            project_globals.ISCHEMIC_STROKE_MODEL_NAME,
                            project_globals.DIABETES_MELLITUS_MODEL_NAME]
        self.population_view = builder.population.get_view(columns_required)
        self.risk_groups = None
        builder.population.initializes_simulants(self.on_initialize_simulants,
                                                 requires_columns=columns_required,
                                                 requires_values=['high_systolic_blood_pressure.exposure',
                                                                  'high_ldl_cholesterol.exposure'])

    # noinspection PyAttributeOutsideInit
    def on_initialize_simulants(self, pop_data: 'SimulantData'):
        risk_groups = pd.Series('', index=pop_data.index)
        pop = self.population_view.get(pop_data.index)
        sbp = self.sbp(pop_data.index)
        ldlc = self.ldlc(pop_data.index)

        post_acs = (
                (pop[project_globals.IHD_MODEL_NAME] != project_globals.IHD_SUSCEPTIBLE_STATE_NAME)
                | (pop[project_globals.ISCHEMIC_STROKE_MODEL_NAME]
                   != project_globals.ISCHEMIC_STROKE_SUSCEPTIBLE_STATE_NAME)
        )
        high_sbp = sbp > 140
        high_ldlc = ldlc > 5
        high_fpg = (pop[project_globals.DIABETES_MELLITUS_MODEL_NAME]
                    != project_globals.DIABETES_MELLITUS_SUSCEPTIBLE_STATE_NAME)

        risk_groups.loc[high_sbp & high_ldlc & high_fpg & post_acs] = project_globals.RISK_GROUPS.cat1
        risk_groups.loc[high_sbp & high_ldlc & high_fpg & ~post_acs] = project_globals.RISK_GROUPS.cat2
        risk_groups.loc[high_sbp & high_ldlc & ~high_fpg & post_acs] = project_globals.RISK_GROUPS.cat3
        risk_groups.loc[high_sbp & high_ldlc & ~high_fpg & ~post_acs] = project_globals.RISK_GROUPS.cat4
        risk_groups.loc[high_sbp & ~high_ldlc & high_fpg & post_acs] = project_globals.RISK_GROUPS.cat5
        risk_groups.loc[high_sbp & ~high_ldlc & high_fpg & ~post_acs] = project_globals.RISK_GROUPS.cat6
        risk_groups.loc[high_sbp & ~high_ldlc & ~high_fpg & post_acs] = project_globals.RISK_GROUPS.cat7
        risk_groups.loc[high_sbp & ~high_ldlc & ~high_fpg & ~post_acs] = project_globals.RISK_GROUPS.cat8
        risk_groups.loc[~high_sbp & high_ldlc & high_fpg & post_acs] = project_globals.RISK_GROUPS.cat9
        risk_groups.loc[~high_sbp & high_ldlc & high_fpg & ~post_acs] = project_globals.RISK_GROUPS.cat10
        risk_groups.loc[~high_sbp & high_ldlc & ~high_fpg & post_acs] = project_globals.RISK_GROUPS.cat11
        risk_groups.loc[~high_sbp & high_ldlc & ~high_fpg & ~post_acs] = project_globals.RISK_GROUPS.cat12
        risk_groups.loc[~high_sbp & ~high_ldlc & high_fpg & post_acs] = project_globals.RISK_GROUPS.cat13
        risk_groups.loc[~high_sbp & ~high_ldlc & high_fpg & ~post_acs] = project_globals.RISK_GROUPS.cat14
        risk_groups.loc[~high_sbp & ~high_ldlc & ~high_fpg & post_acs] = project_globals.RISK_GROUPS.cat15
        risk_groups.loc[~high_sbp & ~high_ldlc & ~high_fpg & ~post_acs] = project_globals.RISK_GROUPS.cat16
        self.risk_groups = risk_groups

    def group(self, population: pd.DataFrame) -> Iterable[Tuple[Tuple[str, ...], pd.DataFrame]]:
        """Takes the full population and yields stratified subgroups.

        Parameters
        ----------
        population
            The population to stratify.

        Yields
        ------
            A tuple of stratification labels and the population subgroup
            corresponding to those labels.

        """
        stratification_group = self.risk_groups.loc[population.index]
        for risk_cat in project_globals.RISK_GROUPS:
            if population.empty:
                pop_in_group = population
            else:
                pop_in_group = population.loc[stratification_group == risk_cat]
            yield (risk_cat,), pop_in_group

    @staticmethod
    def update_labels(measure_data: Dict[str, float], labels: Tuple[str, ...]) -> Dict[str, float]:
        """Updates a dict of measure data with stratification labels.

        Parameters
        ----------
        measure_data
            The measure data with unstratified column names.
        labels
            The stratification labels. Yielded along with the population
            subgroup the measure data was produced from by a call to
            :obj:`ResultsStratifier.group`.

        Returns
        -------
            The measure data with column names updated with the stratification
            labels.

        """
        stratification_label = labels[0]
        measure_data = {f'{k}_{stratification_label}': v for k, v in measure_data.items()}
        return measure_data


class MortalityObserver(MortalityObserver_):

    def __init__(self):
        super().__init__()
        self.stratifier = ResultsStratifier(self.name)

    @property
    def sub_components(self) -> List[ResultsStratifier]:
        return [self.stratifier]

    def setup(self, builder: 'Builder'):
        super().setup(builder)
        if builder.components.get_components_by_type(ChronicKidneyDisease):
            self.causes += [project_globals.CKD_MODEL_NAME]

    def metrics(self, index: pd.Index, metrics: Dict[str, float]) -> Dict[str, float]:
        pop = self.population_view.get(index)
        pop.loc[pop.exit_time.isnull(), 'exit_time'] = self.clock()

        measure_getters = (
            (get_person_time, ()),
            (get_deaths, (self.causes,)),
            (get_years_of_life_lost, (self.life_expectancy, self.causes)),
        )

        for labels, pop_in_group in self.stratifier.group(pop):
            base_args = (pop_in_group, self.config.to_dict(), self.start_time, self.clock(), self.age_bins)

            for measure_getter, extra_args in measure_getters:
                measure_data = measure_getter(*base_args, *extra_args)
                measure_data = self.stratifier.update_labels(measure_data, labels)
                metrics.update(measure_data)

        the_living = pop[(pop.alive == 'alive') & pop.tracked]
        the_dead = pop[pop.alive == 'dead']
        metrics[project_globals.TOTAL_YLLS_COLUMN] = self.life_expectancy(the_dead.index).sum()
        metrics['total_population_living'] = len(the_living)
        metrics['total_population_dead'] = len(the_dead)

        return metrics


class DisabilityObserver(DisabilityObserver_):

    def __init__(self):
        super().__init__()
        self.stratifier = ResultsStratifier(self.name)

    @property
    def sub_components(self) -> List[ResultsStratifier]:
        return [self.stratifier]

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: 'Builder'):
        super().setup(builder)
        if builder.components.get_components_by_type(ChronicKidneyDisease):
            self.causes += [project_globals.CKD_MODEL_NAME]
            self.disability_weight_pipelines = {cause: builder.value.get_value(f'{cause}.disability_weight')
                                                for cause in self.causes}

    def on_time_step_prepare(self, event: 'Event'):
        pop = self.population_view.get(event.index, query='tracked == True and alive == "alive"')
        self.update_metrics(pop)

        pop.loc[:, project_globals.TOTAL_YLDS_COLUMN] += self.disability_weight(pop.index)
        self.population_view.update(pop)

    def update_metrics(self, pop: pd.DataFrame):
        for labels, pop_in_group in self.stratifier.group(pop):
            ylds_this_step = get_years_lived_with_disability(pop_in_group, self.config.to_dict(),
                                                             self.clock().year, self.step_size(),
                                                             self.age_bins, self.disability_weight_pipelines,
                                                             self.causes)
            ylds_this_step = self.stratifier.update_labels(ylds_this_step, labels)
            self.years_lived_with_disability.update(ylds_this_step)


class DiseaseObserver:
    """Observes transition counts and person time for a cause."""
    configuration_defaults = {
        'metrics': {
            'disease_observer': {
                'by_age': False,
                'by_year': False,
                'by_sex': False,
            }
        }
    }

    def __init__(self, disease: str):
        self.disease = disease
        self.configuration_defaults = {
            'metrics': {f'{disease}_observer': DiseaseObserver.configuration_defaults['metrics']['disease_observer']}
        }
        self.stratifier = ResultsStratifier(self.name)

    @property
    def name(self) -> str:
        return f'disease_observer.{self.disease}'

    @property
    def sub_components(self) -> List[ResultsStratifier]:
        return [self.stratifier]

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: 'Builder'):
        self.config = builder.configuration['metrics'][f'{self.disease}_observer'].to_dict()
        self.clock = builder.time.clock()
        self.age_bins = get_age_bins(builder)
        self.counts = Counter()
        self.person_time = Counter()

        self.states = project_globals.DISEASE_MODEL_MAP[self.disease]['states']
        self.transitions = project_globals.DISEASE_MODEL_MAP[self.disease]['transitions']

        self.previous_state_column = f'previous_{self.disease}'
        builder.population.initializes_simulants(self.on_initialize_simulants,
                                                 creates_columns=[self.previous_state_column])

        columns_required = ['alive', f'{self.disease}', self.previous_state_column]
        if self.config['by_age']:
            columns_required += ['age']
        if self.config['by_sex']:
            columns_required += ['sex']
        self.population_view = builder.population.get_view(columns_required)

        builder.value.register_value_modifier('metrics', self.metrics)
        # FIXME: The state table is modified before the clock advances.
        # In order to get an accurate representation of person time we need to look at
        # the state table before anything happens.
        builder.event.register_listener('time_step__prepare', self.on_time_step_prepare)
        builder.event.register_listener('collect_metrics', self.on_collect_metrics)

    def on_initialize_simulants(self, pop_data: 'SimulantData'):
        self.population_view.update(pd.Series('', index=pop_data.index, name=self.previous_state_column))

    def on_time_step_prepare(self, event: 'Event'):
        pop = self.population_view.get(event.index)
        # Ignoring the edge case where the step spans a new year.
        # Accrue all counts and time to the current year.
        for labels, pop_in_group in self.stratifier.group(pop):
            for state in self.states:
                # noinspection PyTypeChecker
                state_person_time_this_step = get_state_person_time(pop_in_group, self.config, self.disease, state,
                                                                    self.clock().year, event.step_size, self.age_bins)
                state_person_time_this_step = self.stratifier.update_labels(state_person_time_this_step, labels)
                self.person_time.update(state_person_time_this_step)

        # This enables tracking of transitions between states
        prior_state_pop = self.population_view.get(event.index)
        prior_state_pop[self.previous_state_column] = prior_state_pop[self.disease]
        self.population_view.update(prior_state_pop)

    def on_collect_metrics(self, event: 'Event'):
        pop = self.population_view.get(event.index)
        for labels, pop_in_group in self.stratifier.group(pop):
            for transition in self.transitions:
                # noinspection PyTypeChecker
                transition_counts_this_step = get_transition_count(pop_in_group, self.config, self.disease, transition,
                                                                   event.time, self.age_bins)
                transition_counts_this_step = self.stratifier.update_labels(transition_counts_this_step, labels)
                self.counts.update(transition_counts_this_step)

    def metrics(self, index: pd.Index, metrics: Dict[str, float]):
        metrics.update(self.counts)
        metrics.update(self.person_time)
        return metrics

    def __repr__(self) -> str:
        return f"DiseaseObserver({self.disease})"


class MiscellaneousObserver:
    """Observes person time weighted by observed metrics."""
    configuration_defaults = {
        'metrics': {
            'miscellaneous_observer': {
                'by_age': False,
                'by_year': False,
                'by_sex': False,
            }
        }
    }

    def __init__(self):
        self.stratifier = ResultsStratifier(self.name)

    @property
    def name(self) -> str:
        return 'miscellaneous_observer'

    @property
    def sub_components(self) -> List[ResultsStratifier]:
        return [self.stratifier]

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: 'Builder'):
        self.config = builder.configuration.metrics.miscellaneous_observer.to_dict()
        self.age_bins = get_age_bins(builder)
        columns_required = [project_globals.TREATMENT.name, 'age', 'sex', 'alive',
                            'initial_treatment_proportion_reduction']
        self.population_view = builder.population.get_view(columns_required)

        self.fpg = builder.value.get_value(f'{project_globals.FPG.name}.exposure')
        self.ldlc = builder.value.get_value(f'{project_globals.LDL_C.name}.exposure')
        self.sbp = builder.value.get_value(f'{project_globals.SBP.name}.exposure')
        self.is_adherent = builder.value.get_value('ldlc_treatment_adherence')
        self.cvd_risk_score = builder.value.get_value('cvd_risk_score')

        self.results = Counter()

        builder.value.register_value_modifier('metrics', self.metrics)
        builder.event.register_listener('collect_metrics', self.on_collect_metrics)

    def on_collect_metrics(self, event: 'Event'):
        pop = self.population_view.get(event.index, query='alive == "alive"')
        initial_proportion_reduction = pop['initial_treatment_proportion_reduction']

        fpg = self.fpg(pop.index)
        sbp = self.sbp(pop.index)
        ldlc = self.ldlc(pop.index)
        cvd_score = self.cvd_risk_score(pop.index)
        measure_map = list(zip(['fpg_person_time', 'sbp_person_time', 'ldlc_person_time', 'cv_risk_score_person_time'],
                               [fpg, sbp, ldlc, cvd_score]))

        adherent = self.is_adherent(pop.index).astype(int)

        raw_ldlc = ldlc / (1 - initial_proportion_reduction)
        at_target = (ldlc / raw_ldlc <= 0.5).astype(int)

        # noinspection PyTypeChecker
        step_size = to_years(event.step_size)

        age_sex_filter, (ages, sexes) = get_age_sex_filter_and_iterables(self.config, self.age_bins)
        base_key = get_output_template(**self.config).substitute(year=event.time.year)
        base_filter = QueryString(f'alive == "alive"') + age_sex_filter
        person_time = {}
        for labels, pop_in_group in self.stratifier.group(pop):
            for group, age_group in ages:
                start, end = age_group.age_start, age_group.age_end
                for sex in sexes:
                    filter_kwargs = {'age_start': start, 'age_end': end, 'sex': sex, 'age_group': group}
                    group_key = base_key.substitute(**filter_kwargs)
                    group_filter = base_filter.format(**filter_kwargs)

                    sub_pop = (pop_in_group.query(group_filter)
                               if group_filter and not pop_in_group.empty else pop_in_group)

                    for measure, attribute in measure_map:
                        person_time[group_key.substitute(measure=measure)] = sum(attribute.loc[sub_pop.index]
                                                                                 * step_size)

                    adherent_pt = sum(adherent.loc[sub_pop.index] * step_size)
                    person_time[group_key.substitute(measure='adherent_person_time')] = adherent_pt

                    at_target_pt = sum(at_target.loc[sub_pop.index] * step_size)
                    person_time[group_key.substitute(measure='at_target_person_time')] = at_target_pt

                    treatments = {group_key.substitute(measure=f'{treatment}_person_time'): 0.
                                  for treatment in project_globals.TREATMENT}

                    treatments.update((sub_pop[project_globals.TREATMENT.name]
                                       .map(lambda x: group_key.substitute(measure=f'{x}_person_time'))
                                       .value_counts() * step_size)
                                      .to_dict())
                    person_time.update(treatments)

            self.results.update(self.stratifier.update_labels(person_time, labels))

    def metrics(self, index: pd.Index, metrics: Dict[str, float]):
        metrics.update(self.results)
        return metrics


class SampleHistoryObserver:

    configuration_defaults = {
        'metrics': {
            'sample_history_observer': {
                'sample_size': 1000,
                'path': f'{paths.RESULTS_ROOT}/sample_history.hdf'
            }
        }
    }

    @property
    def name(self):
        return "sample_history_observer"

    def __init__(self):
        self.history_snapshots = []
        self.sample_index = None

    def setup(self, builder: 'Builder'):
        self.clock = builder.time.clock()
        self.sample_history_parameters = builder.configuration.metrics.sample_history_observer
        self.randomness = builder.randomness.get_stream("sample_history")

        # sets the sample index
        builder.population.initializes_simulants(self.on_initialize_simulants, requires_streams=['sample_history'])

        columns_required = [
            'alive', 'age', 'sex', 'entrance_time', 'exit_time',
            project_globals.TREATMENT.name,
            'initial_treatment_proportion_reduction',
            'cause_of_death',
            'acute_myocardial_infarction_event_time',
            'post_myocardial_infarction_event_time',
            'acute_ischemic_stroke_event_time',
            'post_ischemic_stroke_event_time',
            project_globals.DOCTOR_VISIT,
            project_globals.FOLLOW_UP_DATE,
        ]
        self.population_view = builder.population.get_view(columns_required)

        # keys will become column names in the output
        self.pipelines = {
            'ldl': builder.value.get_value('high_ldl_cholesterol.exposure'),
            'fpg': builder.value.get_value('high_fasting_plasma_glucose_continuous.exposure'),
            'sbp': builder.value.get_value('high_systolic_blood_pressure.exposure'),
            'ikf': builder.value.get_value('impaired_kidney_function.exposure'),
            'healthcare_utilization_rate': builder.value.get_value('utilization_rate'),
        }

        # record on time_step__prepare to make sure all pipelines + state table
        # columns are reflective of same time
        builder.event.register_listener('time_step__prepare', self.on_time_step__prepare)
        builder.event.register_listener('simulation_end', self.on_simulation_end)

    def on_initialize_simulants(self, pop_data):
        sample_size = self.sample_history_parameters.sample_size
        if sample_size is None or sample_size > len(pop_data.index):
            sample_size = len(pop_data.index)
        draw = self.randomness.get_draw(pop_data.index)
        priority_index = [i for d, i in sorted(zip(draw, pop_data.index), key=lambda x:x[0])]
        self.sample_index = pd.Index(priority_index[:sample_size])

    def on_time_step__prepare(self, event):
        pop = self.population_view.get(self.sample_index)

        pipeline_results = []
        for name, pipeline in self.pipelines.items():
            values = pipeline(self.sample_index)
            values = values.rename(name)
            pipeline_results.append(values)

        record = pd.concat(pipeline_results + [pop], axis=1)
        record['time'] = self.clock()

        # Get untreated LDL
        record['untreated_ldl'] = (
                self.pipelines['ldl'].source(self.sample_index) / (1 - pop['initial_treatment_proportion_reduction'])
        )

        # Get doctor visits this time step
        record[project_globals.BACKGROUND_VISIT] = pop[project_globals.DOCTOR_VISIT] == project_globals.BACKGROUND_VISIT
        record[project_globals.FOLLOW_UP_VISIT] = pop[project_globals.DOCTOR_VISIT] == project_globals.FOLLOW_UP_VISIT
        del record[project_globals.DOCTOR_VISIT]

        record.index.rename("simulant", inplace=True)
        record.set_index('time', append=True, inplace=True)

        self.history_snapshots.append(record)

    def on_simulation_end(self, event):
        self.on_time_step__prepare(event)  # record once more since we were recording at the beginning of each time step
        sample_history = pd.concat(self.history_snapshots, axis=0)
        sample_history.to_hdf(self.sample_history_parameters.path, key='trajectories')


def get_state_person_time(pop: pd.DataFrame, config: Dict[str, bool],
                          disease: str, state: str, current_year: Union[str, int],
                          step_size: pd.Timedelta, age_bins: pd.DataFrame) -> Dict[str, float]:
    """Custom person time getter that handles state column name assumptions"""
    base_key = get_output_template(**config).substitute(measure=f'{state}_person_time',
                                                        year=current_year)
    base_filter = QueryString(f'alive == "alive" and {disease} == "{state}"')
    person_time = get_group_counts(pop, base_filter, base_key, config, age_bins,
                                   aggregate=lambda x: len(x) * to_years(step_size))
    return person_time


def get_transition_count(pop: pd.DataFrame, config: Dict[str, bool],
                         disease: str, transition: project_globals.TransitionString,
                         event_time: pd.Timestamp, age_bins: pd.DataFrame) -> Dict[str, float]:
    """Counts transitions that occurred this step."""
    event_this_step = ((pop[f'previous_{disease}'] == transition.from_state)
                       & (pop[disease] == transition.to_state))
    transitioned_pop = pop.loc[event_this_step]
    base_key = get_output_template(**config).substitute(measure=f'{transition}_event_count',
                                                        year=event_time.year)
    base_filter = QueryString('')
    transition_count = get_group_counts(transitioned_pop, base_filter, base_key, config, age_bins)
    return transition_count
