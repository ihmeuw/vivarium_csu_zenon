from collections import Counter

import pandas as pd
from vivarium_public_health.metrics import (MortalityObserver as MortalityObserver_,
                                            DisabilityObserver as DisabilityObserver_)
from vivarium_public_health.metrics.utilities import (get_output_template, get_group_counts,
                                                      QueryString, to_years, get_person_time,
                                                      get_deaths, get_years_of_life_lost,
                                                      get_years_lived_with_disability, get_age_bins)

from vivarium_csu_zenon import globals as project_globals


class MortalityObserver(MortalityObserver_):

    def setup(self, builder):
        super().setup(builder)
        columns_required = ['tracked', 'alive', 'entrance_time', 'exit_time', 'cause_of_death',
                            'years_of_life_lost', 'age']
        if self.config.by_sex:
            columns_required += ['sex']

        self.cvd_risk_category = builder.value.get_value('cvd_risk_category')
        self.population_view = builder.population.get_view(columns_required)

    def metrics(self, index, metrics):
        pop = self.population_view.get(index)
        pop.loc[pop.exit_time.isnull(), 'exit_time'] = self.clock()

        cvd_risk = self.cvd_risk_category(index)

        measure_getters = (
            (get_person_time, ()),
            (get_deaths, (project_globals.CAUSES_OF_DEATH,)),
            (get_years_of_life_lost, (self.life_expectancy, project_globals.CAUSES_OF_DEATH)),
        )

        for cvd_risk_cat in project_globals.CVD_RISK_CATEGORIES:
            pop_in_group = pop.loc[cvd_risk == cvd_risk_cat]
            base_args = (pop_in_group, self.config.to_dict(), self.start_time, self.clock(), self.age_bins)

            for measure_getter, extra_args in measure_getters:
                measure_data = measure_getter(*base_args, *extra_args)
                measure_data = {f'{k}_cvd_{cvd_risk_cat}': v for k, v in measure_data.items()}
                metrics.update(measure_data)

        the_living = pop[(pop.alive == 'alive') & pop.tracked]
        the_dead = pop[pop.alive == 'dead']
        metrics[project_globals.TOTAL_YLLS_COLUMN] = self.life_expectancy(the_dead.index).sum()
        metrics['total_population_living'] = len(the_living)
        metrics['total_population_dead'] = len(the_dead)

        return metrics


class DisabilityObserver(DisabilityObserver_):
    def setup(self, builder):
        super().setup(builder)
        columns_required = ['tracked', 'alive', 'years_lived_with_disability',
                            project_globals.DIABETES_MELLITUS.name, project_globals.CKD_MODEL_NAME]
        if self.config.by_age:
            columns_required += ['age']
        if self.config.by_sex:
            columns_required += ['sex']
        self.population_view = builder.population.get_view(columns_required)

        self.cvd_risk_category = builder.value.get_value('cvd_risk_category')
        self.disability_weight_pipelines = {cause: builder.value.get_value(f'{cause}.disability_weight')
                                            for cause in project_globals.CAUSES_OF_DISABILITY}

    def on_time_step_prepare(self, event):
        pop = self.population_view.get(event.index, query='tracked == True and alive == "alive"')
        self.update_metrics(pop)

        pop.loc[:, project_globals.TOTAL_YLDS_COLUMN] += self.disability_weight(pop.index)
        self.population_view.update(pop)

    def update_metrics(self, pop):
        cvd_risk = self.cvd_risk_category(pop.index)

        for cvd_risk_cat in project_globals.CVD_RISK_CATEGORIES:
            pop_in_group = pop.loc[cvd_risk == cvd_risk_cat]
            ylds_this_step = get_years_lived_with_disability(pop_in_group, self.config.to_dict(),
                                                             self.clock().year, self.step_size(),
                                                             self.age_bins, self.disability_weight_pipelines,
                                                             project_globals.CAUSES_OF_DISABILITY)
            ylds_this_step = {f'{k}_cvd_{cvd_risk_cat}': v for k, v in ylds_this_step.items()}
            self.years_lived_with_disability.update(ylds_this_step)


class DiseaseObserver:
    """Observes disease counts, person time, and prevalent cases for a cause.

    By default, this observer computes aggregate susceptible person time
    and counts of disease cases over the entire simulation.  It can be
    configured to bin these into age_groups, sexes, and years by setting
    the ``by_age``, ``by_sex``, and ``by_year`` flags, respectively.

    It also records prevalent cases on a particular sample date each year.
    These will also be binned based on the flags set for the observer.
    Additionally, the sample date is configurable and defaults to July 1st
    of each year.

    In the model specification, your configuration for this component should
    be specified as, e.g.:

    .. code-block:: yaml

        configuration:
            metrics:
                {YOUR_DISEASE_NAME}_observer:
                    by_age: True
                    by_year: False
                    by_sex: True
                    prevalence_sample_date:
                        month: 4
                        day: 10

    """
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

    @property
    def name(self):
        return f'disease_observer.{self.disease}'

    def setup(self, builder):
        self.config = builder.configuration['metrics'][f'{self.disease}_observer'].to_dict()
        self.clock = builder.time.clock()
        self.age_bins = get_age_bins(builder)
        self.counts = Counter()
        self.person_time = Counter()

        self.states = project_globals.DISEASE_MODEL_MAP[self.disease]['states']
        self.transitions = project_globals.DISEASE_MODEL_MAP[self.disease]['transitions']

        self.previous_state_column = f'previous_{self.disease}'
        self.cvd_risk_category = builder.value.get_value('cvd_risk_category')
        builder.population.initializes_simulants(self.on_initialize_simulants,
                                                 creates_columns=[self.previous_state_column])

        columns_required = ['alive', f'{self.disease}', self.previous_state_column]
        for state in self.states:
            columns_required.append(f'{state}_event_time')
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

    def on_initialize_simulants(self, pop_data):
        self.population_view.update(pd.Series('', index=pop_data.index, name=self.previous_state_column))

    def on_time_step_prepare(self, event):
        pop = self.population_view.get(event.index)
        # Ignoring the edge case where the step spans a new year.
        # Accrue all counts and time to the current year.
        for state in self.states:
            state_person_time_this_step = get_state_person_time(pop, self.config, self.disease, state,
                                                                self.clock().year, event.step_size, self.age_bins)
            self.person_time.update(state_person_time_this_step)

        # This enables tracking of transitions between states
        prior_state_pop = self.population_view.get(event.index)
        prior_state_pop[self.previous_state_column] = prior_state_pop[self.disease]
        self.population_view.update(prior_state_pop)

    def on_collect_metrics(self, event):
        pop = self.population_view.get(event.index)
        for transition in self.transitions:
            event_this_step = pop[self.disease] != pop[f'previous_{self.disease}']
            transitioned_pop = pop.loc[event_this_step]
            cvd_risk = self.cvd_risk_category(transitioned_pop.index)
            base_key = get_output_template(**self.config).substitute(measure=f'{transition}_event_count',
                                                                     year=event.time.year)
            base_filter = QueryString('')
            for cvd_risk_cat in project_globals.CVD_RISK_CATEGORIES:
                pop_in_group = transitioned_pop.loc[cvd_risk == cvd_risk_cat]
                transition_counts = get_group_counts(pop_in_group, base_filter, base_key, self.config, self.age_bins)
                transition_counts = {f'{k}_cvd_{cvd_risk_cat}': v for k, v in transition_counts.items()}
                self.counts.update(transition_counts)

    def metrics(self, index, metrics):
        metrics.update(self.counts)
        metrics.update(self.person_time)
        return metrics

    def __repr__(self):
        return f"DiseaseObserver({self.disease})"
        
    
def get_state_person_time(pop, config, disease, state, current_year, step_size, age_bins):
    """Custom person time getter that handles state column name assumptions"""
    base_key = get_output_template(**config).substitute(measure=f'{state}_person_time',
                                                        year=current_year)
    base_filter = QueryString(f'alive == "alive" and {disease} == "{state}"')
    person_time = get_group_counts(pop, base_filter, base_key, config, age_bins,
                                   aggregate=lambda x: len(x) * to_years(step_size))
    return person_time
