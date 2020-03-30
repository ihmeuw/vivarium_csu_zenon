import typing
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import scipy.stats
from vivarium.framework.randomness import get_hash
from vivarium_public_health.utilities import TargetString

from vivarium_csu_zenon import globals as project_globals

if typing.TYPE_CHECKING:
    from vivarium.framework.engine import Builder
    from vivarium.framework.population import SimulantData


class CorrelatedPropensityGenerator:

    @property
    def name(self) -> str:
        return 'correlated_propensity_generator'

    def setup(self, builder: 'Builder'):
        self.randomness_key = self.get_randomness_key(builder)
        self.randomness = builder.randomness.get_stream(self.name)
        self.correlation = self.load_correlation_data(builder)
        columns_created = project_globals.CORRELATED_PROPENSITY_COLUMNS
        self.population_view = builder.population.get_view(columns_created + ['age'])
        builder.population.initializes_simulants(self.on_initialize_simulants,
                                                 creates_columns=columns_created,
                                                 requires_columns=['age'],
                                                 requires_streams=[self.name])

        for target, rate in project_globals.RATE_TARGET_MAP.items():
            population_attributable_fraction_data = self.load_population_attributable_fraction_data(builder, rate)
            population_attributable_fraction = builder.lookup.build_table(population_attributable_fraction_data,
                                                                          key_columns=['sex'],
                                                                          parameter_columns=['age', 'year'])
            target = TargetString(target)
            builder.value.register_value_modifier(f'{target.name}.{target.measure}.paf',
                                                  modifier=population_attributable_fraction,
                                                  requires_columns=['age', 'sex'])

    def on_initialize_simulants(self, pop_data: 'SimulantData'):
        age = self.population_view.subview(['age']).get(pop_data.index).age
        propensities = pd.DataFrame(data=0,
                                    columns=project_globals.CORRELATED_PROPENSITY_COLUMNS,
                                    index=pop_data.index)
        for (age_start, age_end), correlation in self.correlation.items():
            randomness_key = self.randomness_key + str((age_start, age_end))
            in_group_idx = age[(age_start <= age) & (age < age_end)].index
            propensities.loc[in_group_idx, :] = copula_sample(correlation, len(in_group_idx), randomness_key)
        self.population_view.update(propensities)

    @staticmethod
    def get_randomness_key(builder: 'Builder'):
        location = builder.configuration.input_data.location
        draw = builder.configuration.input_data.input_draw_number
        return f'correlated_propensities_location_{location}_draw_{draw}'

    @staticmethod
    def load_correlation_data(builder: 'Builder') -> Dict[Tuple[int, int], pd.DataFrame]:
        raw_data = builder.data.load(project_globals.POPULATION.PROPENSITY_CORRELATION_DATA)
        data = {}
        for label, group in raw_data.groupby(['age_group']):
            group_matrix = group.set_index(['risk_a', 'risk_b']).value.unstack()
            group_matrix.index.name = None
            group_matrix.columns.name = None
            fpg = group_matrix.loc[project_globals.FPG_PROPENSITY_COLUMN].copy()
            fpg.loc[project_globals.FPG_PROPENSITY_COLUMN] = 0.
            group_matrix[project_globals.DIABETES_PROPENSITY_COLUMN] = fpg
            fpg.loc[project_globals.DIABETES_PROPENSITY_COLUMN] = 1.
            group_matrix.loc[project_globals.DIABETES_PROPENSITY_COLUMN] = fpg
            age_start, age_end = [int(age) for age in label.split('_to_')]
            age_start = 0 if age_start == 30 else age_start
            age_end = 125 if age_end == 79 else age_end + 1
            data_key = (age_start, age_end)
            data[data_key] = group_matrix
        return data

    @staticmethod
    def load_population_attributable_fraction_data(builder: 'Builder', rate: TargetString):
        paf_data = builder.data.load(project_globals.POPULATION.JOINT_PAF_DATA)
        correct_target = ((paf_data['affected_entity'] == rate.name) & (paf_data['affected_measure'] == rate.measure))
        paf_data = paf_data[correct_target].drop(['affected_entity', 'affected_measure'], 'columns')
        return paf_data


def copula_sample(correlation_matrix, samples, randomness_key):
    seed = get_hash(randomness_key)
    np.random.seed(seed)
    dist = scipy.stats.multivariate_normal(cov=correlation_matrix)
    return scipy.stats.norm().cdf(dist.rvs(samples))
