import typing
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import scipy.stats
from vivarium.framework.randomness import get_hash

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
    def load_correlation_data(builder: 'Builder') -> Dict[Tuple[float, float], pd.DataFrame]:
        data = {
            (0., 125.): pd.DataFrame(data=[[1.0, 0.5, 0.5, 0.5, 0.5],
                                           [0.5, 1.0, 0.5, 0.5, 0.0],
                                           [0.5, 0.5, 1.0, 0.5, 0.5],
                                           [0.5, 0.5, 0.5, 1.0, 0.5],
                                           [0.5, 0.0, 0.5, 0.5, 1.0]],
                                     index=project_globals.CORRELATED_PROPENSITY_COLUMNS,
                                     columns=project_globals.CORRELATED_PROPENSITY_COLUMNS)
        }
        return data


def copula_sample(correlation_matrix, samples, randomness_key):
    seed = get_hash(randomness_key)
    np.random.seed(seed)
    dist = scipy.stats.multivariate_normal(cov=correlation_matrix)
    return scipy.stats.norm().cdf(dist.rvs(samples))
