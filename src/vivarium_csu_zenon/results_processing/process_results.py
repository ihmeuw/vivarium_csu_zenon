from pathlib import Path
from typing import NamedTuple, List

import pandas as pd
import yaml

from vivarium_csu_zenon import globals as project_globals


SCENARIO_COLUMN = 'scenario'
GROUPBY_COLUMNS = [
    project_globals.INPUT_DRAW_COLUMN,
    # SCENARIO_COLUMN
]
OUTPUT_COLUMN_SORT_ORDER = [
    'age_group',
    'sex',
    'year',
    'risk',
    'cause',
    'measure',
    'input_draw'
]


def make_measure_data(data):
    measure_data = MeasureData(
        population=get_population_data(data),
        person_time=get_measure_data(data, 'person_time', with_cause=False, risk_factors=True),
        # ylls=get_measure_data(data, 'ylls', risk_factors=True),
        # ylds=get_measure_data(data, 'ylds', risk_factors=True),
        # deaths=get_measure_data(data, 'deaths', risk_factors=True),
        # state_person_time=get_measure_data(data, 'state_person_time', with_cause=False, state=True),
        # transition_count=get_measure_data(data, 'transition_count', with_cause=False, transition=True),
    )
    return measure_data


class MeasureData(NamedTuple):
    population: pd.DataFrame
    person_time: pd.DataFrame
    ylls: pd.DataFrame
    ylds: pd.DataFrame
    deaths: pd.DataFrame
    state_person_time: pd.DataFrame
    transition_count: pd.DataFrame

    def dump(self, output_dir: Path):
        for key, df in self._asdict().items():
            df.to_hdf(output_dir / f'{key}.hdf', key=key)
            df.to_csv(output_dir / f'{key}.csv')


def read_data(path: Path) -> (pd.DataFrame, List[str]):
    data = pd.read_hdf(path)
    data = (data
            .drop(columns=data.columns.intersection(project_globals.THROWAWAY_COLUMNS))
            .reset_index(drop=True)
            # TODO: add back when we have scenarios
            # .rename(columns={project_globals.OUTPUT_SCENARIO_COLUMN: SCENARIO_COLUMN}))
            )
    data[project_globals.INPUT_DRAW_COLUMN] = data[project_globals.INPUT_DRAW_COLUMN].astype(int)
    data[project_globals.RANDOM_SEED_COLUMN] = data[project_globals.RANDOM_SEED_COLUMN].astype(int)
    with (path.parent / 'keyspace.yaml').open() as f:
        keyspace = yaml.full_load(f)
    return data, keyspace


def filter_out_incomplete(data, keyspace):
    output = []
    for draw in keyspace[project_globals.INPUT_DRAW_COLUMN]:
        # For each draw, gather all random seeds completed for all scenarios.
        random_seeds = set(keyspace[project_globals.RANDOM_SEED_COLUMN])
        draw_data = data.loc[data[project_globals.INPUT_DRAW_COLUMN] == draw]
        for scenario in keyspace[project_globals.OUTPUT_SCENARIO_COLUMN]:
            seeds_in_data = draw_data.loc[data[SCENARIO_COLUMN] == scenario,
                                          project_globals.RANDOM_SEED_COLUMN].unique()
            random_seeds = random_seeds.intersection(seeds_in_data)
        draw_data = draw_data.loc[draw_data[project_globals.RANDOM_SEED_COLUMN].isin(random_seeds)]
        output.append(draw_data)
    return pd.concat(output, ignore_index=True).reset_index(drop=True)


def aggregate_over_seed(data):
    non_count_columns = []
    for non_count_template in project_globals.NON_COUNT_TEMPLATES:
        non_count_columns += project_globals.RESULT_COLUMNS(non_count_template)
    count_columns = [c for c in data.columns if c not in non_count_columns + GROUPBY_COLUMNS]

    # non_count_data = data[non_count_columns + GROUPBY_COLUMNS].groupby(GROUPBY_COLUMNS).mean()
    count_data = data[count_columns + GROUPBY_COLUMNS].groupby(GROUPBY_COLUMNS).sum()
    return pd.concat([
        count_data,
        # non_count_data
    ], axis=1).reset_index()


def pivot_data(data):
    return (data
            .set_index(GROUPBY_COLUMNS)
            .stack()
            .reset_index()
            .rename(columns={f'level_{len(GROUPBY_COLUMNS)}': 'process', 0: 'value'}))


def sort_data(data):
    sort_order = [c for c in OUTPUT_COLUMN_SORT_ORDER if c in data.columns]
    other_cols = [c for c in data.columns if c not in sort_order]
    data = data[sort_order + other_cols].sort_values(sort_order)
    return data.reset_index(drop=True)


def split_processing_column(data, with_cause, state, transition, risk_factors):
    if risk_factors:
        data['process'], data['ldl_cholestrol'] = data.process.str.split('_ldl_c_').str
        data['process'], data['systolic_blood_pressure'] = data.process.str.split('_sbp_').str
    data['process'], data['age_group'] = data.process.str.split('_in_age_group_').str
    data['process'], data['sex'] = data.process.str.split('_among_').str
    data['process'], data['year'] = data.process.str.split('_in_').str
    if with_cause:
        data['measure'], data['cause'] = data.process.str.split('_due_to_').str
    elif state:
        data['state'], _ = data.process.str.split('_person_time').str
        data['measure'] = 'person_time'
    elif transition:
        data['measure'], _ = data.process.str.split('_event_count').str
    else:
        data['measure'] = data['process']
    return data.drop(columns='process')


def get_population_data(data):
    total_pop = pivot_data(data[[project_globals.TOTAL_POPULATION_COLUMN]
                                + project_globals.RESULT_COLUMNS('population')
                                + GROUPBY_COLUMNS])
    total_pop = total_pop.rename(columns={'process': 'measure'})
    return sort_data(total_pop)


def get_measure_data(data, measure, with_cause=True, state=False, transition=False, risk_factors=False):
    import pdb;
    pdb.set_trace()
    data = pivot_data(data[project_globals.RESULT_COLUMNS(measure) + GROUPBY_COLUMNS])
    data = split_processing_column(data, with_cause, state, transition, risk_factors)
    return sort_data(data)
