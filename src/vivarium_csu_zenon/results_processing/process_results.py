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
    'diabetes_state',
    'ckd_state',
    'measure',
    'input_draw'
]


def make_measure_data(data):
    measure_data = MeasureData(
        population=get_population_data(data),
        person_time=get_measure_data(data, 'person_time'),
        ylls=get_by_cause_measure_data(data, 'ylls'),
        ylds=get_by_cause_measure_data(data, 'ylds'),
        deaths=get_by_cause_measure_data(data, 'deaths'),
        state_person_time=get_state_person_time_measure_data(data),
        transition_count=get_transition_count_measure_data(data),
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


def split_processing_column(data):
    data['measure'], year_and_sex, process = data.process.str.split('_in_').str
    data['year'], data['sex'] = year_and_sex.str.split('_among_').str
    process = process.str.split('age_group_').str[1]
    data['age_group'], process = process.str.split('_diabetes_').str
    data['diabetes_state'], data['ckd_state'] = process.str.split('_ckd_').str
    data['diabetes_state'] = data['diabetes_state'].map(project_globals.DIABETES_SHORT_TO_LONG_MAP)
    data['ckd_state'] = data['ckd_state'].map(project_globals.CKD_SHORT_TO_LONG_MAP)
    return data.drop(columns='process')


def get_population_data(data):
    total_pop = pivot_data(data[[project_globals.TOTAL_POPULATION_COLUMN]
                                + project_globals.RESULT_COLUMNS('population')
                                + GROUPBY_COLUMNS])
    total_pop = total_pop.rename(columns={'process': 'measure'})
    return sort_data(total_pop)


def get_measure_data(data, measure):
    data = pivot_data(data[project_globals.RESULT_COLUMNS(measure) + GROUPBY_COLUMNS])
    data = split_processing_column(data)
    return sort_data(data)


def get_by_cause_measure_data(data, measure):
    data = get_measure_data(data, measure)
    data['measure'], data['cause'] = data.measure.str.split('_due_to_').str
    return sort_data(data)


def get_state_person_time_measure_data(data):
    data = get_measure_data(data, 'state_person_time')
    data['measure'], data['cause'] = 'state_person_time', data.measure.str.split('_person_time').str[0]
    return sort_data(data)


def get_transition_count_measure_data(data):
    # Oops, edge case.
    data = data.drop(columns=[c for c in data.columns if 'event_count' in c and '2025' in c])
    data = pivot_data(data[project_globals.RESULT_COLUMNS('transition_count') + GROUPBY_COLUMNS])
    data = split_processing_column(data)
    return sort_data(data)
