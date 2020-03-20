import json
import re

import ast
import pandas as pd

from loguru import logger

from vivarium_csu_zenon import paths, globals as project_globals

version_name = 'v3.1_fpg_ikf_model'
version_dir_path = paths.RESULTS_ROOT / version_name

DRAW_SEED_CODE_LOCATION = 'vivarium_cluster_tools.psimulate.distributed_worker:worker:148'
RESULTS_CODE_LOCATION = 'vivarium.framework.engine:report:160'

DRAW_REGEX = re.compile(r'input_draw_number:\\n +override: ([0-9]+)')
RANDOM_SEED_REGEX = re.compile(r'random_seed:\\n +override: ([0-9]+)')

location_runtimes = {
    'brazil': '2020_03_19_19_40_00',
    'china': '2020_03_19_19_41_55',
    'france': '2020_03_19_19_42_45',
    'italy': '2020_03_19_19_42_56',
    'russian_federation': '2020_03_19_19_43_15',
    'spain': '2020_03_19_19_43_29',
}

for location, runtime in location_runtimes.items():
    logger.info(f'Starting {location}: {runtime}')
    output_dir = version_dir_path / location / runtime
    log_dir = output_dir / 'logs' / f'{runtime}_run' / 'worker_logs'

    results_list = []

    for filename in log_dir.iterdir():
        logger.info(f'Processing {filename}')
        with open(filename) as log:
            for line in log:
                if DRAW_SEED_CODE_LOCATION in line:
                    input_draw_number = DRAW_REGEX.findall(line)[0]
                    random_seed = RANDOM_SEED_REGEX.findall(line)[0]

                if RESULTS_CODE_LOCATION in line:
                    log_message = json.loads(line)['text']
                    _, dict_string = log_message.split(f'{RESULTS_CODE_LOCATION} - ')
                    data_dict = {col_name: [value] for col_name, value in ast.literal_eval(dict_string).items()}
                    data_dict['input_draw_number'] = input_draw_number
                    data_dict['random_seed'] = random_seed

                    draw_seed = pd.DataFrame(data_dict).set_index(['input_draw_number', 'random_seed'])
                    results_list.append(draw_seed)

    output = pd.concat(results_list)
    output.to_hdf(output_dir / 'output.hdf')

logger.info("DONE!")
