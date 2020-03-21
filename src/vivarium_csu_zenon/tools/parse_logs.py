"""Application code for parsing results from output logs."""
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple, Union

from loguru import logger
import pandas as pd
import tqdm


def get_results(log_path: Path) -> Dict[Tuple[int, int], Dict[str, Union[str, float]]]:
    """Retrieve results from a log file.

    Parameters
    ----------
    log_path
        Path to the log file.

    Returns
    -------
        A dict whose keys are a (draw, seed) tuple and whose value is a dict
        of column-value pairs.

    """
    output = {}
    for spec, result in read_and_filter(log_path):
        draw, seed = parse_job_spec(spec)
        result_dict = parse_result(result)
        # For consistency with psimulate
        # TODO: add scenario specification if necessary.
        result_dict['input_draw_number'] = draw
        result_dict['random_seed'] = seed
        output[(draw, seed)] = result_dict

    return output


def read_and_filter(log_path: Path) -> Iterable[Tuple[str, str]]:
    """Reads in the data from the log and filters to the messages we need.

    Parameters
    ----------
    log_path
        Path to the log file.

    Returns
    -------
        An iterable of job_spec, result message pairs for parsing.

    """
    with log_path.open() as f:
        data = f.readlines()

    data = [json.loads(line) for line in data]
    job_spec_filter = 'vivarium_cluster_tools.psimulate.distributed_worker:worker:114'
    result_filter = 'vivarium.framework.engine:report'
    job_specs = [d['text'] for d in data if job_spec_filter in d['text']]
    results = [d['text'] for d in data if result_filter in d['text']]
    return zip(job_specs, results)


def parse_job_spec(job_spec: str) -> (int, int):
    """Parses a job spec message into a draw and random seed."""
    draw, seed = [int(param) for param in job_spec.split('Starting job: (')[1].split(', ')[:2]]
    return draw, seed


def parse_result(result: str) -> Dict[str, Union[str, float]]:
    """Parses a result message into a dict of column-value pairs."""
    raw_result_dict = json.loads(result.split(' - ')[1].replace("'", '"'))
    result_dict = {}
    for k, v in raw_result_dict.items():
        try:
            result_dict[k] = float(v)
        except ValueError:
            result_dict[k] = v
    return result_dict


def build_results_from_logs(log_dir: str, results_path: str):
    """Parses worker logs and writes them to an hdf file.

    Parameters
    ----------
    log_dir
        The path to the worker log directory.
    results_path
        The path to the output hdf file we wish to write.

    """
    log_dir, results_path = Path(log_dir), Path(results_path)
    index = []
    results = []
    logger.info(f'Parsing logs')
    for log_path in tqdm.tqdm(list(log_dir.iterdir())):
        if log_path.suffix == '.log':

            log_output = get_results(log_path)
            for idx, output in log_output.items():
                index.append(idx)
                results.append(output)
    logger.info('Collating results')
    index = pd.MultiIndex.from_tuples(index, names=['input_draw', 'random_seed'])
    output = pd.DataFrame(results, index=index)
    logger.info(f'Writing output to {str(results_path)}')
    output.to_hdf(results_path, key='data')
    logger.info('***DONE***')
