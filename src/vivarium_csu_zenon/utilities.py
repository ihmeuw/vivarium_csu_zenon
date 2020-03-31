import numpy as np
import pandas as pd
from scipy.stats import truncnorm

from vivarium_public_health.risks.data_transformations import pivot_categorical

from vivarium_csu_zenon import globals as project_globals


class TruncnormParams:
    def __init__(self, mean, sd, lower=0, upper=1):
        self.a = (lower - mean) / sd if sd else mean
        self.b = (upper - mean) / sd if sd else mean
        self.loc = mean
        self.scale = sd


def sample_truncnorm_distribution(seed: int, mean: float, sd: float, lower: float = 0.0, upper: float = 1.0) -> float:
    """Gets a single random draw from a truncated normal distribution.
    Parameters
    ----------
    seed
        Seed for the random number generator.
    mean
        mean of truncnorm distribution
    sd
        standard deviation of truncnorm distribution
    lower
        lower bound of truncnorm distribution
    upper
        upper bound of truncnorm distribution
    Returns
    -------
        The random variate from the truncated normal distribution.
    """
    # Handle degenerate distribution
    if not sd:
        return mean

    np.random.seed(seed)
    params = TruncnormParams(mean, sd, lower, upper)
    return truncnorm.rvs(params.a, params.b, params.loc, params.scale)


def sanitize_location(location: str):
    """Cleans up location formatting for writing and reading from file names.

    Parameters
    ----------
    location
        The unsanitized location name.

    Returns
    -------
        The sanitized location name (lower-case with white-space and
        special characters removed.

    """
    # FIXME: Should make this a reversible transformation.
    return location.replace(" ", "_").replace("'", "_").lower()


def read_data_by_draw(artifact_path: str, key : str, draw: int) -> pd.DataFrame:
    """Reads data from the artifact on a per-draw basis. This
    is necessary for Low Birthweight Short Gestation (LBWSG) data.

    Parameters
    ----------
    artifact
        The artifact to read from.
    key
        The entity key associated with the data to read.
    draw
        The data to retrieve.

    """
    key = key.replace(".", "/")
    with pd.HDFStore(artifact_path, mode='r') as store:
        index = store.get(f'{key}/index')
        draw = store.get(f'{key}/draw_{draw}')
    draw = draw.rename("value")
    data = pd.concat([index, draw], axis=1)
    data = data.drop(columns='location')
    data = pivot_categorical(data)
    data[project_globals.LBWSG_MISSING_CATEGORY.CAT] = project_globals.LBWSG_MISSING_CATEGORY.EXPOSURE
    return data
