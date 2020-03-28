from pathlib import Path

import vivarium_csu_zenon
import vivarium_csu_zenon.globals as project_globals

ARTIFACT_ROOT = Path(f"/share/costeffectiveness/artifacts/{project_globals.PROJECT_NAME}/")
MODEL_SPEC_DIR = (Path(__file__).parent / 'model_specifications').resolve()
RESULTS_ROOT = Path(f'/share/costeffectiveness/results/{project_globals.PROJECT_NAME}/')
FPG_THRESHOLD_DIR = Path('/share/costeffectiveness/auxiliary_data/GBD_2017/03_untracked_data/fpg_diabetes_threshold')
JOINT_PAF_DIR = Path('/share/costeffectiveness/auxiliary_data/GBD_2017/03_untracked_data/joint_pafs')

INTERNAL_DATA_ROOT = Path(vivarium_csu_zenon.__file__).resolve().parent / 'data'
CORRELATION_DATA_PATH = INTERNAL_DATA_ROOT / 'spearman_correlations.csv'
