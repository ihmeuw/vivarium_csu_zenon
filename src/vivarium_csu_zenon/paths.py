from pathlib import Path

import vivarium_csu_zenon
import vivarium_csu_zenon.globals as project_globals

ARTIFACT_ROOT = Path(f"/share/costeffectiveness/artifacts/{project_globals.PROJECT_NAME}/")
MODEL_SPEC_DIR = (Path(__file__).parent / 'model_specifications').resolve()
RESULTS_ROOT = Path(f'/share/costeffectiveness/results/{project_globals.PROJECT_NAME}/')
FPG_THRESHOLD_DIR = Path('/share/costeffectiveness/auxiliary_data/GBD_2017/03_untracked_data/fpg_diabetes_threshold')
LDL_C_THRESHOLD_DIR = Path('/share/costeffectiveness/auxiliary_data/GBD_2017/03_untracked_data/ldl_c_threshold')
JOINT_PAF_DIR = Path('/share/costeffectiveness/auxiliary_data/GBD_2017/03_untracked_data/joint_pafs')

INTERNAL_DATA_ROOT = Path(vivarium_csu_zenon.__file__).resolve().parent / 'data' / 'raw_data'

CORRELATION_DATA_PATH = INTERNAL_DATA_ROOT / 'spearman_correlations.csv'
CURRENT_RX_DATA_PATH = INTERNAL_DATA_ROOT / 'current_rx.csv'
PROB_TESTING_LDL_C_PATH = INTERNAL_DATA_ROOT / 'prob_testing_ldlc.csv'
PROB_RX_GIVEN_HIGH_LDL_C = INTERNAL_DATA_ROOT / 'prob_rx_given_high_ldlc.csv'
PROB_TARGET_GIVEN_RX = INTERNAL_DATA_ROOT / 'prob_target_given_rx.csv'
ADHERENCE_PARAMETERS = INTERNAL_DATA_ROOT / 'adherence_parameters.csv'
PROB_ADDING_DRUGS = INTERNAL_DATA_ROOT / 'prob_adding_drugs.csv'
PROB_THERAPY_TYPE = INTERNAL_DATA_ROOT / 'dist_therapy_type.csv'
LDLC_REDUCTION = INTERNAL_DATA_ROOT / 'reduction_in_ldlc.csv'

HEALTHCARE_UTILIZATION = INTERNAL_DATA_ROOT / 'outpatient_utilization.csv'
