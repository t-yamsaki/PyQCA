"""Machine-learning-enhanced QCA foundations and predictor backends."""

from qca.mlqca._optional import (
    MLQCAExtra,
    mlqca_dependencies_available,
    optional_dependency_status,
    require_mlqca_dependencies,
)
from qca.mlqca.backend import PredictorBackend, PredictorFitResult
from qca.mlqca.calibration import (
    CalibrationDirection,
    CrispCalibrationSet,
    calibrate_crisp_conditions,
    propose_crisp_calibrations,
)
from qca.mlqca.config import (
    MLQCAConfig,
    MLQCAEngine,
    MLQCAMode,
    MLQCAValidation,
)
from qca.mlqca.csqca import (
    MLQCACSQCAResult,
    build_necessity_table,
    fit_csqca_from_calibration,
    fit_csqca_from_predictor,
)
from qca.mlqca.cutoffs import extract_split_cutoffs, xgboost_split_nodes
from qca.mlqca.fuzzy import (
    FuzzyCalibrationSet,
    MLQCAAnchorSensitivityResult,
    MLQCAFSQCAResult,
    calibrate_fuzzy_conditions,
    fit_fsqca_from_calibration,
    fit_fsqca_from_predictor,
    proposal_anchor_grid,
    propose_fuzzy_anchors,
    run_anchor_sensitivity,
)
from qca.mlqca.importance import extract_xgboost_importance
from qca.mlqca.results import MLQCAResult
from qca.mlqca.schema import (
    MLConditionDataType,
    MLConditionDirection,
    MLConditionSpec,
)
from qca.mlqca.search import (
    SearchErrorMode,
    generate_condition_combinations,
    pareto_frontier,
    search_csqca_models,
)
from qca.mlqca.validation import ValidatedMLQCAInput, validate_mlqca_input
from qca.mlqca.xgboost import XGBoostBackend, fit_xgboost_predictor

__all__ = [
    "MLConditionDataType",
    "MLConditionDirection",
    "MLConditionSpec",
    "MLQCACSQCAResult",
    "MLQCAConfig",
    "MLQCAEngine",
    "MLQCAExtra",
    "MLQCAMode",
    "MLQCAResult",
    "MLQCAValidation",
    "PredictorBackend",
    "PredictorFitResult",
    "SearchErrorMode",
    "CalibrationDirection",
    "CrispCalibrationSet",
    "FuzzyCalibrationSet",
    "MLQCAAnchorSensitivityResult",
    "MLQCAFSQCAResult",
    "ValidatedMLQCAInput",
    "XGBoostBackend",
    "build_necessity_table",
    "calibrate_crisp_conditions",
    "calibrate_fuzzy_conditions",
    "extract_split_cutoffs",
    "extract_xgboost_importance",
    "fit_xgboost_predictor",
    "generate_condition_combinations",
    "fit_csqca_from_calibration",
    "fit_csqca_from_predictor",
    "fit_fsqca_from_calibration",
    "fit_fsqca_from_predictor",
    "mlqca_dependencies_available",
    "optional_dependency_status",
    "pareto_frontier",
    "proposal_anchor_grid",
    "propose_crisp_calibrations",
    "propose_fuzzy_anchors",
    "require_mlqca_dependencies",
    "search_csqca_models",
    "run_anchor_sensitivity",
    "validate_mlqca_input",
    "xgboost_split_nodes",
]
