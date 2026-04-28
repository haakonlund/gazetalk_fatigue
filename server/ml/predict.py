import os
import sys

import joblib
import pandas as pd

try:
    from .feature_extraction import FEATURE_COLUMNS
except ImportError:
    from feature_extraction import FEATURE_COLUMNS  # type: ignore

BASE_DIR = os.path.dirname(__file__)
SERVER_DIR = os.path.dirname(BASE_DIR)
GAZETALK_DIR = os.path.dirname(SERVER_DIR)
PROJECT_DIR = os.path.dirname(GAZETALK_DIR)

MODEL_PATHS = [
    os.path.join(BASE_DIR, "fatigue_model.pkl"),
    os.path.join(PROJECT_DIR, "fatigue-llm", "models", "fatigue_rf_regressor.joblib"),
]

model = None


def _build_validation_frame(loaded_model):
    feature_names = getattr(loaded_model, "feature_names_in_", None)
    columns = list(feature_names) if feature_names is not None else list(FEATURE_COLUMNS)
    frame = pd.DataFrame([[0.0] * len(columns)], columns=columns)
    return frame.astype(float)


def load_model():
    global model
    if model is not None:
        return model

    for model_path in MODEL_PATHS:
        if os.path.exists(model_path):
            try:
                print(f"Loading model from: {model_path}")
                candidate = joblib.load(model_path)
                validation_frame = _build_validation_frame(candidate)
                candidate.predict(validation_frame)
                model = candidate
                return model
            except Exception as error:
                print(f"Skipping unusable model at {model_path}: {error}", file=sys.stderr)
                continue

    raise FileNotFoundError(f"Model file not found in any location: {MODEL_PATHS}")


def _coerce_feature_frame(feature_dict):
    row = {column: feature_dict.get(column, 0) for column in FEATURE_COLUMNS}
    frame = pd.DataFrame([row])

    loaded_model = load_model()
    feature_names = getattr(loaded_model, "feature_names_in_", None)
    if feature_names is not None:
        frame = frame.reindex(columns=list(feature_names), fill_value=0)

    # Force numeric dtypes so the sklearn preprocessing pipeline receives
    # the same kind of inputs it was trained on.
    frame = frame.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)

    return frame


def _estimate_confidence(loaded_model, frame, score):
    regressor = None
    if hasattr(loaded_model, "named_steps"):
        regressor = loaded_model.named_steps.get("regressor")
    elif hasattr(loaded_model, "estimators_"):
        regressor = loaded_model

    estimators = getattr(regressor, "estimators_", None)
    if not estimators:
        return 0.0

    transformed = frame
    if hasattr(loaded_model, "named_steps"):
        for step_name, step in loaded_model.named_steps.items():
            if step_name == "regressor":
                break
            transformed = step.transform(transformed)

    tree_predictions = [estimator.predict(transformed)[0] for estimator in estimators]
    spread = max(tree_predictions) - min(tree_predictions)
    confidence = max(0.0, 1.0 - min(spread / 10.0, 1.0))
    return float(round(confidence, 3))


def predict_fatigue(feature_dict):
    try:
        loaded_model = load_model()
        feature_frame = _coerce_feature_frame(feature_dict)
        prediction = float(loaded_model.predict(feature_frame)[0])
        score = max(1.0, min(10.0, round(prediction, 2)))
        confidence = _estimate_confidence(loaded_model, feature_frame, score)
        return {
            "score": score,
            "raw": prediction,
            "confidence": confidence,
        }
    except Exception as error:
        print(f"Prediction error: {error}", file=sys.stderr)
        raise
