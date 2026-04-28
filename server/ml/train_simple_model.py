#!/usr/bin/env python3
"""Train the fatigue model from historical data files and live labeled sessions."""

import os
import math

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from . import dataset_manager
    from .feature_extraction import FEATURE_COLUMNS, TARGET_COLUMN
except ImportError:
    import dataset_manager  # type: ignore
    from feature_extraction import FEATURE_COLUMNS, TARGET_COLUMN  # type: ignore

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "fatigue_model.pkl")
EXCLUDED_MODEL_FEATURES = {"session_time"}
MODEL_FEATURE_COLUMNS = [
    column for column in FEATURE_COLUMNS if column not in EXCLUDED_MODEL_FEATURES
]
FATIGUE_BAND_LABELS = ["low", "medium", "high"]


def _compute_correlations(y_true, y_pred):
    truth = pd.Series(y_true, dtype=float)
    pred = pd.Series(y_pred, dtype=float)
    pearson = truth.corr(pred, method="pearson")
    spearman = truth.corr(pred, method="spearman")
    return {
        "pearson": None if pd.isna(pearson) else float(pearson),
        "spearman": None if pd.isna(spearman) else float(spearman),
    }


def _band_scores(values):
    series = pd.Series(values, dtype=float)
    return pd.cut(
        series,
        bins=[0.0, 3.5, 6.5, 10.0],
        labels=FATIGUE_BAND_LABELS,
        include_lowest=True,
    ).astype(str)


def _compute_band_metrics(y_true, y_pred):
    true_bands = _band_scores(y_true)
    pred_bands = _band_scores(y_pred)
    matrix = confusion_matrix(true_bands, pred_bands, labels=FATIGUE_BAND_LABELS)
    confusion = {
        actual: {
            predicted: int(matrix[row_index][col_index])
            for col_index, predicted in enumerate(FATIGUE_BAND_LABELS)
        }
        for row_index, actual in enumerate(FATIGUE_BAND_LABELS)
    }
    return {
        "accuracy": float(accuracy_score(true_bands, pred_bands)),
        "macro_f1": float(f1_score(true_bands, pred_bands, labels=FATIGUE_BAND_LABELS, average="macro", zero_division=0)),
        "confusion_matrix": confusion,
    }


def _rmse(y_true, y_pred):
    try:
        return float(mean_squared_error(y_true, y_pred, squared=False))
    except TypeError:
        return float(math.sqrt(mean_squared_error(y_true, y_pred)))


def _compute_sample_weights(X_train, y_train):
    weights = pd.Series(1.0, index=X_train.index, dtype=float)

    low_fatigue = y_train <= 3.5
    fluent = (
        (X_train["wpm"] >= X_train["wpm"].quantile(0.6))
        & (X_train["error_rate"] <= X_train["error_rate"].quantile(0.4))
        & (X_train["backspace_count"] <= X_train["backspace_count"].quantile(0.5))
    )

    weights.loc[low_fatigue] = 1.35
    weights.loc[low_fatigue & fluent] = 2.0
    weights.loc[~low_fatigue & fluent] = 0.9

    return weights


def train_model_with_metrics():
    dataset = dataset_manager.build_training_dataframe()
    if dataset.empty:
        print("No labeled fatigue windows found for training.")
        return {"success": False, "message": "No labeled fatigue windows found for training."}

    X = dataset[MODEL_FEATURE_COLUMNS].copy()
    y = dataset[TARGET_COLUMN].astype(float)
    source_counts = dataset["source"].value_counts(dropna=False).to_dict()

    print(f"Loaded {len(dataset)} labeled windows for training.")
    print(dataset["source"].value_counts(dropna=False).to_string())
    print("Training features:", ", ".join(MODEL_FEATURE_COLUMNS))

    if len(dataset) < 10:
        print("Not enough training data to build a stable realtime model.")
        return {
            "success": False,
            "message": "Not enough training data to build a stable realtime model.",
            "dataset_rows": int(len(dataset)),
            "source_counts": source_counts,
        }

    test_size = 0.2 if len(dataset) >= 25 else 0.1
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
    )
    train_weights = _compute_sample_weights(X_train, y_train)

    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
            ("scaler", StandardScaler()),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=200,
                    random_state=42,
                    max_depth=12,
                    min_samples_leaf=2,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train, regressor__sample_weight=train_weights.to_numpy())

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    train_mae = mean_absolute_error(y_train, train_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    train_rmse = _rmse(y_train, train_pred)
    test_rmse = _rmse(y_test, test_pred)
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    train_corr = _compute_correlations(y_train, train_pred)
    test_corr = _compute_correlations(y_test, test_pred)
    train_band = _compute_band_metrics(y_train, train_pred)
    test_band = _compute_band_metrics(y_test, test_pred)

    print(f"Train MAE: {train_mae:.3f}")
    print(f"Test MAE: {test_mae:.3f}")
    print(f"Train RMSE: {train_rmse:.3f}")
    print(f"Test RMSE: {test_rmse:.3f}")
    print(f"Train R²: {train_r2:.3f}")
    print(f"Test R²: {test_r2:.3f}")
    print(f"Train Pearson: {train_corr['pearson']:.3f}" if train_corr["pearson"] is not None else "Train Pearson: n/a")
    print(f"Test Pearson: {test_corr['pearson']:.3f}" if test_corr["pearson"] is not None else "Test Pearson: n/a")
    print(f"Train Spearman: {train_corr['spearman']:.3f}" if train_corr["spearman"] is not None else "Train Spearman: n/a")
    print(f"Test Spearman: {test_corr['spearman']:.3f}" if test_corr["spearman"] is not None else "Test Spearman: n/a")
    print(f"Train Band Accuracy: {train_band['accuracy']:.3f}")
    print(f"Test Band Accuracy: {test_band['accuracy']:.3f}")
    print(f"Train Band Macro-F1: {train_band['macro_f1']:.3f}")
    print(f"Test Band Macro-F1: {test_band['macro_f1']:.3f}")
    print(
        "Training sample weights:"
        f" min={train_weights.min():.2f}"
        f" max={train_weights.max():.2f}"
        f" mean={train_weights.mean():.2f}"
        f" boosted_low={(train_weights > 1.0).sum()}"
    )

    dataset_manager.save_training_snapshot(dataset)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    return {
        "success": True,
        "message": "Model trained successfully",
        "dataset_rows": int(len(dataset)),
        "source_counts": source_counts,
        "split": {
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "test_size": float(test_size),
        },
        "sample_weights": {
            "min": float(train_weights.min()),
            "max": float(train_weights.max()),
            "mean": float(train_weights.mean()),
            "boosted_low_rows": int((train_weights > 1.0).sum()),
        },
        "metrics": {
            "train_mae": float(train_mae),
            "test_mae": float(test_mae),
            "train_rmse": float(train_rmse),
            "test_rmse": float(test_rmse),
            "train_r2": float(train_r2),
            "test_r2": float(test_r2),
            "train_pearson": train_corr["pearson"],
            "test_pearson": test_corr["pearson"],
            "train_spearman": train_corr["spearman"],
            "test_spearman": test_corr["spearman"],
        },
        "band_metrics": {
            "train_accuracy": float(train_band["accuracy"]),
            "test_accuracy": float(test_band["accuracy"]),
            "train_macro_f1": float(train_band["macro_f1"]),
            "test_macro_f1": float(test_band["macro_f1"]),
            "labels": FATIGUE_BAND_LABELS,
            "train_confusion_matrix": train_band["confusion_matrix"],
            "test_confusion_matrix": test_band["confusion_matrix"],
        },
        "windowing": {
            "window_seconds": dataset_manager.WINDOW_SECONDS,
            "unit": "window",
            "includes_historical": bool(source_counts.get("historical", 0)),
            "includes_realtime": bool(source_counts.get("realtime", 0)),
        },
        "model_features": MODEL_FEATURE_COLUMNS,
        "excluded_model_features": sorted(EXCLUDED_MODEL_FEATURES),
        "model_path": MODEL_PATH,
    }


def train_model():
    result = train_model_with_metrics()
    return bool(result.get("success"))


if __name__ == "__main__":
    raise SystemExit(0 if train_model() else 1)
