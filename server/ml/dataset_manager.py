import csv
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

try:
    from .feature_extraction import (
        DEFAULT_WINDOW_SECONDS,
        FEATURE_COLUMNS,
        TARGET_COLUMN,
        extract_feature_vector,
        extract_features_from_window,
        normalize_events,
        split_into_windows,
    )
except ImportError:
    from feature_extraction import (  # type: ignore
        DEFAULT_WINDOW_SECONDS,
        FEATURE_COLUMNS,
        TARGET_COLUMN,
        extract_feature_vector,
        extract_features_from_window,
        normalize_events,
        split_into_windows,
    )

BASE_DIR = os.path.dirname(__file__)
SERVER_DIR = os.path.dirname(BASE_DIR)
GAZETALK_DIR = os.path.dirname(SERVER_DIR)
PROJECT_DIR = os.path.dirname(GAZETALK_DIR)

JSON_DIR = os.path.join(SERVER_DIR, "json_data")
REALTIME_DATASET_PATH = os.path.join(BASE_DIR, "fatigue_realtime_dataset.csv")
TRAINING_WINDOWS_PATH = os.path.join(BASE_DIR, "fatigue_training_windows.csv")

HISTORICAL_RAW_DIR = os.path.join(PROJECT_DIR, "fatigue-llm", "data", "raw")
HISTORICAL_LABELS_PATH = os.path.join(
    PROJECT_DIR,
    "fatigue-llm",
    "data",
    "processed",
    "fatigue_llm_training_data.csv",
)

WINDOW_SECONDS = DEFAULT_WINDOW_SECONDS
REALTIME_DATASET_FIELDS = FEATURE_COLUMNS + [TARGET_COLUMN]


def _scale_score_to_slider(score: float, lower_bound: float, upper_bound: float) -> float:
    if upper_bound <= lower_bound:
        return 5.0
    normalized = (float(score) - lower_bound) / (upper_bound - lower_bound)
    return 1.0 + max(0.0, min(normalized, 1.0)) * 9.0


def ensure_json_dir():
    os.makedirs(JSON_DIR, exist_ok=True)


def ensure_realtime_dataset_file():
    if os.path.exists(REALTIME_DATASET_PATH):
        return

    with open(REALTIME_DATASET_PATH, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=REALTIME_DATASET_FIELDS)
        writer.writeheader()


def save_session_json(session_data: Dict[str, Any]) -> str:
    ensure_json_dir()
    filename = session_data.get("session_id") or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    filename = f"{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filepath = os.path.join(JSON_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as file_handle:
        json.dump(session_data, file_handle, indent=2)
    return filepath


def extract_features_from_events(events: List[Dict[str, Any]], session_time: float = 0) -> Optional[Dict[str, Any]]:
    normalized = normalize_events(events)
    if not normalized:
        return None
    return extract_feature_vector(normalized, session_time=session_time)


def append_session_to_dataset(session_data: Dict[str, Any]) -> int:
    ensure_realtime_dataset_file()

    events = session_data.get("writing_test", [])
    fatigue_label = session_data.get("fatigue")
    normalized = normalize_events(events)
    if not normalized:
        return 0

    windows = split_into_windows(normalized, window_seconds=WINDOW_SECONDS)
    if not windows:
        return 0

    rows_added = 0
    with open(REALTIME_DATASET_PATH, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=REALTIME_DATASET_FIELDS)
        elapsed = 0
        for window_index, window in enumerate(windows):
            row = extract_features_from_window(window, fatigue_label=fatigue_label, session_time=elapsed)
            if row:
                writer.writerow(row)
                rows_added += 1
            elapsed = (window_index + 1) * WINDOW_SECONDS

    return rows_added


def _load_historical_labels() -> Dict[tuple, float]:
    if not os.path.exists(HISTORICAL_LABELS_PATH):
        return {}

    labels_df = pd.read_csv(HISTORICAL_LABELS_PATH)
    numeric_scores = pd.to_numeric(labels_df["fatigue_score"], errors="coerce").dropna()
    lower_bound = float(numeric_scores.min()) if not numeric_scores.empty else 0.0
    upper_bound = float(numeric_scores.max()) if not numeric_scores.empty else 1.0
    labels = {}

    for row in labels_df.itertuples(index=False):
        fatigue_value = getattr(row, "fatigue_score", None)
        if pd.isna(fatigue_value):
            continue

        file_name = getattr(row, "file_name", None)
        local_trial_id = getattr(row, "local_trial_id", None)
        if file_name is None or pd.isna(local_trial_id):
            continue

        key = (str(file_name), int(local_trial_id))
        labels[key] = _scale_score_to_slider(float(fatigue_value), lower_bound, upper_bound)

    return labels


def _iter_historical_trial_rows():
    if not os.path.isdir(HISTORICAL_RAW_DIR):
        return

    labels = _load_historical_labels()

    for root, _, files in os.walk(HISTORICAL_RAW_DIR):
        for file_name in files:
            if not file_name.endswith(".json"):
                continue

            path = os.path.join(root, file_name)
            try:
                with open(path, "r", encoding="utf-8") as file_handle:
                    payload = json.load(file_handle)
            except Exception:
                continue

            writing_test = payload.get("writing_test")
            if not isinstance(writing_test, list):
                continue

            for trial_index, trial_events in enumerate(writing_test):
                normalized = normalize_events(trial_events)
                if not normalized:
                    continue

                fatigue_value = labels.get((file_name, trial_index))
                if fatigue_value is None:
                    continue

                windows = split_into_windows(normalized, window_seconds=WINDOW_SECONDS)
                if not windows:
                    continue

                for window_index, window in enumerate(windows):
                    row = extract_features_from_window(
                        window,
                        fatigue_label=fatigue_value,
                        session_time=window_index * WINDOW_SECONDS,
                    )
                    if not row:
                        continue

                    row["file_name"] = file_name
                    row["trial_id"] = trial_index
                    row["window_index"] = window_index
                    row["source"] = "historical"
                    yield row


def _load_realtime_rows() -> List[Dict[str, Any]]:
    if not os.path.exists(REALTIME_DATASET_PATH):
        return []

    realtime_df = pd.read_csv(REALTIME_DATASET_PATH)
    if realtime_df.empty:
        return []

    rows = []
    for row in realtime_df.to_dict(orient="records"):
        clean_row = {column: row.get(column, 0) for column in REALTIME_DATASET_FIELDS}
        clean_row["source"] = "realtime"
        rows.append(clean_row)
    return rows


def build_training_dataframe(include_historical=True, include_realtime=True) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    if include_historical:
        rows.extend(list(_iter_historical_trial_rows()))

    if include_realtime:
        rows.extend(_load_realtime_rows())

    if not rows:
        return pd.DataFrame(columns=FEATURE_COLUMNS + [TARGET_COLUMN, "source"])

    dataset = pd.DataFrame(rows)

    for column in FEATURE_COLUMNS:
        if column not in dataset.columns:
            dataset[column] = 0.0

    dataset[TARGET_COLUMN] = pd.to_numeric(dataset[TARGET_COLUMN], errors="coerce")
    dataset = dataset.dropna(subset=[TARGET_COLUMN]).copy()
    dataset = dataset[dataset[TARGET_COLUMN].between(1, 10)].copy()

    return dataset


def save_training_snapshot(dataset: pd.DataFrame) -> Optional[str]:
    if dataset.empty:
        return None

    dataset.to_csv(TRAINING_WINDOWS_PATH, index=False)
    return TRAINING_WINDOWS_PATH


def get_dataset_summary() -> Dict[str, Any]:
    realtime_rows = 0
    if os.path.exists(REALTIME_DATASET_PATH):
        with open(REALTIME_DATASET_PATH, "r", encoding="utf-8") as file_handle:
            realtime_rows = max(sum(1 for _ in file_handle) - 1, 0)

    historical_labels = _load_historical_labels()

    return {
        "realtime_rows": realtime_rows,
        "realtime_path": REALTIME_DATASET_PATH,
        "historical_trials_with_labels": len(historical_labels),
        "historical_raw_dir": HISTORICAL_RAW_DIR,
    }
