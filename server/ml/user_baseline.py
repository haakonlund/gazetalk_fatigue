"""
User baseline management for personalized fatigue normalization.
Stores per-user baseline typing features and provides z-score normalization.
"""

import csv
import os
import statistics
from typing import Dict, Optional, Tuple

BASE_DIR = os.path.dirname(__file__)
BASELINES_PATH = os.path.join(BASE_DIR, "user_baselines.csv")

BASELINE_FIELDS = [
    "user_id",
    "mean_wpm",
    "std_wpm",
    "mean_error_rate",
    "std_error_rate",
    "mean_pause_count",
    "std_pause_count",
    "mean_pause_duration",
    "std_pause_duration",
    "mean_backspace_count",
    "std_backspace_count",
    "samples_count",
]

# Features used for baseline normalization
NORMALIZABLE_FEATURES = {
    "wpm": ("mean_wpm", "std_wpm"),
    "error_rate": ("mean_error_rate", "std_error_rate"),
    "pause_count": ("mean_pause_count", "std_pause_count"),
    "mean_pause": ("mean_pause_duration", "std_pause_duration"),
    "backspace_count": ("mean_backspace_count", "std_backspace_count"),
}


def _ensure_baselines_file():
    """Create baselines CSV if it doesn't exist."""
    if not os.path.exists(BASELINES_PATH):
        with open(BASELINES_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=BASELINE_FIELDS)
            writer.writeheader()


def get_user_baseline(user_id: str) -> Optional[Dict]:
    """Retrieve baseline for a user."""
    _ensure_baselines_file()
    try:
        with open(BASELINES_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["user_id"] == user_id:
                    return {
                        k: float(v) if v else 0.0
                        for k, v in row.items()
                        if k != "user_id"
                    }
    except Exception as e:
        print(f"Error reading baseline for {user_id}: {e}")
    return None


def save_user_baseline(user_id: str, features_list: list) -> bool:
    """Create baseline from first session features (list of feature dicts)."""
    if not features_list or len(features_list) == 0:
        return False

    try:
        # Calculate aggregates across all windows in the session
        wpm_values = [f.get("wpm", 0) for f in features_list]
        error_rates = [f.get("error_rate", 0) for f in features_list]
        pause_counts = [f.get("pause_count", 0) for f in features_list]
        pause_durations = [f.get("mean_pause", 0) for f in features_list]
        backspace_counts = [f.get("backspace_count", 0) for f in features_list]

        baseline = {
            "user_id": user_id,
            "mean_wpm": statistics.mean(wpm_values) if wpm_values else 0,
            "std_wpm": statistics.stdev(wpm_values) if len(wpm_values) > 1 else 0,
            "mean_error_rate": statistics.mean(error_rates) if error_rates else 0,
            "std_error_rate": statistics.stdev(error_rates)
            if len(error_rates) > 1
            else 0,
            "mean_pause_count": statistics.mean(pause_counts) if pause_counts else 0,
            "std_pause_count": statistics.stdev(pause_counts)
            if len(pause_counts) > 1
            else 0,
            "mean_pause_duration": statistics.mean(pause_durations)
            if pause_durations
            else 0,
            "std_pause_duration": statistics.stdev(pause_durations)
            if len(pause_durations) > 1
            else 0,
            "mean_backspace_count": statistics.mean(backspace_counts)
            if backspace_counts
            else 0,
            "std_backspace_count": statistics.stdev(backspace_counts)
            if len(backspace_counts) > 1
            else 0,
            "samples_count": len(features_list),
        }

        # Check if user already exists and update or append
        _ensure_baselines_file()
        rows = []
        user_exists = False

        try:
            with open(BASELINES_PATH, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row["user_id"] == user_id:
                        user_exists = True
                        rows.append(baseline)
                    else:
                        rows.append(row)
        except FileNotFoundError:
            rows.append(baseline)

        if not user_exists:
            rows.append(baseline)

        with open(BASELINES_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=BASELINE_FIELDS)
            writer.writeheader()
            writer.writerows(rows)

        return True
    except Exception as e:
        print(f"Error saving baseline for {user_id}: {e}")
        return False


def normalize_fatigue_prediction(
    fatigue_score: float, features: Dict, user_id: str
) -> Tuple[float, Dict]:
    """
    Normalize fatigue prediction using user baseline.
    Returns (normalized_score, normalization_info).
    """
    baseline = get_user_baseline(user_id)

    if not baseline:
        # No baseline yet, return original score
        return fatigue_score, {"normalized": False, "reason": "no_baseline"}

    # Calculate z-scores for key features
    z_scores = {}
    deviations = {}

    for feature_name, (mean_field, std_field) in NORMALIZABLE_FEATURES.items():
        current_value = features.get(feature_name, 0)
        mean_value = baseline.get(mean_field, 0)
        std_value = baseline.get(std_field, 0.1)  # Avoid division by zero

        if std_value == 0:
            std_value = 0.1

        z_score = (mean_value - current_value) / std_value
        z_scores[feature_name] = z_score
        deviations[feature_name] = {
            "current": current_value,
            "baseline_mean": mean_value,
            "z_score": z_score,
        }

    # Compute average z-score (higher z = more fatigue relative to baseline)
    avg_z_score = statistics.mean(z_scores.values()) if z_scores else 0

    # Normalize fatigue: combine raw prediction with z-score
    # If user is 1σ below baseline (positive z), add fatigue weight
    # If user is 1σ above baseline (negative z), subtract fatigue weight
    normalized_score = fatigue_score + (avg_z_score * 1.5)  # 1.5x weight for z-score
    normalized_score = max(1.0, min(10.0, normalized_score))  # Clamp to 1-10

    return normalized_score, {
        "normalized": True,
        "raw_score": fatigue_score,
        "normalized_score": normalized_score,
        "avg_z_score": avg_z_score,
        "deviations": deviations,
    }


def reset_user_baseline(user_id: str) -> bool:
    """Delete baseline for a user (e.g., for re-calibration)."""
    _ensure_baselines_file()
    try:
        rows = []
        with open(BASELINES_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["user_id"] != user_id:
                    rows.append(row)

        with open(BASELINES_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=BASELINE_FIELDS)
            writer.writeheader()
            writer.writerows(rows)

        return True
    except Exception as e:
        print(f"Error resetting baseline for {user_id}: {e}")
        return False
