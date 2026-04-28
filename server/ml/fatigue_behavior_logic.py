from typing import Dict, Any


def score_fatigue_level(score: float) -> str:
    if score <= 3.5:
        return "low"
    if score <= 6.5:
        return "medium"
    return "high"


def analyze_behavior(features: Dict[str, Any]) -> Dict[str, Any]:
    analysis = {}
    wpm = float(features.get("wpm", 0))
    mean_iki = float(features.get("mean_iki", 0))
    pause_count = int(features.get("pause_count", 0))
    long_pause_count = int(features.get("long_pause_count", 0))
    error_rate = float(features.get("error_rate", 0))
    backspace_count = int(features.get("backspace_count", 0))
    insert_suggestion_count = int(features.get("insert_suggestion_count", 0))
    insert_letter_suggestion_count = int(features.get("insert_letter_suggestion_count", 0))
    tile_gaze_not_selected_count = int(features.get("tile_gaze_not_selected_count", 0))

    analysis["wpm"] = {
        "value": wpm,
        "impact": "lower is more fatigued",
        "note": "WPM falls when typing becomes slow under fatigue"
    }
    analysis["mean_iki"] = {
        "value": mean_iki,
        "impact": "higher is more fatigued",
        "note": "Longer keystroke intervals indicate hesitation and fatigue"
    }
    analysis["pause_count"] = {
        "value": pause_count,
        "impact": "higher is more fatigued",
        "note": "More pauses suggest disrupted flow and tiredness"
    }
    analysis["long_pause_count"] = {
        "value": long_pause_count,
        "impact": "higher is more fatigued",
        "note": "Repeated pauses above two seconds are a strong fatigue signal"
    }
    analysis["error_rate"] = {
        "value": round(error_rate, 3),
        "impact": "higher is more fatigued",
        "note": "Higher error rate and corrections are common when fatigued"
    }
    analysis["backspace_count"] = {
        "value": backspace_count,
        "impact": "higher is more fatigued",
        "note": "More backspaces usually mean reduced accuracy from fatigue"
    }
    analysis["suggestions"] = {
        "value": insert_suggestion_count + insert_letter_suggestion_count,
        "impact": "context dependent",
        "note": "Frequent suggestion use can reflect either efficient support or increasing reliance when tired"
    }
    analysis["gaze_indecision"] = {
        "value": tile_gaze_not_selected_count,
        "impact": "higher is more fatigued",
        "note": "More gazed-but-not-selected tiles suggest hesitation and reduced decisiveness"
    }
    analysis["summary"] = "Fatigue rises when speed drops, pauses lengthen, corrections grow, and gaze hesitation increases."
    return analysis
