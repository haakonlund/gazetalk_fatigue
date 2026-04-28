import statistics
from datetime import datetime

FEATURE_COLUMNS = [
    "mean_iki",
    "std_iki",
    "pause_count",
    "mean_pause",
    "long_pause_count",
    "total_events",
    "insert_count",
    "insert_suggestion_count",
    "insert_letter_suggestion_count",
    "backspace_count",
    "delete_word_count",
    "delete_sentence_count",
    "switch_view_count",
    "gaze_event_count",
    "text_area_gaze_count",
    "tile_gaze_not_selected_count",
    "error_rate",
    "wpm",
    "event_rate",
    "session_time",
]

TARGET_COLUMN = "fatigue"
DEFAULT_WINDOW_SECONDS = 20

INSERT_EVENT_TYPES = {
    "enter_letter",
    "insert_letter",
    "newline",
}
LETTER_SUGGESTION_TYPES = {
    "insert_letter_suggestion",
}
WORD_SUGGESTION_TYPES = {
    "insert_suggestion",
}
DELETE_EVENT_TYPES = {
    "delete_letter",
    "delete_letter_edit",
    "delete_word",
    "delete_sentence",
    "delete_section",
    "backspace",
}
GAZE_EVENT_TYPES = {
    "tile_gazed_and_not_selected",
    "text_area_gazed",
}


def safe_mean(values):
    return sum(values) / len(values) if values else 0.0


def safe_std(values):
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values)


def parse_timestamp(ts):
    if ts is None:
        return None

    if isinstance(ts, (int, float)):
        value = float(ts)
        return value / 1000 if value > 1e12 else value

    if not isinstance(ts, str):
        return None

    try:
        ts = ts.replace("Z", "+00:00")
        return datetime.fromisoformat(ts).timestamp()
    except Exception:
        return None


def flatten_items(items):
    flat = []
    if isinstance(items, list):
        for item in items:
            flat.extend(flatten_items(item))
    else:
        flat.append(items)
    return flat


def normalize_events(raw_events):
    normalized = []

    for event in flatten_items(raw_events):
        if not isinstance(event, dict):
            continue

        ts = parse_timestamp(event.get("timestamp"))
        if ts is None:
            continue

        normalized.append(
            {
                "ts": ts,
                "type": str(event.get("type", "")),
                "value": event.get("value", ""),
                "label": event.get("label", ""),
            }
        )

    return sorted(normalized, key=lambda event: event["ts"])


def split_into_windows(events, window_seconds=DEFAULT_WINDOW_SECONDS):
    if not events:
        return []

    windows = []
    current_window = []
    window_start = events[0]["ts"]

    for event in events:
        if event["ts"] - window_start < window_seconds:
            current_window.append(event)
            continue

        if current_window:
            windows.append(current_window)
        current_window = [event]
        window_start = event["ts"]

    if current_window:
        windows.append(current_window)

    return windows


def _count_events(events, event_types):
    return sum(1 for event in events if event["type"] in event_types)


def extract_features_from_window(events, fatigue_label=None, session_time=0):
    if len(events) < 2:
        return None

    timestamps = [event["ts"] for event in events]
    duration = timestamps[-1] - timestamps[0]
    if duration <= 0:
        return None

    intervals = []
    for index in range(1, len(timestamps)):
        interval = timestamps[index] - timestamps[index - 1]
        if interval >= 0:
            intervals.append(interval)

    if not intervals:
        return None

    insert_count = _count_events(events, INSERT_EVENT_TYPES)
    insert_suggestion_count = _count_events(events, WORD_SUGGESTION_TYPES)
    insert_letter_suggestion_count = _count_events(events, LETTER_SUGGESTION_TYPES)
    backspace_count = _count_events(events, DELETE_EVENT_TYPES)
    delete_word_count = _count_events(events, {"delete_word"})
    delete_sentence_count = _count_events(events, {"delete_sentence", "delete_section"})
    switch_view_count = _count_events(events, {"switch_view"})
    text_area_gaze_count = _count_events(events, {"text_area_gazed"})
    tile_gaze_not_selected_count = _count_events(events, {"tile_gazed_and_not_selected"})
    gaze_event_count = _count_events(events, GAZE_EVENT_TYPES)
    pauses = [interval for interval in intervals if interval > 1.0]
    long_pauses = [interval for interval in intervals if interval > 2.0]

    typed_units = (
        insert_count
        + insert_letter_suggestion_count
        + insert_suggestion_count * 5
    )
    wpm = (typed_units / 5) / (duration / 60)

    return {
        "mean_iki": safe_mean(intervals),
        "std_iki": safe_std(intervals),
        "pause_count": len(pauses),
        "mean_pause": safe_mean(pauses),
        "long_pause_count": len(long_pauses),
        "total_events": len(events),
        "insert_count": insert_count,
        "insert_suggestion_count": insert_suggestion_count,
        "insert_letter_suggestion_count": insert_letter_suggestion_count,
        "backspace_count": backspace_count,
        "delete_word_count": delete_word_count,
        "delete_sentence_count": delete_sentence_count,
        "switch_view_count": switch_view_count,
        "gaze_event_count": gaze_event_count,
        "text_area_gaze_count": text_area_gaze_count,
        "tile_gaze_not_selected_count": tile_gaze_not_selected_count,
        "error_rate": backspace_count / max(insert_count + insert_letter_suggestion_count, 1),
        "wpm": wpm,
        "event_rate": len(events) / duration,
        "session_time": float(session_time),
        TARGET_COLUMN: fatigue_label if fatigue_label is not None else "",
    }


def extract_feature_vector(events, session_time=0):
    row = extract_features_from_window(events, fatigue_label=None, session_time=session_time)
    if row is None:
        return None

    row.pop(TARGET_COLUMN, None)
    return row
