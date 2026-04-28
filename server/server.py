from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import os
from datetime import datetime
import uuid

from ml import dataset_manager, predict, user_baseline
from ml.fatigue_behavior_logic import analyze_behavior
from ml.train_simple_model import train_model_with_metrics

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

WARMUP_SENTENCE_TARGET = 15
WARMUP_SECONDS_TARGET = 7 * 60
WARMUP_EVENT_TARGET = 180
WARMUP_BASELINE_SCORE = 1.0
SESSION_TIME_EFFECT_START = 8 * 60
SESSION_TIME_EFFECT_FULL = 30 * 60
LEARNING_PHASE_HIDE_SENTENCES = 5
LEARNING_PHASE_HIDE_SECONDS = 150
READINESS_FULL_SENTENCES = 18
READINESS_FULL_SECONDS = 12 * 60
READINESS_FULL_EVENTS = 600
SESSION_BASELINES = {}

# Configuration
STORAGE_DIR = "json_data"
if not os.path.exists(STORAGE_DIR):
    os.makedirs(STORAGE_DIR)


def _parse_event_timestamp(ts):
    if ts is None:
        return None

    if isinstance(ts, (int, float)):
        value = float(ts)
        return value / 1000 if value > 1e12 else value

    if not isinstance(ts, str):
        return None

    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _estimate_session_seconds(events):
    timestamps = []
    for event in events:
        if not isinstance(event, dict):
            continue
        parsed = _parse_event_timestamp(event.get("timestamp"))
        if parsed is not None:
            timestamps.append(parsed)

    if len(timestamps) < 2:
        return 0.0

    return max(0.0, timestamps[-1] - timestamps[0])


def _extract_current_attempt(text):
    if not isinstance(text, str):
        return ""

    normalized = text.replace("\r\n", "\n").strip()
    if not normalized:
        return ""

    lines = [line.strip() for line in normalized.split("\n") if line.strip()]
    if not lines:
        return ""

    return lines[-1]


def _levenshtein_distance(left, right):
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)

    previous_row = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, start=1):
        current_row = [left_index]
        for right_index, right_char in enumerate(right, start=1):
            insert_cost = current_row[right_index - 1] + 1
            delete_cost = previous_row[right_index] + 1
            replace_cost = previous_row[right_index - 1] + (0 if left_char == right_char else 1)
            current_row.append(min(insert_cost, delete_cost, replace_cost))
        previous_row = current_row
    return previous_row[-1]


def _compute_sentence_difference(text, target_sentence):
    if not isinstance(target_sentence, str) or not target_sentence.strip():
        return None

    attempt = _extract_current_attempt(text)
    target = target_sentence.strip()
    if not attempt:
        return None

    distance = _levenshtein_distance(attempt.lower(), target.lower())
    max_length = max(len(attempt), len(target), 1)
    return {
        "attempt": attempt,
        "target": target,
        "distance": distance,
        "ratio": distance / max_length,
    }


def _analyze_written_sentences(text):
    if not isinstance(text, str):
        return None

    normalized = text.replace("\r\n", "\n")
    fragments = []
    current = []
    for char in normalized:
        current.append(char)
        if char in ".!?":
            sentence = "".join(current).strip()
            if sentence:
                fragments.append(sentence)
            current = []

    if not fragments:
        return None

    lengths = []
    very_short_count = 0
    short_count = 0
    for sentence in fragments:
        core = "".join(character for character in sentence if character.isalpha())
        length = len(core)
        lengths.append(length)
        if length <= 1:
            very_short_count += 1
        if length <= 3:
            short_count += 1

    average_length = sum(lengths) / len(lengths) if lengths else 0.0
    return {
        "sentence_count": len(fragments),
        "average_length": average_length,
        "very_short_ratio": very_short_count / len(fragments),
        "short_ratio": short_count / len(fragments),
    }


def _apply_warmup_adjustment(score, events, sentence_count):
    session_seconds = _estimate_session_seconds(events)
    event_count = len(events) if isinstance(events, list) else 0
    completed_sentences = max(0, int(sentence_count or 0))

    progress = max(
        completed_sentences / WARMUP_SENTENCE_TARGET,
        session_seconds / WARMUP_SECONDS_TARGET,
        event_count / WARMUP_EVENT_TARGET,
    )
    progress = max(0.0, min(progress, 1.0))

    if progress >= 1.0:
        return score, {
            "applied": False,
            "progress": round(progress, 3),
            "sentence_count": completed_sentences,
            "session_seconds": round(session_seconds, 1),
            "event_count": event_count,
        }

    # Keep early-session scores more conservative while people are still learning.
    adjustment_weight = 0.02 + (0.98 * progress)
    adjusted_score = WARMUP_BASELINE_SCORE + ((score - WARMUP_BASELINE_SCORE) * adjustment_weight)
    adjusted_score = max(1.0, min(10.0, adjusted_score))

    return adjusted_score, {
        "applied": True,
        "progress": round(progress, 3),
        "sentence_count": completed_sentences,
        "session_seconds": round(session_seconds, 1),
        "event_count": event_count,
        "baseline_score": WARMUP_BASELINE_SCORE,
        "original_score": round(float(score), 2),
        "adjusted_score": round(float(adjusted_score), 2),
        "reason": "early_session_learning_phase",
    }


def _capture_session_baseline(session_id, sentence_count, features):
    if not session_id:
        return None

    existing = SESSION_BASELINES.get(session_id)
    if existing:
        return existing

    if int(sentence_count or 0) > 3:
        return None

    baseline = {
        "wpm": float(features.get("wpm", 0)),
        "error_rate": float(features.get("error_rate", 0)),
        "pause_count": float(features.get("pause_count", 0)),
        "mean_pause": float(features.get("mean_pause", 0)),
        "backspace_count": float(features.get("backspace_count", 0)),
        "sentence_count": int(sentence_count or 0),
    }
    SESSION_BASELINES[session_id] = baseline
    return baseline


def _apply_session_baseline_adjustment(score, session_id, sentence_count, features):
    baseline = _capture_session_baseline(session_id, sentence_count, features)
    if not session_id or not baseline:
        return score, {
            "applied": False,
            "reason": "no_session_baseline",
        }

    adjusted_score = float(score)
    effects = {}

    baseline_wpm = max(baseline["wpm"], 1.0)
    current_wpm = float(features.get("wpm", 0))
    wpm_drop_ratio = max(0.0, (baseline_wpm - current_wpm) / baseline_wpm)
    if wpm_drop_ratio > 0.08:
        effects["wpm"] = min(0.9, wpm_drop_ratio * 1.4)
    elif current_wpm > baseline_wpm * 1.08:
        effects["wpm"] = max(-0.25, -((current_wpm - baseline_wpm) / baseline_wpm) * 0.4)

    baseline_error = baseline["error_rate"]
    current_error = float(features.get("error_rate", 0))
    error_delta = current_error - baseline_error
    if error_delta > 0.03:
        effects["error_rate"] = min(0.6, error_delta * 4.0)
    elif error_delta < -0.03:
        effects["error_rate"] = max(-0.2, error_delta * 1.5)

    baseline_pause_count = max(baseline["pause_count"], 0.5)
    current_pause_count = float(features.get("pause_count", 0))
    pause_count_delta = (current_pause_count - baseline_pause_count) / baseline_pause_count
    if pause_count_delta > 0.2:
        effects["pause_count"] = min(0.45, pause_count_delta * 0.35)

    baseline_mean_pause = max(baseline["mean_pause"], 0.4)
    current_mean_pause = float(features.get("mean_pause", 0))
    pause_duration_delta = (current_mean_pause - baseline_mean_pause) / baseline_mean_pause
    if pause_duration_delta > 0.15:
        effects["mean_pause"] = min(0.4, pause_duration_delta * 0.3)

    baseline_backspace = baseline["backspace_count"]
    current_backspace = float(features.get("backspace_count", 0))
    backspace_delta = current_backspace - baseline_backspace
    if backspace_delta > 0:
        effects["backspace_count"] = min(0.35, backspace_delta * 0.12)

    total_effect = sum(effects.values())
    adjusted_score = max(1.0, min(10.0, adjusted_score + total_effect))
    return adjusted_score, {
        "applied": bool(effects),
        "baseline_sentence_count": baseline["sentence_count"],
        "effects": {key: round(value, 3) for key, value in effects.items()},
        "adjusted_score": round(adjusted_score, 2),
    }


def _apply_context_adjustment(score, session_seconds, sentence_count, sentence_difference, sentence_quality):
    adjusted_score = float(score)
    session_time_effect = 0.0
    sentence_progress_effect = 0.0
    sentence_difference_effect = 0.0
    sentence_quality_effect = 0.0

    if session_seconds > SESSION_TIME_EFFECT_START:
        session_progress = (
            (session_seconds - SESSION_TIME_EFFECT_START)
            / max(SESSION_TIME_EFFECT_FULL - SESSION_TIME_EFFECT_START, 1)
        )
        session_progress = max(0.0, min(session_progress, 1.0))
        session_time_effect = session_progress * 0.45
        adjusted_score += session_time_effect

    completed_sentences = max(0, int(sentence_count or 0))
    if completed_sentences > 4:
        sentence_progress = min((completed_sentences - 4) / 20.0, 1.0)
        sentence_progress_effect = sentence_progress * 0.7
        adjusted_score += sentence_progress_effect

    if sentence_difference:
        difference_ratio = float(sentence_difference["ratio"])
        # Strongly accurate sentences slightly reduce fatigue score.
        if difference_ratio <= 0.08:
            sentence_difference_effect = -0.55
        elif difference_ratio <= 0.18:
            sentence_difference_effect = -0.25
        elif difference_ratio >= 0.75:
            sentence_difference_effect = 1.9
        elif difference_ratio >= 0.6:
            sentence_difference_effect = 1.4
        elif difference_ratio >= 0.45:
            sentence_difference_effect = 0.95
        elif difference_ratio >= 0.3:
            sentence_difference_effect = 0.45

        adjusted_score += sentence_difference_effect

    if sentence_quality:
        very_short_ratio = float(sentence_quality["very_short_ratio"])
        short_ratio = float(sentence_quality["short_ratio"])
        average_length = float(sentence_quality["average_length"])

        if very_short_ratio >= 0.6:
            sentence_quality_effect += 2.2
        elif very_short_ratio >= 0.3:
            sentence_quality_effect += 1.2

        if short_ratio >= 0.75:
            sentence_quality_effect += 1.1
        elif short_ratio >= 0.45:
            sentence_quality_effect += 0.55

        if average_length > 0 and average_length <= 2.5:
            sentence_quality_effect += 1.0
        elif average_length <= 4.5:
            sentence_quality_effect += 0.45

        adjusted_score += sentence_quality_effect

    adjusted_score = max(1.0, min(10.0, adjusted_score))
    return adjusted_score, {
        "applied": (
            abs(session_time_effect) > 0
            or abs(sentence_progress_effect) > 0
            or abs(sentence_difference_effect) > 0
            or abs(sentence_quality_effect) > 0
        ),
        "session_seconds": round(session_seconds, 1),
        "session_time_effect": round(session_time_effect, 3),
        "sentence_progress_effect": round(sentence_progress_effect, 3),
        "sentence_difference_ratio": round(float(sentence_difference["ratio"]), 3) if sentence_difference else None,
        "sentence_difference_effect": round(sentence_difference_effect, 3),
        "sentence_quality_effect": round(sentence_quality_effect, 3),
        "sentence_quality": sentence_quality,
        "adjusted_score": round(adjusted_score, 2),
    }


def _apply_readiness_gate(score, confidence, sentence_count, session_seconds, event_count):
    completed_sentences = max(0, int(sentence_count or 0))
    total_events = max(0, int(event_count or 0))
    session_seconds = max(0.0, float(session_seconds or 0.0))
    confidence = max(0.0, min(float(confidence or 0.0), 1.0))

    should_hide_score = (
        completed_sentences < LEARNING_PHASE_HIDE_SENTENCES
        and session_seconds < LEARNING_PHASE_HIDE_SECONDS
        and total_events < 220
    )

    sentence_progress = max(
        0.0,
        min(
            (completed_sentences - LEARNING_PHASE_HIDE_SENTENCES)
            / max(READINESS_FULL_SENTENCES - LEARNING_PHASE_HIDE_SENTENCES, 1),
            1.0,
        ),
    )
    time_progress = max(
        0.0,
        min(
            (session_seconds - LEARNING_PHASE_HIDE_SECONDS)
            / max(READINESS_FULL_SECONDS - LEARNING_PHASE_HIDE_SECONDS, 1),
            1.0,
        ),
    )
    event_progress = max(
        0.0,
        min(
            (total_events - 220) / max(READINESS_FULL_EVENTS - 220, 1),
            1.0,
        ),
    )
    data_readiness = max(sentence_progress, time_progress, event_progress)
    confidence_weight = max(0.0, min((confidence - 0.45) / 0.45, 1.0))

    dynamic_baseline = min(2.2, 1.0 + (completed_sentences * 0.08) + (data_readiness * 0.25))
    readiness_weight = (0.2 + (0.8 * data_readiness)) * (0.35 + (0.65 * confidence_weight))
    readiness_weight = max(0.0, min(readiness_weight, 1.0))
    gated_score = dynamic_baseline + ((float(score) - dynamic_baseline) * readiness_weight)
    gated_score = max(1.0, min(10.0, gated_score))

    if should_hide_score:
        message = "Calibrating fatigue from the first sentences..."
    elif data_readiness < 0.45:
        message = "Learning your typing pattern..."
    else:
        message = "Fatigue estimate is now based on your session trend."

    return gated_score, {
        "should_hide_score": should_hide_score,
        "message": message,
        "sentence_count": completed_sentences,
        "session_seconds": round(session_seconds, 1),
        "event_count": total_events,
        "confidence": round(confidence, 3),
        "data_readiness": round(data_readiness, 3),
        "confidence_weight": round(confidence_weight, 3),
        "dynamic_baseline": round(dynamic_baseline, 2),
        "readiness_weight": round(readiness_weight, 3),
        "gated_score": round(gated_score, 2),
    }

@app.route('/save-json', methods=['POST'])
def save_json():
    try:
        # Get JSON data from request
        data = request.json
        
        # Validate the data
        if not data:
            return jsonify({"message": "No data provided"}), 400
        
        # Generate a unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        filename = f"{timestamp}_{unique_id}.json"
        filepath = os.path.join(STORAGE_DIR, filename)
        
        # Save the JSON data to a file
        with open(filepath, 'w') as file:
            json.dump(data, file, indent=2)
        
        # Return success response
        return jsonify({
            "message": "JSON data saved successfully",
            "filename": filename,
            "filepath": filepath
        }), 200
    except Exception as e:
        # Return error response
        return jsonify({
            "message": f"Error saving JSON data: {str(e)}"
        }), 500
    
@app.route('/save-test-data', methods=['POST']) 
def complete_test():
    try:
        # Get JSON data from request
        data = request.json
        
        # Validate the data
        if not data:
            return jsonify({"message": "No data provided"}), 400
        
        # get name inside the json data
        name = data.get("form_data", {}).get("name", "unknown")
        
        # Generate a unique filename
        timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
        filename = f"{name}_{timestamp}.json"
        filepath = os.path.join(STORAGE_DIR, filename)
        
        # Save the JSON data to a file
        with open(filepath, 'w') as file:
            json.dump(data, file, indent=2)
        
        # Return success response
        return jsonify({
            "message": "JSON data saved successfully",
            "filename": filename,
            "filepath": filepath
        }), 200
    except Exception as e:
        return jsonify({
            "message": f"Error saving test data: {str(e)}"
        }), 500
    
@app.route('/list-json', methods=['GET'])
def list_json():
    try:
        files = os.listdir(STORAGE_DIR)
        json_files = [f for f in files if f.endswith('.json')]
        
        return jsonify({
            "files": json_files,
            "count": len(json_files)
        }), 200
    
    except Exception as e:
        return jsonify({
            "message": f"Error listing JSON files: {str(e)}"
        }), 500

@app.route('/get-json/<filename>', methods=['GET'])
def get_json(filename):
    try:
        filepath = os.path.join(STORAGE_DIR, filename)
        
        if not os.path.exists(filepath):
            return jsonify({
                "message": f"File not found: {filename}"
            }), 404
        
        with open(filepath, 'r') as file:
            data = json.load(file)
        
        return jsonify(data), 200
    
    except Exception as e:
        return jsonify({
            "message": f"Error retrieving JSON data: {str(e)}"
        }), 500

@app.route('/log', methods=['POST'])
def log():
    try:
        log_data = request.json
        if not log_data:
            return jsonify({"message": "No log data provided"}), 400

        # one file for everything (simple)
        filepath = os.path.join(STORAGE_DIR, "remote_logs.jsonl")

        log_data["_server_ts"] = datetime.now().isoformat()

        with open(filepath, "a", encoding="utf-8") as f:
          f.write(json.dumps(log_data, ensure_ascii=False) + "\n")

        return jsonify({"message": "Log received"}), 200

    except Exception as e:
        return jsonify({"message": f"Error logging data: {str(e)}"}), 500


@app.route('/fatigue/predict', methods=['POST'])
def fatigue_predict():
    try:
        data = request.json or {}
        events = data.get('events', [])
        text = data.get('text', '')
        user_id = data.get('user_id', 'unknown')
        session_id = data.get('session_id')
        sentence_count = data.get('sentence_count', 0)
        target_sentence = data.get('target_sentence', '')
        session_seconds = _estimate_session_seconds(events)

        features = dataset_manager.extract_features_from_events(events, session_time=session_seconds)
        if not features:
            return jsonify({"message": "Not enough typing data to extract features"}), 400

        prediction = predict.predict_fatigue(features)
        behavior = analyze_behavior(features)

        # Apply baseline normalization if available
        normalized_score, normalization_info = user_baseline.normalize_fatigue_prediction(
            prediction["score"], features, user_id
        )
        warmup_adjusted_score, warmup_adjustment = _apply_warmup_adjustment(
            normalized_score, events, sentence_count
        )
        sentence_difference = _compute_sentence_difference(text, target_sentence)
        sentence_quality = _analyze_written_sentences(text)
        context_adjusted_score, context_adjustment = _apply_context_adjustment(
            warmup_adjusted_score, session_seconds, sentence_count, sentence_difference, sentence_quality
        )
        session_adjusted_score, session_adjustment = _apply_session_baseline_adjustment(
            context_adjusted_score, session_id, sentence_count, features
        )
        display_score, learning_phase = _apply_readiness_gate(
            session_adjusted_score,
            prediction.get("confidence", 0),
            sentence_count,
            session_seconds,
            len(events) if isinstance(events, list) else 0,
        )

        return jsonify({
            "success": True,
            "features": features,
            "fatigue_score": prediction["score"],
            "normalized_fatigue_score": normalized_score,
            "warmup_adjusted_fatigue_score": warmup_adjusted_score,
            "context_adjusted_fatigue_score": context_adjusted_score,
            "session_adjusted_fatigue_score": session_adjusted_score,
            "display_fatigue_score": display_score,
            "raw_prediction": prediction["raw"],
            "confidence": prediction.get("confidence", 0),
            "behavior": behavior,
            "normalization": normalization_info,
            "warmup_adjustment": warmup_adjustment,
            "context_adjustment": context_adjustment,
            "session_adjustment": session_adjustment,
            "learning_phase": learning_phase,
            "sentence_difference": sentence_difference,
            "sentence_quality": sentence_quality,
        }), 200

    except FileNotFoundError as e:
        return jsonify({"message": str(e)}), 500

    except Exception as e:
        return jsonify({"message": f"Error predicting fatigue: {str(e)}"}), 500


@app.route('/fatigue/label', methods=['POST'])
def fatigue_label():
    try:
        data = request.json or {}
        user_id = data.get('user_id', 'anonymous')
        session_id = data.get('session_id') or f"session_{uuid.uuid4().hex[:8]}"
        fatigue_value = data.get('fatigue')
        events = data.get('events', [])
        text = data.get('text', '')
        layout_condition = data.get('layout_condition')
        active_layout = data.get('active_layout')
        save_session_json = data.get('save_session_json', True)

        if fatigue_value is None:
            return jsonify({"message": "Fatigue label is required"}), 400

        session_data = {
            "user_id": user_id,
            "session_id": session_id,
            "fatigue": float(fatigue_value),
            "text": text,
            "writing_test": events,
            "created_at": datetime.now().isoformat()
        }
        if layout_condition:
            session_data["layout_condition"] = layout_condition
        if active_layout:
            session_data["active_layout"] = active_layout

        file_path = dataset_manager.save_session_json(session_data) if save_session_json else None
        rows_added = dataset_manager.append_session_to_dataset(session_data)
        prediction = None
        baseline_created = False

        try:
            features = dataset_manager.extract_features_from_events(events)
            if features:
                prediction = predict.predict_fatigue(features)
        except Exception:
            prediction = None

        # Auto-save baseline from first session if it doesn't exist yet
        try:
            existing_baseline = user_baseline.get_user_baseline(user_id)
            if not existing_baseline and events:
                # Extract features for each window in the session
                windows = dataset_manager.split_into_windows(
                    dataset_manager.normalize_events(events)
                )
                features_list = []
                for window in windows:
                    window_features = dataset_manager.extract_features_from_window(
                        window, fatigue_label=None, session_time=0
                    )
                    if window_features:
                        features_list.append(window_features)
                
                if features_list:
                    baseline_created = user_baseline.save_user_baseline(user_id, features_list)
        except Exception as e:
            print(f"Error creating baseline for {user_id}: {e}")

        return jsonify({
            "success": True,
            "saved_file": file_path,
            "dataset_rows_added": rows_added,
            "prediction": prediction,
            "baseline_created": baseline_created
        }), 200

    except Exception as e:
        return jsonify({"message": f"Error saving labeled fatigue data: {str(e)}"}), 500


@app.route('/baseline/get', methods=['GET'])
def get_baseline():
    try:
        user_id = request.args.get('user_id', 'unknown')
        baseline = user_baseline.get_user_baseline(user_id)
        
        if baseline:
            return jsonify({
                "success": True,
                "user_id": user_id,
                "baseline": baseline
            }), 200
        else:
            return jsonify({
                "success": False,
                "message": f"No baseline found for user {user_id}"
            }), 404
    except Exception as e:
        return jsonify({"message": f"Error retrieving baseline: {str(e)}"}), 500


@app.route('/baseline/reset', methods=['POST'])
def reset_baseline():
    try:
        data = request.json or {}
        user_id = data.get('user_id', 'unknown')
        
        success = user_baseline.reset_user_baseline(user_id)
        
        if success:
            return jsonify({
                "success": True,
                "message": f"Baseline for user {user_id} has been reset"
            }), 200
        else:
            return jsonify({
                "success": False,
                "message": f"Failed to reset baseline for user {user_id}"
            }), 400
    except Exception as e:
        return jsonify({"message": f"Error resetting baseline: {str(e)}"}), 500

@app.route('/fatigue/train', methods=['POST'])
def fatigue_train():
    try:
        result = train_model_with_metrics()
        status = 200 if result.get("success") else 400
        return jsonify(result), status
    except Exception as e:
        return jsonify({"message": f"Error training model: {str(e)}"}), 500


if __name__ == '__main__':
    print(f"Server started. JSON files will be saved to {os.path.abspath(STORAGE_DIR)}")
    app.run(debug=True, port=5001, host="0.0.0.0") # change the ip to the same as the wifi network interface if there is problems
