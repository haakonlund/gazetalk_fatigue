# GazeTalk

GazeTalk is a React + Flask project for gaze-based text entry, data collection, and fatigue prediction.

### 1. Frontend
From `GazeTalk/`:

```bash
npm install
npm run dev -- --host
```

Open the URL shown by Vite, usually:

```text
http://localhost:5173
```

### 2. Backend
From `GazeTalk/server/`:

```bash
pip3 install -r requirements.txt
for 
```

The backend runs on:

```text
http://localhost:5001
```

## Typical Run Flow

Use two terminals.

Terminal 1:

```bash
cd GazeTalk
npm run dev -- --host
```

Terminal 2:

```bash
cd GazeTalk/server
pip3 install -r requirements.txt
python3 server.py
```

Then open the frontend URL in the browser.

## Useful Commands

### Run tests
From `GazeTalk/`:

```bash
npm test
```

For a specific test file:

```bash
npx jest src/tests/views/Configs.test.js
```

### Preview the large-tile layout
Append this to the app URL:

```text
?previewLargeTiles=1
```

Example:

```text
http://localhost:5173/?previewLargeTiles=1
```

## Data Collection

The Flask server stores test data and logs under:

```text
GazeTalk/server/json_data/
```

Important files you may see there:

- session JSON files
- `remote_logs.jsonl`

Model-related CSV files live in:

```text
GazeTalk/server/ml/
```

Important files there:

- `fatigue_realtime_dataset.csv`
- `fatigue_training_windows.csv`
- `fatigue_model.pkl`

## Fatigue System Overview

There are three connected stages:

### 1. Feature extraction

Raw user events are converted into normalized event objects and then grouped into fixed time windows.

Main file:

- [server/ml/feature_extraction.py](/Users/lorenzogentili/Desktop/GazeTalkTerminal/GazeTalk/server/ml/feature_extraction.py)

The feature extraction pipeline is:

```text
raw events -> normalize -> split into 20s windows -> compute features
```

Examples of extracted features:

- `wpm`
- `pause_count`
- `mean_pause`
- `backspace_count`
- `error_rate`
- `gaze_event_count`
- `tile_gaze_not_selected_count`

### 2. Offline training

Training combines:

- historical labeled data
- realtime labeled windows collected from the app

Main files:

- [server/ml/dataset_manager.py](/Users/lorenzogentili/Desktop/GazeTalkTerminal/GazeTalk/server/ml/dataset_manager.py)
- [server/ml/train_simple_model.py](/Users/lorenzogentili/Desktop/GazeTalkTerminal/GazeTalk/server/ml/train_simple_model.py)

The trained model is a `RandomForestRegressor`.

Important note:

- `session_time` is still extracted
- but it is currently excluded from model training
- so the model relies more on interaction features than on simply being later in the session

### 3. Realtime prediction

The frontend sends recent events to:

```text
POST /fatigue/predict
```

The backend extracts features using the same logic as in training and runs prediction with the saved model.

Main files:

- [server/server.py](/Users/lorenzogentili/Desktop/GazeTalkTerminal/GazeTalk/server/server.py)
- [server/ml/predict.py](/Users/lorenzogentili/Desktop/GazeTalkTerminal/GazeTalk/server/ml/predict.py)

## End-of-Test Model Evaluation

At the end of a test, the app can retrain/evaluate the fatigue model and print metrics.

This happens:

- when you press `End Test`
- when you finish all test sentences

### Where the metrics appear

You get them in:

- the browser console
- the Python server terminal

You do not currently get a dedicated metrics file.

### Metrics currently printed

- Train/Test `MAE`
- Train/Test `RMSE`
- Train/Test `R²`
- Train/Test `Pearson`
- Train/Test `Spearman`
- Train/Test band `accuracy`
- Train/Test band `macro-F1`
- test band confusion matrix for `low / medium / high`

Open browser DevTools and check the `Console` tab for lines starting with:

```text
[FatigueModel] ...
```

## Fatigue Levels in the UI

Current UI ranges:

- `low`: score `<= 3.5`
- `medium`: `3.5 < score <= 6.5`
- `high`: score `> 6.5`

Current assistance thresholds:

- larger tiles trigger from `6.5`
- break suggestion triggers from `7.0`

## Project Structure

### `src/App.jsx`

Main state and orchestration layer for the frontend.

### `src/components`

Reusable UI components such as tiles, fatigue indicator, popups, and text area handling.

### `src/layouts`

Keyboard and interface layouts. New layouts must be registered in:

- `src/layouts/LayoutPicker.jsx`

### `src/config`

Hardcoded view and tile definitions.

### `src/constants`, `src/singleton`, `src/util`

Shared constants, singleton state, and utility logic.

### `src/tests`

Frontend tests.

### `server/`

Flask backend for:

- saving collected data
- fatigue prediction
- fatigue labeling
- model training/evaluation

## Notes

- If model evaluation fails at end of test, the most common reason is missing Python ML dependencies.
- If that happens, reinstall backend dependencies with:

```bash
cd GazeTalk/server
pip3 install -r requirements.txt
```
