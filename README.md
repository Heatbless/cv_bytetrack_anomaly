# CV CCTV: YOLO Monitoring + Split Temporal VAE Anomaly Detection

This project combines two pipelines for traffic CCTV analysis:

1. YOLO-based vehicle monitoring, directional sidewalk violation counting, and tracklet feature export.
2. Temporal VAE anomaly detection with left-road and right-road specialization.

The main workflow is in the notebook `YOLO.ipynb`, with helper scripts for dataset preparation and zone calibration.

## What This Project Does

- Detects traffic objects with Ultralytics YOLO and ByteTrack.
- Uses Kalman smoothing to reduce visual jitter in tracked centroids.
- Counts directional violations on predefined sidewalk polygons.
- Exports per-track features (speed and distance statistics).
- Trains two independent Temporal VAEs:
	- Left model trained from `vae_split_samples/left`
	- Right model trained from `vae_split_samples/right`
- Produces timeline-aware anomaly scores and seekable playback windows.

## Repository Structure

- `YOLO.ipynb`: Main notebook containing YOLO monitoring and split-VAE anomaly workflow.
- `vae_anomaly_module.py`: Reusable Temporal VAE module (training, thresholding, inference, ROI crop support).
- `split_vae_samples_lr.py`: Splits normal traffic clips into left and right road datasets.
- `vae_sample_splicer_gui.py`: GUI tool to splice long videos into normal-traffic clips.
- `create_boundary.py`: Interactive zone calibration helper for sidewalk polygon points.
- `bytetrack_tuned.yaml`: ByteTrack configuration for continuity-oriented tracking.
- `pyproject.toml`: Project metadata and dependency definitions.

## Requirements

- Python 3.12+
- pip (comes with standard Python installations)
- Optional but recommended: NVIDIA GPU for faster YOLO and VAE runs

## Setup

For most users, this is the recommended setup.

1. Create a virtual environment:

```bash
python -m venv .venv
```

2. Activate it:

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Bash:

```bash
source .venv/Scripts/activate
```

3. Upgrade pip and install required packages:

```bash
python -m pip install --upgrade pip
pip install notebook ipykernel matplotlib opencv-python ultralytics torch torchvision lap tqdm shapely pandas
```

4. Launch notebook:

```bash
python -m notebook
```

5. Open `YOLO.ipynb` and run the cells in workflow order.

### Optional: Other environment managers

If you prefer conda:

```bash
conda create -n cv-cctv python=3.12 -y
conda activate cv-cctv
pip install notebook ipykernel matplotlib opencv-python ultralytics torch torchvision lap tqdm shapely pandas
python -m notebook
```

If you prefer uv:

```bash
uv sync
source .venv/Scripts/activate
uv run jupyter notebook
```

## Recommended End-to-End Workflow

### 1) Run YOLO monitoring pipeline

In `YOLO.ipynb`, run Cell 1 first.

This generates:

- monitoring output video (for example `11x_monitoring_output.mp4`)
- tracklet feature export (`tracklet_features.csv`)

### 2) Build normal-traffic clips (if needed)

Use the splicer GUI to cut normal traffic segments:

```bash
python vae_sample_splicer_gui.py
```

Default output is typically a folder of clips (for example `vae_normal_samples`).

### 3) Split normal clips into left and right datasets

Split using divider logic from `YOLO.ipynb` (for example `road_divider_x = int(w / 1.88)`):

```bash
python split_vae_samples_lr.py
```

Default outputs:

- `vae_split_samples/left`
- `vae_split_samples/right`

Useful options:

```bash
python split_vae_samples_lr.py --input vae_normal_samples --recursive
python split_vae_samples_lr.py --divider-x 1021 --buffer-px 8
```

If you use uv, prepend commands with `uv run`.

### 4) Train and run split VAE in notebook

In `YOLO.ipynb`:

1. Run Cell 5 to load or train left and right VAE models.
2. Run Cell 6 to calibrate thresholds and detect anomalies.

Cell 5 currently uses:

- `VAE_NORMAL_SOURCE_LEFT = "vae_split_samples/left"`
- `VAE_NORMAL_SOURCE_RIGHT = "vae_split_samples/right"`

Cell 6 writes split anomaly output:

- `vae_anomaly_scores_split_lr.csv`

## Key Configuration Notes

### Split VAE data and weights

- Left and right models are trained independently from separate folders.
- Weight files are separate:
	- `temporal_vae_left_weights.pt`
	- `temporal_vae_right_weights.pt`

### Test source

In Cell 5, `VAE_TEST_SOURCE` points to the video used for anomaly scoring timeline.

If it is missing, the notebook falls back to split-from-normal mode.

### Divider quality

If left and right behavior seems mixed, verify:

- `road_divider_x` in Cell 1
- `VAE_DIVIDER_BUFFER` in Cell 5

## ByteTrack Tuning Guidance

Current file: `bytetrack_tuned.yaml`

For stronger continuity (less blinking):

- Lower `track_high_thresh`
- Lower `track_low_thresh`
- Lower `new_track_thresh`
- Higher `track_buffer`
- Higher `match_thresh`
- Keep `fuse_score: true`

Tradeoff: continuity improves, but risk of false matches and ID switches can increase in crowded scenes.

## Outputs You Should Expect

- Monitoring video output, for example `11x_monitoring_output.mp4`
- Tracklet feature table: `tracklet_features.csv`
- VAE anomaly scores: `vae_anomaly_scores_split_lr.csv`
- VAE weights: `temporal_vae_left_weights.pt`, `temporal_vae_right_weights.pt`

## Troubleshooting

- If notebook says missing split folders, run the splitter first and check:
	- `vae_split_samples/left`
	- `vae_split_samples/right`

- If model is not retraining, existing weights may be loaded automatically.
	- Delete old weight files or change filenames to force retraining.

- If timeline windows do not appear:
	- Confirm OpenCV GUI support is available in your environment.
	- Ensure video paths in Cell 5 and Cell 6 are valid.

- If `python` is not recognized on Windows due to App Execution Alias:
	- Try `./.venv/Scripts/python.exe` directly, or
	- Run through your environment manager (`uv run`, conda env python, etc.)

```bash
./.venv/Scripts/python.exe split_vae_samples_lr.py --help
```

