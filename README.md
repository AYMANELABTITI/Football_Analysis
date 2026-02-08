# Football Analysis Pipeline (YOLOv8 + Tracking + Reports)

This project analyzes a football match video and generates:

- an annotated output video
- player trajectories and distance metrics
- pass detection outputs
- an Excel report (`.xlsx`) with detailed match statistics
- visual analytics images (heatmap, pass map, trajectory map, dashboards)

The current main entry point is `main.py`.

## 1. Features

- Player and ball detection using YOLOv8
- Multi-object tracking (SORT/Kalman)
- Team color assignment (jersey color clustering + temporal stabilization)
- Ball continuity recovery when short detection gaps happen
- Pass detection with anti-false-positive filters:
  - minimum pass distance
  - minimum/maximum duration
  - ball displacement check
  - pass direction angle check
- Per-player trajectory and distance tracking
- Automatic exports:
  - annotated video
  - `analysis.xlsx` (multiple sheets)
  - pass map image
  - trajectory map image
  - distance chart image
  - global stats dashboard image
  - heatmap image

## 2. Project Structure

```text
football_analysis/
|- config/
|  |- config.yaml
|  `- config_pro.yaml
|- data/
|  `- raw/
|- results/
|  |- videos/
|  |- stats/
|  `- reports/
|- src/
|  |- detection/
|  |- tracking/
|  |- events/
|  |- visualization/
|  `- utils/
|- main.py
|- requirements.txt
`- test_system.py
```

## 3. Requirements

- Python 3.10+ (3.11 recommended)
- NVIDIA GPU recommended for practical speed
- CUDA-compatible PyTorch for GPU inference

## 4. Installation

### Windows (recommended)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If activation is blocked, run commands directly with:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Linux/macOS

```bash
python -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 5. Quick Start

Put your match video in `data/raw/` and run:

```powershell
.\venv\Scripts\python.exe main.py data/raw/Barcelona.mp4 --config config/config_pro.yaml
```

## 6. CLI Usage

```text
python main.py video
  [--output OUTPUT]
  [--no-skeleton]
  [--no-heatmap]
  [--no-minimap]
  [--config CONFIG]
  [--calib-p1 x,y]
  [--calib-p2 x,y]
  [--calib-distance-m M]
```

### Main options

- `video`: input video path (required)
- `--output`: custom output video path
- `--config`: config file (default: `config/config_pro.yaml`)
- `--calib-p1`, `--calib-p2`, `--calib-distance-m`: metric calibration inputs

## 7. Metric Calibration (px -> meters)

Distance conversion is done in this priority order:

1. `field.distance_scale_px_to_m` in config, if > 0
2. two-point calibration (`p1`, `p2`, known distance)
3. fallback approximation: `105 / video_width`

Example with midfield width:

```powershell
.\venv\Scripts\python.exe main.py data/raw/Barcelona.mp4 `
  --config config/config_pro.yaml `
  --calib-p1 120,540 `
  --calib-p2 1810,540 `
  --calib-distance-m 68
```

## 8. Outputs

For an input like `Barcelona.mp4`, outputs are saved under `results/videos/`:

- `professional_Barcelona.mp4`
- `professional_Barcelona_analysis.xlsx`
- `professional_Barcelona_heatmap.png`
- `professional_Barcelona_passes.png`
- `professional_Barcelona_trajectories.png`
- `professional_Barcelona_distance_chart.png`
- `professional_Barcelona_stats_dashboard.png`

### Excel sheets

`professional_Barcelona_analysis.xlsx` includes:

- `summary`
- `frame_details`
- `passes`
- `pass_network`
- `player_distances`
- `team_distances`

## 9. Important Config Keys

Edit `config/config_pro.yaml` (or `config/config.yaml`):

### Detection

- `detection.model`
- `detection.confidence`
- `detection.player_confidence`
- `detection.ball_confidence`
- `detection.img_size`

### Ball continuity

- `detection.ball_hold_frames`
- `detection.ball_link_distance_px`

### Pass filtering

- `events.pass_min_distance_px`
- `events.pass_min_duration_frames`
- `events.pass_max_duration_frames`
- `events.pass_max_angle_deg`
- `events.pass_min_ball_displacement_px`
- `events.pass_min_owner_frames`

### Distance scale / calibration

- `field.distance_scale_px_to_m`
- `field.calibration_p1`
- `field.calibration_p2`
- `field.calibration_distance_m`

## 10. Troubleshooting

### `ModuleNotFoundError` (example: `filterpy`)

You are likely using a different Python interpreter than project venv.

Use:

```powershell
.\venv\Scripts\python.exe main.py data/raw/Barcelona.mp4 --config config/config_pro.yaml
```

### `import` yellow warnings in VS Code

Set:

```json
{
  "python.analysis.extraPaths": ["./src"]
}
```

### OpenCV font errors

Use `cv2.FONT_HERSHEY_SIMPLEX` (already applied in this project).

### Slow processing

- lower `detection.img_size`
- use a smaller YOLO model in config
- disable some visual overlays if needed
