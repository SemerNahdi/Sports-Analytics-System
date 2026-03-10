# Juventus Sports Analytics System

A high-performance biomechanical analysis tool designed for single-player tracking and movement metrics extraction. This system leverages computer vision and pose estimation to provide detailed insights into athletic performance, gait symmetry, and injury risk.

---

## Core Architecture

The system follows a modular pipeline designed for precision and reliability:

1.  **Pre-scan Phase**:
    Analyzes the entire video to identify the most persistent player ID, ensuring the correct subject is tracked even in crowded scenes.
2.  **Tracking & Verification**:
    Utilizes a MIL (Multiple Instance Learning) tracker combined with MOG2 background subtraction for robust player localization and blob verification.
3.  **Pose Estimation**:
    Integrates MediaPipe Pose to extract 3D-equivalent keypoints for 18 distinct joints, including head, neck, hips, knees, and ankles.
4.  **Signal Processing**:
    Implements Kalman filtering for joint coordinate smoothing and Savitzky-Golay filters for derived metrics like velocity and acceleration.
5.  **Metrics Extraction**:
    Calculates biomechanical data including speed, cadence, stride length, joint angles, and valgus/varus stress.
6.  **Risk Assessment**:
    Analyzes gait symmetry and joint stress to generate a composite injury risk score.

---

## Features

- **Automated Player Selection**: Smart detection of the primary subject based on visibility duration.
- **Interactive Tracking**: Option to manually pick the player to track by clicking in the video frame.
- **Biomechanical Metrics**:
  - Real-time speed and acceleration tracking.
  - Gait analysis (cadence, stride length, step time).
  - Joint angle monitoring (knee flexion/extension, valgus stress).
- **Risk Scoring**: Composite analysis of movement patterns to identify potential injury risks.
- **Annotated Visualization**: Generates a high-quality video with overlaid skeletal tracking and live metrics dashboard.
- **Data Export**: Comprehensive data output in JSON and CSV formats for further research and longitudinal analysis.

---

## Prerequisites

- Python 3.8 or higher
- FFmpeg (for video encoding)

### Dependencies

Install the required Python packages using pip:

```bash
pip install -r requirements.txt
```

Core libraries include:

- `mediapipe`: Pose estimation and keypoint detection.
- `opencv-python`: Video processing and visualization.
- `numpy` & `pandas`: Data manipulation and numerical analysis.
- `scipy` & `scikit-learn`: Signal processing and advanced analytics.

---

## How to Run

The system is executed via `run_analysis.py`.

### Basic Usage

Automatically select and track the most persistent player:

```bash
python run_analysis.py --video path/to/match.mp4
```

### Interactive Manual Selection

Open a window to click on the specific player you wish to track:

```bash
python run_analysis.py --video path/to/match.mp4 --pick
```

### Advanced Options

Specify output paths, frames per second, and specific player IDs:

```bash
python run_analysis.py --video match.mp4 --output analysis.mp4 --player 10 --fps 30
```

### Arguments

| Argument   | Description                               | Default                       |
| :--------- | :---------------------------------------- | :---------------------------- |
| `--video`  | Path to the input video file (Required).  | -                             |
| `--output` | Path for the annotated output video.      | `Output/output_annotated.mp4` |
| `--player` | Player jersey/ID label to search for.     | `1`                           |
| `--pick`   | Enable interactive mouse-click selection. | `False`                       |
| `--fps`    | Override video FPS.                       | Video Metadata                |
| `--json`   | Path for JSON metrics export.             | `Output/metrics.json`         |
| `--csv`    | Path for CSV metrics export.              | `Output/metrics.csv`          |

---

## Project Structure

- `run_analysis.py`: Main entry point and CLI handler.
- `sports_analytics.py`: Core logic, including tracking, pose estimation, and metrics calculation.
- `Output/`: Default directory for all generated analysis results.
- `input/`: Recommended directory for source video files.

---

## Output Artifacts

Upon completion, the system generates:

- **Annotated Video**: A visual representation of the tracking and biomechanical data.
- **Metrics JSON**: Full raw data for integration with other software.
- **Metrics CSV**: Tabular data ready for Excel or Python-based research.
- **Terminal Summary**: A text report containing aggregate performance statistics.
