# Sports Analytics System

An advanced biomechanical analysis platform designed for professional football performance tracking, injury risk assessment, and musculoskeletal modeling. This system utilizes a hybrid pipeline combining custom YOLOv11-based tracking with Sports2D for clinical-grade joint kinematics and OpenSim compatibility.

---

## Core Capabilities

### Hybrid Tracking and Pose Estimation

- **Neural Engine**: Utilizes YOLOv11-Pose for robust 2D keypoint detection.
- **ByteTracker + Kalman**: Implements consistent subject tracking with MOG2 background subtraction, maintaining lock through scene cuts and occlusions.
- **Scene Intelligence**: Automated scene change detection and target re-acquisition.

### Clinical Biomechanics (Sports2D Integration)

- **Deep Kinematics**: Optional integration with Sports2D for joint angle calculations and automated side-view analysis.
- **OpenSim Pipeline**: Generates .TRC (marker trajectories) and .MOT (joint angles) files compatible with Inverse Kinematics (IK) in OpenSim.
- **Interactive Subject Selection**: High-precision selection via mouse click for complex professional match footage.

### Comprehensive Metric Suite

- **Gait and Locomotion**: Analysis of speed, acceleration, cadence, stride length, and flight time.
- **Joint Monitoring**: Real-time tracking of knee flexion/extension, hip flexion, and clinical Valgus/Varus indices.
- **Injury Risk Intelligence**: Multi-factor scoring systems involving fatigue indexing, joint stress, and gait asymmetry.
- **Energy Expenditure**: Metabolic cost estimation (kcal/hr) based on movement intensity.

---

## Project Structure

```text
Sports Analytics System/
📂 assets/             # Branding and logo assets
📂 dashboard/          # Frontend source code (HTML/CSS/JS)
  📄index.html
  📄dashboard.js
  📄dashboard.html
  📄dashboard.css
📂 docs/               # Technical documentation and reports
📂 models/             # YOLO model weights (.pt files)
📂 src/                # Core Package Source
  📂 analytics/        # Biomechanical engine
    📄sports_analytics.py
  📂 api/              # FastAPI Backend and local servers
    📄main.py
    📄serve_dashboard.py
📂 videos/             # Input and sample video files
📂 Output/             # Local analysis results (JSON, CSV, MOT, TRC)
📄 run_analysis.py     # Main CLI entry point
📄 requirements.txt    # Python dependencies
☁️ Procfile            # Cloud deployment configuration
📄 .env                # Environment variables
📄 README.md           # Project documentation
```

---

## Installation and Setup (Conda / Windows)

This guide covers the complete setup including Anaconda installation, environment creation, and OpenSim integration.

### 1. Prerequisite: Anaconda

1. Download the Windows Installer from [Anaconda Individual Edition](https://www.anaconda.com/download).
2. Follow the installation prompts. It is recommended to check "Add Anaconda3 to my PATH" and "Register Anaconda3 as my default Python".
3. Verify by running `conda --version` in your terminal.

### 2. Environment Creation

Create a dedicated environment to avoid dependency conflicts:

```bash
conda create -n sports2d-full python=3.10 -y
conda activate sports2d-full
```

### 3. Dependency Installation

Install core components in the following order:

**OpenSim API (Required for 3D Markers)**

```bash
conda install -c opensim-org opensim -y
```

**Sports2D and Pose2Sim**

```bash
pip install sports2d pose2sim
```

**Computer Vision and AI Libraries**

```bash
pip install ultralytics opencv-python pandas scipy numpy matplotlib
```

### 4. Verification

Ensure all components are correctly linked:

```bash
python -c "import opensim; import sports2d; from ultralytics import YOLO; print('SUCCESS: Pipeline Linked')"
```

---

## Usage Guide

### 1. Local CLI Analysis

Run the full pipeline with interactive player picking:

```bash
python run_analysis.py --video videos/20.mp4 --sports2d --s2d-pick
```

### 2. Advanced OpenSim IK

Generate scaled models and motion data for OpenSim:

```bash
python run_analysis.py --video videos/20.mp4 --sports2d --s2d-pick --s2d-ik --height 1.80 --mass 75
```

### 3. Web Dashboard

Launch the unified management portal locally:

```bash
python src/api/serve_dashboard.py
```

Access the interface at: [http://localhost:8000](http://localhost:8000)

---

## Citations

The biomechanical analysis in this project is powered by **Sports2D**. Please cite the following if you use this research:

> [!NOTE]
> Pagnon, D., & Kim, H. (2024). Sports2D: Compute 2D human pose and angles from a video or a webcam. Journal of Open Source Software, 9(101), 6849. [https://doi.org/10.21105/joss.06849](https://doi.org/10.21105/joss.06849)

---

© 2026 Sports Analytics System. Powered by Mitus AI.
