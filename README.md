# Mitus AI Sports Analytics System (v6)

An advanced, high-performance biomechanical analysis platform designed for precise movement metrics extraction, injury risk assessment, and musculoskeletal modeling. The system integrates state-of-the-art computer vision (YOLOv11), professional kinematics (Sports2D), and cloud-native infrastructure to provide clinical-grade insights into athletic performance.

---

## 🚀 Core Capabilities

### 1. Advanced Machine Learning
- **Pose Estimation**: Powered by **YOLOv11-Pose** (Nano to X-Large models) for robust keypoint tracking in 2D.
- **Robust Tracking**: Custom **ByteTracker** implementation with Kalman filtering and MOG2 background subtraction for consistent subject lock across scene cuts and occlusions.
- **Scene Intelligence**: Automated scene change detection and target re-acquisition.

### 2. Professional Biomechanics (Sports2D & OpenSim)
- **Kinematics Pipeline**: Optional integration with **Sports2D** for authoritative joint angle calculations and automated side-view analysis.
- **OpenSim Compatibility**: Generates **.TRC** (marker trajectories) and **.MOT** (joint angles) files ready for Inverse Kinematics (IK) in OpenSim.
- **IK & Augmentation**: Support for LSTM-based marker augmentation to enhance musculoskeletal modeling precision.

### 3. Comprehensive Analytics Suite
- **Gait Analysis**: Cadence, stride length, step width, double support percentage, and foot progression angles.
- **Joint Monitoring**: Knee flexion/extension, hip flexion, and clinical Valgus/Varus (knee collapse) tracking.
- **Trunk & Posture**: Sagittal/Lateral trunk lean, pelvic rotation, and arm swing asymmetry.
- **Performance & Health**: Real-time speed, acceleration, energy expenditure (kcal/hr), fatigue indexing, and multi-factor injury risk scoring.

---

## 🏗️ Architecture & Stack

- **Backend**: FastAPI (Python 3.8+) for high-performance RESTful operations.
- **Computer Vision**: OpenCV, Ultralytics (YOLO), Sports2D, Pose2Sim.
- **Data Engineering**: NumPy, Pandas, SciPy for signal processing (Kalman, Savitzky-Golay).
- **Cloud Infrastructure**:
  - **Supabase**: Persistent storage for analysis metadata, JSON/CSV reports, and analytical plots.
  - **Cloudinary**: Optimized video hosting with automatic transcoding (H.264/WebM) and global delivery.
- **Frontend**: Modern, glassmorphic dashboard (HTML5/Vanilla CSS/JavaScript) with Chart.js for dynamic visualization.

---

## 🛠️ Getting Started

### Prerequisites
- **Python 3.9+**
- **FFmpeg** (must be on system PATH for video re-encoding)
- **Supabase & Cloudinary Accounts** (if running the API/Cloud mode)

### Installation
```bash
# Clone the repository
git clone https://github.com/SemerNahdi/Juventus-Sports-Analytics-System.git
cd "Juventus Sports Analytics System"

# Setup virtual environment
python -m venv .venv
source .venv/Scripts/activate  # Windows: .\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

---

## 📖 Usage Guide

### 1. CLI Entry (run_analysis.py)
The primary tool for local processing and OpenSim research.

**Basic Analysis (Custom Tracker):**
```bash
python run_analysis.py --video match.mp4 --player 1
```

**Clinical Analysis (Sports2D + OpenSim):**
```bash
# --sports2d: Run the deep kinematics pipeline
# --pick: Interactively select the subject via mouse click
# --s2d-ik: Generate OpenSim Inverse Kinematics files
python run_analysis.py --video march_clip.mp4 --sports2d --pick --s2d-ik
```

### 2. REST API (main.py)
Used for deployment (e.g., Render) and integration with the management portal.
```bash
python main.py
```
*The API automatically handles file uploads to Supabase and video transcoding via Cloudinary.*

### 3. Interactive Dashboard
Serve the portal and dashboard locally to browse historical results.
```bash
python serve_dashboard.py
```
Visit: [http://localhost:8000](http://localhost:8000)

---

## 📂 Project Structure

- `sports_analytics.py`: The "Engine" — contains tracking, pose logic, and the biomechanics calculator.
- `run_analysis.py`: CLI wrapper for batch and research processing.
- `main.py`: Production API with Supabase/Cloudinary middleware.
- `static_ui/`: Frontend source (Management Portal & Dashboard).
- `Output/`: standard directory for local results (JSON, CSV, MOT, TRC, MP4).

---

## 📊 Output Artifacts
Upon completion, the system generates:
- **Annotated Video**: High-fidelity skeleton overlays and live metric dashboards.
- **Kinematic Graphs**: Authoritative joint-angle plots (PNG) from Sports2D.
- **Unified Data**: Comprehensive metrics export in **JSON** and **CSV**.
- **Musculoskeletal Data**: **.TRC** and **.MOT** files for OpenSim integration.
- **Clinical Report**: Executive summary of gait symmetry, joint stress, and injury risks.

---
© 2026 Mitus AI Sports Analytics System. Proprietary Biomechanical Assessment Framework.

