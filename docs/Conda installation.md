# Installation & Setup Guide (Windows)

This guide covers the complete setup from scratch, including Anaconda installation, environment creation, OpenSim API integration, and running your first 3D analysis.

---


## Step 1: Install Anaconda (If not already installed)

1.  **Download:** Go to [Anaconda Individual Edition](https://www.anaconda.com/download) and download the **Windows Installer**.
2.  **Install:** Run the installer.
    - Check **"Add Anaconda3 to my PATH environment variable"** (Recommended for easy terminal access, though Anaconda Prompt is safer).
    - Check **"Register Anaconda3 as my default Python"**.
3.  **Verify:**
    - Open **Git Bash** or **Command Prompt**.
    - Type: `conda --version`
    - *Result:* Should print a version number (e.g., `conda 24.x.x`).

> ⚠️ **First-Time Setup (Terms of Service):**
> If you see a `CondaToSNonInteractiveError`, run these three commands once to accept Anaconda's terms:
> ```bash
> conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
> conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
> conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/msys2
> ```

---

## Step 2: Create the Dedicated Environment

We create a specific environment (`sports2d-full`) to isolate dependencies and avoid conflicts with your other projects.

1.  **Open Terminal:** Open **Git Bash** or **Anaconda Prompt**.
2.  **Create Environment:**
    ```bash
    conda create -n sports2d-full python=3.10 -y
    ```
3.  **Activate Environment:**
    ```bash
    conda activate sports2d-full
    ```
    *(You should see `(sports2d-full)` at the start of your command line).*

---

## Step 3: Install Dependencies

Install the core libraries in this specific order to ensure DLL compatibility on Windows.

### 3.1 Install OpenSim API (Critical for 3D Skeleton)
This is the most important step for generating `.osim` and `_ik.mot` files.
```bash
conda install -c opensim-org opensim -y
```

### 3.2 Install Sports2D & Pose2Sim
```bash
pip install sports2d pose2sim
```

### 3.3 Install Computer Vision & AI Libraries
```bash
pip install ultralytics opencv-python pandas scipy numpy matplotlib
```

### 3.4 Verify Installation
Run this single command to ensure everything is linked correctly:
```bash
python -c "import opensim; import sports2d; from ultralytics import YOLO; print('✅ SUCCESS: OpenSim', opensim.__version__, '| Sports2D | YOLO')"
```
*If this prints without error, you are ready.*

---

## 🏃 Step 4: Running Your First Analysis

Navigate to your project folder and run the analysis with Inverse Kinematics enabled.

1.  **Navigate to Project:**
    ```bash
    cd "C:/Users/semer/Desktop/eSteps/Mitus AI Sports Analytics System"
    ```
    *(Note: Use forward slashes `/` or escape backslashes `\\` in Git Bash)*

2.  **Ensure Environment is Active:**
    ```bash
    conda activate sports2d-full
    ```

3.  **Run the Command:**
    Replace `match.mp4` with your actual video file.
    ```bash
 sports2d --video_input white_attack.mp4 --person_ordering_method on_click --do_ik true --use_augmentation True --visible_side auto front --show_realtime_results false
    ```

    **Flag Explanation:**
    - `--pick`: Opens a window to click and select the player.
    - `--sports2d`: Triggers the native Sports2D pipeline (generates TRC/MOT files).
    - `--height 1.80`: Sets player height for accurate scaling (adjust as needed).

---

## Step 5: Viewing the 3D Skeleton in OpenSim GUI

Once the script finishes, it will generate files in the `Output/Sports2D/` folder.

1.  **Open OpenSim GUI:**
    - Search "OpenSim" in your Windows Start Menu and launch it.
2.  **Load the Scaled Model:**
    - Go to `File` → `Open Model`.
    - Navigate to `Output/Sports2D/` and select the file ending in **`_LSTM.osim`** (e.g., `match_m_person00_LSTM.osim`).
    - *You will see a 3D skeleton appear.*
3.  **Load the Motion Data:**
    - Go to `File` → `Load Motion`.
    - Select the file ending in **`_ik.mot`** (e.g., `match_m_person00_LSTM_ik.mot`).
    - *The skeleton will now animate based on your video.*
4.  **Play:** Click the **Play** button (green triangle) in the bottom toolbar.

---

## Troubleshooting Cheat Sheet

| Issue | Solution |
| :--- | :--- |
| **`ModuleNotFoundError: No module named 'opensim'`** | You forgot to activate the env. Run `con activate sports2d-full` before running python. |
| **`CondaToSNonInteractiveError`** | Run the 3 `conda tos accept ...` commands listed in Step 1. |
| **VideoWriter Error / Codec Issue** | The script tries multiple codecs (`mp4v`, `avc1`). If all fail, install `ffmpeg` via conda: `conda install -c conda-forge ffmpeg`. |
| **OpenSim GUI crashes on load** | Ensure you loaded the `_LSTM.osim` file, not a generic model. Ensure the `_ik.mot` file matches the model name. |
| **Slow Performance** | Ensure you are using `--yolo-size n` or `s` for testing. Use `m` or `l` only if you have a dedicated GPU. |

---

## Quick Reference: Daily Workflow

Every time you want to work on the project:

```bash
# 1. Open Terminal (Git Bash / Anaconda Prompt)
# 2. Activate Env
conda activate sports2d-full

# 3. Go to Project
cd "C:/Users/semer/Desktop/eSteps/Juventus Sports Analytics System"

# 4. Run Analysis
python run_analysis.py --video <your_video>.mp4 --pick --sports2d
```