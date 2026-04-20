# Deployment Needs: Juventus Sports Analytics System

This document outlines the infrastructure, system dependencies, hardware requirements, and external services necessary to deploy the Juventus Sports Analytics System successfully to a production environment.

## 1. Application Architecture Overview

The system is a unified web application comprising:
- **Backend API:** Built with Python (FastAPI).
- **Core Processing Engine:** Video analysis using `ultralytics` (YOLOv11), `sports2d`, `opencv`, and `pose2sim`.
- **Frontend Dashboard:** Static web assets (HTML/CSS/JS) served by the backend or a CDN.

The application is fully containerized with a `Dockerfile` and includes configurations for multiple cloud providers (Fly.io, Hugging Face Spaces, Heroku/Render).

## 2. Infrastructure & Hosting Options

The repository already defines configurations for three main deployment routes:
- **Hugging Face Spaces (Recommended for AI):** Supported natively via the `Dockerfile`. Runs on default port `7860`. Hugging Face Spaces easily allows provisioning cloud GPUs (like T4, A10g) which is ideal for this workload.
- **Fly.io:** Configured in `fly.toml`. Currently scoped for a `shared` 2 vCPU machine with 4GB RAM. *Note: Running video pose estimation on CPUs will be significantly slower than on a GPU.*
- **Heroku / Render:** Supported via the included `Procfile`.

## 3. Hardware Requirements

Given the intense machine learning and computer vision processing required by the pipeline:
- **CPU:** Minimum 2-4 vCPUs for background task handling and FastAPI concurrency.
- **Memory (RAM):** Minimum 4GB (8GB+ recommended) to handle video frames in memory and load ML model weights.
- **GPU (Highly Recommended):** An NVIDIA GPU with CUDA support will accelerate YOLO and pose-estimation tasks drastically (e.g., from taking 10x video duration to near real-time). If deploying on AWS, GCP, or Hugging Face, look for T4 or A10g instances.

## 4. System & OS Dependencies

The core AI/CV libraries (`OpenCV`, `Sports2D`, OpenSim) require low-level system graphics and media libraries. These are handled in the `Dockerfile` and must be present in any non-Docker deployment server:

- **Media Processing:** `ffmpeg` (Hard requirement for video extraction/compression)
- **Computer Vision Graphics Libraries:** `libgl1`, `libglx-mesa0`, `libglib2.0-0`, `libsm6`, `libxext6`
- **Headless Display Servers:** `xvfb`, `fluxbox`, `x11vnc` (required if rendering any simulated views or plots without a physical display)

## 5. External Services (3rd Party APIs)

The project relies on external SaaS platforms for state and file management. These must be provisioned and their corresponding API keys acquired:
- **Supabase:** Used for the database, user authentication, or logging (as indicated by `supabase` in `requirements.txt`).
- **Cloudinary:** Used for robust video and media storage (as indicated by `cloudinary` in `requirements.txt`).

## 6. Environment Variables

Your deployment environment must pass the following secrets (typically defined in the local `.env` or `template.env` file):

- **Port Configuration:** Usually auto-assigned by the cloud provider, but the default internal port is `7860` (or `8080` for fly.io).
- **Service Keys:**
  - Supabase connection strings and API keys.
  - Cloudinary URL/credentials for media hosting.

*(Check `template.env` to view the exact expected keys).*

## 7. Storage / File System Needs

The system will momentarily generate substantial intermediate files:
- Temporary cached frames from videos.
- Large output assets like `.mp4` rendered videos, `.trc` markers, and `.mot` joint angles files.
- It is crucial that the deployment platform supports a **writable ephemeral file system** (such as Docker's internal file system or an attached volume), as serverless functions like AWS Lambda often restrict write access to `/tmp`.
