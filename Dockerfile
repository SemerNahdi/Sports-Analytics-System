# Use a slim Python base image
FROM python:3.10-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies for OpenCV and Sports2D
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglx-mesa0 \
    libglib2.0-0 \
    ffmpeg \
    libsm6 \
    libxext6 \
    curl \
    xvfb \
    fluxbox \
    x11vnc \
    novnc \
    websockify \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces requirements: user ID 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Copy requirements and install
# Note: we use --chown=user for HF permissions
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY --chown=user . .

# Expose the API port (HF expects 7860 by default)
EXPOSE 7860

# Use uvicorn to serve the FastAPI app
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "7860"]
