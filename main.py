import os
import ssl

# Fix for SSL_CERT_FILE issue: if it's set to a non-existent path, httpx (used by supabase) will fail.
# We remove it from the environment if it's invalid so and ssl.create_default_context() uses system certs.
if "SSL_CERT_FILE" in os.environ and not os.path.exists(os.environ["SSL_CERT_FILE"]):
    os.environ.pop("SSL_CERT_FILE", None)
import uuid
import json
import shutil
import tempfile
import glob
from typing import Optional, List
from dataclasses import asdict

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from supabase import create_client, Client

# Import the analytics core
from sports_analytics import SportsAnalyzer, AnalyticsPlotter, HAS_SPORTS2D

# Load environment variables
load_dotenv()

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://your-url.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "your-service-role-key")
# Use the bucket name from .env or default to 'Sports Analytics'
BUCKET_NAME = os.getenv("SUPABASE_BUCKET", "Sports Analytics")

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

app = FastAPI(
    title="Juventus Sports Analytics API",
    description="Advanced Biomechanics & Tracking backend with Supabase Storage integration."
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Helpers (moved up)

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_content_type(filename: str) -> str:
    """Guess content type based on file extension."""
    ext = os.path.splitext(filename)[1].lower()
    mapping = {
        ".mp4": "video/mp4",
        ".json": "application/json",
        ".csv": "text/csv",
        ".txt": "text/plain",
        ".png": "image/png",
        ".svg": "image/svg+xml",
        ".trc": "text/plain",
        ".mot": "text/plain",
    }
    return mapping.get(ext, "application/octet-stream")

def upload_file_to_supabase(local_path: str, remote_path: str) -> str:
    """Upload a single file to Supabase and return its public URL."""
    try:
        content_type = get_content_type(local_path)
        with open(local_path, "rb") as f:
            supabase.storage.from_(BUCKET_NAME).upload(
                path=remote_path,
                file=f,
                file_options={"content-type": content_type, "upsert": "true"}
            )
        return supabase.storage.from_(BUCKET_NAME).get_public_url(remote_path)
    except Exception as e:
        print(f"[Supabase Upload Error] {local_path} -> {remote_path}: {e}")
        return ""

def upload_directory_to_supabase(directory: str, prefix: str) -> dict:
    """Recursively upload a directory to Supabase and return a dict of public URLs."""
    urls = {}
    for root, _, files in os.walk(directory):
        for file in files:
            local_path = os.path.join(root, file)
            # Create a relative path for Supabase
            rel_path = os.path.relpath(local_path, directory).replace("\\", "/")
            remote_path = f"{prefix}/{rel_path}"
            url = upload_file_to_supabase(local_path, remote_path)
            if url:
                urls[rel_path] = url
    return urls

# ── API Endpoints ─────────────────────────────────────────────────────────────

static_dir = os.path.join(os.path.dirname(__file__), "static_ui")

@app.get("/")
async def root():
    """Serve the root index page."""
    return FileResponse(os.path.join(static_dir, "index.html"))

@app.get("/dashboard.html")
async def dashboard():
    """Serve the analytics dashboard."""
    return FileResponse(os.path.join(static_dir, "dashboard.html"))

@app.post("/analyze")
async def analyze_video(
    file: UploadFile = File(...),
    player_id: int = 1,
    yolo_size: str = "m",
    player_height: float = 1.75,
    mass_kg: float = 75.0,
    session_tags: str = "juventus-match",
    run_sports2d: bool = False
):
    """
    Core analysis endpoint:
    1. Receives video file
    2. Runs full tracking & biomechanics
    3. Generates plots and reports
    4. Uploads EVERYTHING to Supabase
    5. Cleans up all local temporary storage
    """
    if not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="File must be a video.")

    # Generate a unique job ID
    job_id = str(uuid.uuid4())
    print(f"\n[JOB {job_id[:8]}] Starting analysis for player #{player_id}")

    # Use a temporary directory for processing - ensures NO local leftover files
    with tempfile.TemporaryDirectory() as temp_dir:
        # 1. Setup paths
        input_path = os.path.join(temp_dir, "input_" + file.filename)
        output_video_name = "output_annotated.mp4"
        output_video_path = os.path.join(temp_dir, output_video_name)
        
        results_dir = os.path.join(temp_dir, "results")
        data_dir = os.path.join(temp_dir, "data")
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)

        # 2. Save uploaded file to temp directory
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        try:
            # 3. Initialize and run SportsAnalyzer
            # Note: pick=False for API (no interactive UI)
            analyzer = SportsAnalyzer(
                video_path=input_path,
                output_video_path=output_video_path,
                player_id=player_id,
                yolo_size=yolo_size,
                player_height_m=player_height,
                pick=False 
            )
            
            summary = analyzer.process_video()
            
            # 4. Run optional Sports2D pipeline
            if run_sports2d:
                print(f"[JOB {job_id[:8]}] Running Sports2D pipeline...")
                s2d_dir = os.path.join(temp_dir, "Sports2D")
                analyzer.run_sports2d(
                    result_dir=s2d_dir,
                    mode="balanced",
                    participant_mass_kg=mass_kg
                )

            # 5. Export unified data (JSON, CSV, TRC, MOT)
            json_out = os.path.join(data_dir, "analytics_unified.json")
            csv_out = os.path.join(data_dir, "bio_metrics.csv")
            trc_out = os.path.join(data_dir, "trajectories.trc")
            mot_out = os.path.join(data_dir, "motions.mot")
            report_out = os.path.join(data_dir, "report.txt")

            analyzer.export_unified(
                json_path=json_out,
                csv_path=csv_out,
                trc_path=trc_out,
                mot_path=mot_out,
            )
            
            # Save the report string too
            with open(report_out, "w", encoding="utf-8") as f:
                f.write(analyzer.get_report_string())

            # 6. Generate Analytics Plots
            plotter = AnalyticsPlotter(results_dir=results_dir, player_id=player_id)
            plotter.generate_all(
                frame_metrics=analyzer.frame_metrics,
                bio_engine=analyzer.bio_engine
            )

            # 7. Upload All Assets to Supabase
            print(f"[JOB {job_id[:8]}] Uploading results to Supabase bucket: {BUCKET_NAME}")
            
            asset_prefix = f"jobs/{job_id}"
            
            # Upload annotated video
            video_url = upload_file_to_supabase(output_video_path, f"{asset_prefix}/{output_video_name}")
            
            # Upload data files
            data_urls = upload_directory_to_supabase(data_dir, f"{asset_prefix}/data")
            
            # Upload plots
            plot_urls = upload_directory_to_supabase(results_dir, f"{asset_prefix}/plots")

            # Final response payload
            results = {
                "status": "success",
                "job_id": job_id,
                "player_id": player_id,
                "player_height": player_height,
                "mass_kg": mass_kg,
                "yolo_size": yolo_size,
                "run_sports2d": run_sports2d,
                "session_tags": session_tags,
                "results": {
                    "annotated_video": video_url,
                    "data_files": data_urls,
                    "analytical_plots": plot_urls
                },
                "summary": asdict(summary)
            }

            # 8. Store results in Supabase Database Table 'analyses'
            # (Assumes table 'analyses' exists with columns matching the data structure)
            try:
                supabase.table("analyses").insert({
                    "id": job_id,
                    "player_id": player_id,
                    "player_height": player_height,
                    "mass_kg": mass_kg,
                    "yolo_size": yolo_size,
                    "run_sports2d": run_sports2d,
                    "session_tags": session_tags,
                    "video_url": video_url,
                    "summary": results["summary"],
                    "data_urls": data_urls,
                    "plot_urls": plot_urls
                }).execute()
            except Exception as e:
                print(f"[Supabase DB Error] Could not insert record: {e}")
                # We don't fail the whole request because assets are already in storage

            return JSONResponse(content=results)

        except Exception as e:
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/analyses")
async def list_analyses(limit: int = 20):
    """List all recent analyses from Supabase."""
    try:
        response = supabase.table("analyses").select("*").order("created_at", desc=True).limit(limit).execute()
        return response.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch analyses: {str(e)}")

@app.get("/analyses/latest")
async def get_latest_analysis():
    """Fetch the single most recent analysis."""
    try:
        response = supabase.table("analyses").select("*").order("created_at", desc=True).limit(1).execute()
        if not response.data:
            raise HTTPException(status_code=404, detail="No analyses found.")
        return response.data[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch latest analysis: {str(e)}")

@app.get("/analyses/{job_id}")
async def get_analysis(job_id: str):
    """Fetch detail for a specific analysis job."""
    try:
        response = supabase.table("analyses").select("*").eq("id", job_id).execute()
        if not response.data:
            raise HTTPException(status_code=404, detail="Analysis not found.")
        return response.data[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch analysis {job_id}: {str(e)}")
        # tempfile.TemporaryDirectory() automatically handles cleanup of ALL files when 'with' block exits

# Final Catch-all for Static Assets (JS, CSS, images)
app.mount("/", StaticFiles(directory=static_dir), name="static")

if __name__ == "__main__":
    import uvicorn
    # Get port from environment or default to 8000
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting Juventus Sports Analytics API on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)
