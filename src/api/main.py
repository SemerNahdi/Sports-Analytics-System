import os
import ssl
import datetime
import uuid
import json
import shutil
import tempfile
import glob
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List, Dict

# Fix for SSL_CERT_FILE issue: if it's set to a non-existent path, httpx (used by supabase) will fail.
if "SSL_CERT_FILE" in os.environ and not os.path.exists(os.environ["SSL_CERT_FILE"]):
    os.environ.pop("SSL_CERT_FILE", None)
from dataclasses import asdict
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from supabase import create_client, Client
import cloudinary
import cloudinary.uploader
from cloudinary.utils import cloudinary_url

# Import the analytics core lazily in the endpoint to speed up startup on Render
# from src.analytics.sports_analytics import SportsAnalyzer, AnalyticsPlotter, HAS_SPORTS2D

# Load environment variables
load_dotenv()

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://your-url.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "your-service-role-key")
# Use the bucket name from .env or default to 'Sports Analytics'
BUCKET_NAME = os.getenv("SUPABASE_BUCKET", "Sports Analytics")

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Cloudinary configuration
cloudinary.config(
    cloud_name = os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key = os.getenv("CLOUDINARY_API_KEY"),
    api_secret = os.getenv("CLOUDINARY_API_SECRET"),
    secure = True
)


app = FastAPI(
    title="Sports Analytics API",
    description="Advanced Biomechanics & Tracking backend with Supabase Storage integration."
)

# Defaults for analysis speed optimizations (override via env)
DEFAULT_YOLO_SIZE = os.getenv("YOLO_SIZE_DEFAULT", "n")
DEFAULT_STRIDE = int(os.getenv("ANALYSIS_STRIDE", "2"))
DEFAULT_TARGET_HEIGHT = int(os.getenv("ANALYSIS_TARGET_HEIGHT", "640"))

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Helpers (moved up)

# --- GLOBAL STATE FOR JOB TRACKING ---
# Stores {job_id: threading.Event()} to signal cancellation
active_jobs: Dict[str, threading.Event] = {}

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
    """Upload a single file to Supabase and return its public URL or a signed URL."""
    try:
        content_type = get_content_type(local_path)
        with open(local_path, "rb") as f:
            supabase.storage.from_(BUCKET_NAME).upload(
                path=remote_path,
                file=f,
                file_options={"content-type": content_type, "upsert": "true"}
            )
        
        # We use a signed URL to avoid issues if the bucket is not public
        # Setting a very long expiry (e.g., 10 years) for "permanent" dashboard access
        try:
            signed_url_res = supabase.storage.from_(BUCKET_NAME).create_signed_url(remote_path, 315360000) # 10 years
            if isinstance(signed_url_res, dict) and "signedURL" in signed_url_res:
                return signed_url_res["signedURL"]
            elif hasattr(signed_url_res, "signed_url"):
                return signed_url_res.signed_url
            return supabase.storage.from_(BUCKET_NAME).get_public_url(remote_path)
        except:
            return supabase.storage.from_(BUCKET_NAME).get_public_url(remote_path)
            
    except Exception as e:
        print(f"[Supabase Upload Error] {local_path} -> {remote_path}: {e}")
        return ""


def upload_video_to_cloudinary(local_path: str, public_id: str) -> str:
    """
    Upload a video to Cloudinary with automatic transcoding (f_auto).
    Returns the secure URL for public viewing.
    """
    try:
        print(f"[Cloudinary] Uploading video: {local_path}...")
        response = cloudinary.uploader.upload(
            local_path,
            public_id = public_id,
            resource_type = "video",
            overwrite = True,
            # 'f_auto' automatically chooses the best video codec for the viewer's browser
            # 'q_auto' chooses optimal compression quality
            transformation = [
                {'fetch_format': "auto", 'quality': "auto"}
            ]
        )
        return response.get("secure_url", "")
    except Exception as e:
        print(f"[Cloudinary Error] {e}")
        return ""

def upload_directory_to_supabase(directory: str, prefix: str) -> dict:
    """Recursively upload a directory to Supabase in parallel and return a dict of public URLs."""
    urls = {}
    tasks = []
    
    # We use a ThreadPoolExecutor for concurrent uploads
    with ThreadPoolExecutor(max_workers=5) as executor:
        for root, _, files in os.walk(directory):
            for file in files:
                local_path = os.path.join(root, file)
                # Create a relative path for Supabase
                rel_path = os.path.relpath(local_path, directory).replace("\\", "/")
                remote_path = f"{prefix}/{rel_path}"
                
                # Submit upload task
                tasks.append((rel_path, executor.submit(upload_file_to_supabase, local_path, remote_path)))
        
        # Collect results as they complete
        for rel_path, future in tasks:
            url = future.result()
            if url:
                urls[rel_path] = url
                
    return urls

def send_analysis_email(to_email: str, job_id: str, player_id: int, video_url: str):
    """Sends a summary email to the user with a link to their results."""
    smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER")
    smtp_password = os.getenv("SMTP_PASSWORD")

    if not all([smtp_user, smtp_password]):
        print(f"[Email Bypassed] Missing SMTP credentials. Email would have gone to {to_email}")
        return

    # Clean the credentials (Gmail App Passwords should have no spaces)
    smtp_user = smtp_user.strip()
    smtp_password = smtp_password.replace(" ", "").strip()
    to_email = to_email.strip()

    try:
        msg = MIMEMultipart()
        msg['From'] = smtp_user
        msg['To'] = to_email
        msg['Subject'] = f"Mitus AI: Analysis Complete - Player #{player_id}"

        dashboard_url = f"{os.getenv('BASE_URL', 'http://localhost:8000')}/dashboard.html?job_id={job_id}"
        
        body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
            <div style="max-width: 600px; margin: 0 auto; border: 1px solid #eee; padding: 20px; border-radius: 10px;">
                <h1 style="color: #00f0ff; background: #06070a; padding: 20px; border-radius: 10px; text-align: center; margin: 0;">Mitus AI</h1>
                <h3 style="text-align: center; color: #555; margin-top: 20px;">Analysis Report Finalized</h3>
                <hr style="border: 0; border-top: 1px solid #eee; margin: 20px 0;">
                <p>Hello coach,</p>
                <p>The biomechanical analysis for <strong>Player #{player_id}</strong> is ready for review.</p>
                <div style="margin: 20px 0; padding: 15px; background: #f9f9f9; border-left: 4px solid #00f0ff;">
                    <p style="margin: 0;"><strong>Job ID:</strong> {job_id}</p>
                    <p style="margin: 5px 0 0 0;"><strong>Status:</strong> Success</p>
                </div>
                <p>You can access the full interactive dashboard and downloadable reports below:</p>
                <div style="text-align: center; margin-top: 30px;">
                    <a href="{dashboard_url}" style="display: inline-block; padding: 15px 35px; background: #00f0ff; color: #000; text-decoration: none; border-radius: 8px; font-weight: 900; letter-spacing: 1px; text-transform: uppercase; box-shadow: 0 4px 15px rgba(0,240,255,0.3);">View Results Dashboard</a>
                </div>
                <p style="margin-top: 30px; font-size: 0.8rem; color: #888;">Note: The annotated video is also available directly at: <a href="{video_url}">{video_url}</a></p>
                <hr style="border: 0; border-top: 1px solid #eee; margin: 20px 0;">
                <p style="font-size: 0.7rem; color: #aaa; text-align: center;">This is an automated notification from the Mitus AI Sports Analytics System.</p>
            </div>
        </body>
        </html>
        """
        msg.attach(MIMEText(body, 'html'))

        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(smtp_user, smtp_password)
        server.send_message(msg)
        server.quit()
        print(f"[Email Sent] Analysis report for {job_id} sent to {to_email}")
    except Exception as e:
        print(f"[Email Error] Failed to send to {to_email}: {e}")

# ── API Endpoints ─────────────────────────────────────────────────────────────

# static_dir is now in the project root as 'dashboard'
static_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "dashboard")

@app.get("/")
async def root():
    """Serve the root index page."""
    return FileResponse(os.path.join(static_dir, "index.html"))

@app.get("/dashboard.html")
async def dashboard():
    """Serve the analytics dashboard."""
    return FileResponse(os.path.join(static_dir, "dashboard.html"))

@app.get("/health")
async def health():
    """Fast health check for Render to detect the open port instantly."""
    return {"status": "ok", "service": "Mitus AI Sports Analytics System"}

def run_full_analysis_job(
    job_id: str,
    temp_input_path: str,
    player_id: int,
    yolo_size: str,
    player_height: float,
    mass_kg: float,
    session_tags: str,
    run_sports2d: bool,
    original_filename: str,
    email: Optional[str] = None,
    stride: int = DEFAULT_STRIDE,
    target_height: int = DEFAULT_TARGET_HEIGHT
):
    """
    The heavy-lifting background task that runs the AI analysis and uploads results.
    """
    import psutil
    def log_mem():
        mem = psutil.virtual_memory()
        print(f"[MEM] {mem.percent}% used | Available: {mem.available // (1024*1024)}MB")

    # Telemetry logger to push backend updates to the frontend
    job_logs = []
    
    # Initialize cancellation event for this job
    cancel_event = threading.Event()
    active_jobs[job_id] = cancel_event

    def log_step(msg):
        # Check for cancellation before every major step
        if cancel_event.is_set():
            raise InterruptedError("Job cancelled by user.")

        print(f"[JOB {job_id[:8]}] {msg}")
        log_mem()
        job_logs.append(f"{datetime.datetime.now().strftime('%H:%M:%S')} - {msg}")
        try:
            supabase.table("analyses").update({"logs": job_logs}).eq("id", job_id).execute()
        except: pass

    log_step("Initializing AI environment...")
    
    # Lazy import to prevent blocking startup during port binding
    from src.analytics.sports_analytics import SportsAnalyzer, AnalyticsPlotter, HAS_SPORTS2D

    # Use a temporary directory for processing - ensures NO local leftover files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Setup final paths
        input_path = os.path.join(temp_dir, "input_" + original_filename)
        output_video_name = "output_annotated.mp4"
        output_video_path = os.path.join(temp_dir, output_video_name)
        
        results_dir = os.path.join(temp_dir, "results")
        data_dir = os.path.join(temp_dir, "data")
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)

        # Move the uploaded file from its temp landing spot to our working dir
        try:
            shutil.move(temp_input_path, input_path)
            log_step(f"Video file validated. Size: {os.path.getsize(input_path)//1024}KB")
        except Exception as e:
            log_step(f"CRITICAL ERROR: File move failed: {e}")
            supabase.table("analyses").update({"status": "failed", "error": str(e)}).eq("id", job_id).execute()
            return

        try:
            # 3. Initialize and run SportsAnalyzer
            log_step(f"Loading Neural Network ({yolo_size})...")
            import gc
            gc.collect() # Force cleanup before loading heavy model

            analyzer = SportsAnalyzer(
                video_path=input_path,
                output_video_path=output_video_path,
                player_id=player_id,
                yolo_size=yolo_size,
                player_height_m=player_height,
                pick=False 
            )
            
            log_step("Commencing Pose Estimation & Tracking...")
            summary = analyzer.process_video(stride=stride, target_height=target_height, cancel_event=cancel_event)
            log_step(f"Tracking concluded. {len(analyzer.frame_metrics)} frames analyzed.")
            gc.collect()

            # 4. Run optional Sports2D pipeline
            if run_sports2d:
                log_step("Invoking deep clinical pipeline (Sports2D)...")
                s2d_dir = os.path.join(temp_dir, "Sports2D")
                analyzer.run_sports2d(
                    result_dir=s2d_dir,
                    mode="balanced",
                    participant_mass_kg=mass_kg
                )
                log_step("Clinical data extracted.")
                gc.collect()

            # 5. Export unified data (JSON, CSV, TRC, MOT)
            log_step("Synchronizing biomechanical datasets...")
            json_out = os.path.join(data_dir, "analytics_unified.json")
            csv_out = os.path.join(data_dir, "bio_metrics.csv")
            trc_out = os.path.join(data_dir, "trajectories.trc")
            mot_out = os.path.join(data_dir, "motions.mot")
            report_out = os.path.join(data_dir, "report.txt")

            unified_data = analyzer.export_unified(
                json_path=json_out,
                csv_path=csv_out,
                trc_path=trc_out,
                mot_path=mot_out,
            )
            unified_frames = unified_data.get("frames", [])
            
            # Save the report string too
            with open(report_out, "w", encoding="utf-8") as f:
                f.write(analyzer.get_report_string())

            # 6. Generate Analytics Plots
            log_step("Synthesizing graphical metrics...")
            plotter = AnalyticsPlotter(results_dir=results_dir, player_id=player_id)
            plotter.generate_all(
                frame_metrics=analyzer.frame_metrics,
                bio_engine=analyzer.bio_engine
            )

            # 7. Upload All Assets (Video to Cloudinary, Others to Supabase)
            log_step("Uploading finalized assets to cloud storage...")
            asset_prefix = f"jobs/{job_id}"
            
            # VIDEO OPTIMIZATION
            video_url = upload_video_to_cloudinary(output_video_path, f"mitus_ai_analytics_{job_id}")
            if not video_url:
                log_step("Cloudinary bypass: using direct Supabase storage.")
                video_url = upload_file_to_supabase(output_video_path, f"{asset_prefix}/{output_video_name}")

            # Upload data files and plots
            data_urls = upload_directory_to_supabase(data_dir, f"{asset_prefix}/data")
            plot_urls = upload_directory_to_supabase(results_dir, f"{asset_prefix}/plots")

            # Final update to Supabase record
            log_step("Job finalized successfully.")
            full_summary = {
                "player_summary": asdict(summary),
                "biomechanics_summary": analyzer.bio_engine.summary_dict() if analyzer.bio_engine else {},
                "frame_metrics": unified_frames
            }

            supabase.table("analyses").update({
                "status": "success",
                "video_url": video_url,
                "summary": full_summary,
                "data_urls": data_urls,
                "plot_urls": plot_urls,
                "player_height": player_height,
                "mass_kg": mass_kg,
                "yolo_size": yolo_size,
                "run_sports2d": run_sports2d
            }).eq("id", job_id).execute()

            # --- SEND EMAIL NOTIFICATION ---
            if email:
                print(f"[JOB {job_id[:8]}] Queuing email notification...")
                supabase.table("analyses").update({
                    "logs": logs + [f"[{datetime.now().isoformat()}] - Dispatching report email to {email}..."]
                }).eq("id", job_id).execute()
                
                try:
                    send_analysis_email(email, job_id, player_id, video_url)
                    logs.append(f"[{datetime.now().isoformat()}] - Email report delivered successfully.")
                except Exception as eval_err:
                    logs.append(f"[{datetime.now().isoformat()}] - Email Error: {str(eval_err)}")
                
                supabase.table("analyses").update({"logs": logs}).eq("id", job_id).execute()

            print(f"[JOB {job_id[:8]}] Successfully completed.")

        except Exception as e:
            import traceback
            traceback.print_exc()
            
            # Use specific status if it was a cancellation
            final_status = "cancelled" if isinstance(e, InterruptedError) else "failed"
            error_msg = "Job cancelled by user." if isinstance(e, InterruptedError) else str(e)
            
            log_step(f"FATAL ERROR: {error_msg}")
            try:
                supabase.table("analyses").update({
                    "status": final_status,
                    "error": error_msg
                }).eq("id", job_id).execute()
            except:
                pass
            print(f"[JOB {job_id[:8]}] {final_status.capitalize()} with error: {error_msg}")
        
        finally:
            # Clean up the initial uploaded file
            if os.path.exists(temp_input_path):
                try:
                    os.remove(temp_input_path)
                except: pass

            # Always clean up the job tracker
            if job_id in active_jobs:
                del active_jobs[job_id]

@app.post("/analyze")
async def analyze_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    player_id: int = 1,
    yolo_size: str = DEFAULT_YOLO_SIZE,
    player_height: float = 1.75,
    mass_kg: float = 75.0,
    session_tags: str = "performance-match",
    run_sports2d: bool = False,
    email: Optional[str] = None,
    stride: int = DEFAULT_STRIDE,
    target_height: int = DEFAULT_TARGET_HEIGHT
):
    """
    Asynchronous analysis endpoint:
    1. Verifies the database connection and schema first.
    2. Returns '202 Accepted' with job_id immediately (avoids Render 502 timeout).
    3. Processes AI analysis in the BackgroundTask pool.
    """
    if not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="File must be a video.")

    job_id = str(uuid.uuid4())
    print(f"\n[JOB {job_id[:8]}] Attempting to initialize job record...")

    # --- CRITICAL: FAST DATABASE VERIFICATION ---
    # We try to create the record NOW. If this fails (e.g. missing 'status' column), 
    # we return an error to the user IMMEDIATELY instead of a 404 loop later.
    try:
        supabase.table("analyses").insert({
            "id": job_id,
            "player_id": player_id,
            "status": "processing",
            "session_tags": session_tags
        }).execute()
        print(f"[JOB {job_id[:8]}] DB record initialized.")
    except Exception as e:
        error_str = str(e)
        print(f"[JOB {job_id[:8]}] DB Initialization FAILED: {error_str}")
        
        # If the error is 'PGRST204', we give the specific fix
        if "PGRST204" in error_str or "status" in error_str.lower():
            return JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "message": "SCHEMA ERROR: Your Supabase 'analyses' table is missing the 'status' column.",
                    "detail": "Please go to Suapbase SQL Editor and run: ALTER TABLE public.analyses ADD COLUMN status TEXT DEFAULT 'processing';"
                }
            )
        
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": "Database Error", "detail": error_str}
        )

    # Save a temporary copy of the file for the background worker
    temp_dir_base = tempfile.gettempdir()
    os.makedirs(os.path.join(temp_dir_base, "mitus_uploads"), exist_ok=True)
    temp_input_path = os.path.join(temp_dir_base, "mitus_uploads", f"{job_id}_{file.filename}")

    with open(temp_input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Queue the heavy work
    background_tasks.add_task(
        run_full_analysis_job,
        job_id,
        temp_input_path,
        player_id,
        yolo_size,
        player_height,
        mass_kg,
        session_tags,
        run_sports2d,
        file.filename,
        email,
        stride,
        target_height
    )

    return JSONResponse(
        status_code=202,
        content={
            "status": "processing",
            "job_id": job_id,
            "message": "Analysis queued successfully."
        }
    )

@app.post("/analyses/{job_id}/cancel")
async def cancel_analysis(job_id: str):
    """
    Cancel a running or queued job.
    """
    if job_id in active_jobs:
        print(f"[API] Signaling cancellation for job {job_id[:8]}...")
        active_jobs[job_id].set()
        
        # Immediate DB update to reflect intent
        try:
            supabase.table("analyses").update({
                "status": "cancelling",
                "error": "User requested cancellation..."
            }).eq("id", job_id).execute()
        except: pass
        
        return {"status": "success", "message": "Cancellation signal sent."}
    
    # If not in active_jobs, it might be in the database as 'processing' or 'pending'
    # we should still try to mark it as cancelled there.
    try:
        res = supabase.table("analyses").select("status").eq("id", job_id).execute()
        if res.data and res.data[0]["status"] in ["processing", "pending"]:
            supabase.table("analyses").update({"status": "cancelled"}).eq("id", job_id).execute()
            return {"status": "success", "message": "Job marked as cancelled in database."}
    except: pass

    return JSONResponse(
        status_code=404,
        content={"status": "error", "message": "Job not found or not active."}
    )

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
        # Standard Supabase query
        res = supabase.table("analyses").select("*").eq("id", job_id).execute()
        
        # If the query itself succeeds but returns no data, it's a 404
        if not res.data:
            return JSONResponse(
                status_code=404,
                content={"status": "error", "message": f"Job {job_id} not found in database."}
            )
        
        return res.data[0]
    except Exception as e:
        # Log to server console
        print(f"[Supabase DEBUG] Error fetching {job_id}: {str(e)}")
        # Return the error to the user so they can read it in the browser console
        return JSONResponse(
            status_code=500,
            content={
                "status": "error", 
                "message": "Database query failed", 
                "detail": str(e)
            }
        )
        # tempfile.TemporaryDirectory() automatically handles cleanup of ALL files when 'with' block exits

# Final Catch-all for Static Assets (JS, CSS, images)
app.mount("/", StaticFiles(directory=static_dir), name="static")

if __name__ == "__main__":
    import uvicorn
    # Get port from environment or default to 8000
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting Mitus AI Sports Analytics API on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)
