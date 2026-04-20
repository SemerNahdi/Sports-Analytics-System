import os
import sys
import uuid
import time
import json
import shutil
import signal
import socket
import tempfile
import subprocess
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@dataclass
class PickerSession:
    session_id: str
    created_at: float
    status: str  # creating | loading | running | finished | failed | cancelled
    error: Optional[str]
    video_path: str
    original_filename: str
    result_dir: str
    display: str
    vnc_port: int
    ws_port: int
    pids: Dict[str, int]
    selection: Optional[Dict[str, Any]]


class PickerSessionManager:
    """
    Manages Sports2D 'on_click' picker sessions by running Sports2D in a headless X11
    display and exposing the GUI through VNC + noVNC (websockify).
    """

    def __init__(self):
        self._sessions: Dict[str, PickerSession] = {}
        self._persistence_file = os.path.join(tempfile.gettempdir(), "mitus_picker_sessions", "sessions.json")
        self._load_sessions()

    def _save_sessions(self):
        try:
            os.makedirs(os.path.dirname(self._persistence_file), exist_ok=True)
            data = {sid: asdict(s) for sid, s in self._sessions.items()}
            with open(self._persistence_file, "w") as f:
                json.dump(data, f)
        except Exception as e:
            print(f"[PICKER] Failed to save sessions: {e}")

    def _load_sessions(self):
        if not os.path.exists(self._persistence_file):
            return
        try:
            with open(self._persistence_file, "r") as f:
                data = json.load(f)
            for sid, sdata in data.items():
                # On load, if a session was 'running', 'loading' or 'creating', 
                # it's now stale because the parent process restarted.
                if sdata.get("status") in ("running", "loading", "creating"):
                    sdata["status"] = "failed"
                    sdata["error"] = "Session lost due to server restart."
                self._sessions[sid] = PickerSession(**sdata)
        except Exception as e:
            print(f"[PICKER] Failed to load sessions: {e}")

    def get(self, session_id: str) -> Optional[PickerSession]:
        return self._sessions.get(session_id)

    def create_from_upload(self, upload_tmp_path: str, original_filename: str) -> PickerSession:
        session_id = str(uuid.uuid4())
        base_dir = os.path.join(tempfile.gettempdir(), "mitus_picker_sessions", session_id)
        os.makedirs(base_dir, exist_ok=True)

        video_path = os.path.join(base_dir, f"input_{original_filename}")
        shutil.move(upload_tmp_path, video_path)

        result_dir = os.path.join(base_dir, "Sports2D")
        os.makedirs(result_dir, exist_ok=True)

        # Allocate ports per session
        vnc_port = _pick_free_port()
        ws_port = _pick_free_port()

        # Allocate a display number derived from ports (best-effort uniqueness)
        display_num = int(str(ws_port)[-2:]) + 90
        display = f":{display_num}"

        sess = PickerSession(
            session_id=session_id,
            created_at=time.time(),
            status="creating",
            error=None,
            video_path=video_path,
            original_filename=original_filename,
            result_dir=result_dir,
            display=display,
            vnc_port=vnc_port,
            ws_port=ws_port,
            pids={},
            selection=None,
        )
        self._sessions[session_id] = sess
        self._save_sessions()

        # Launch the GUI stack and Sports2D in background
        try:
            self._launch(sess)
        except Exception as e:
            print(f"[PICKER] Launch error for {session_id}: {e}")
            sess.status = "failed"
            sess.error = str(e)
        
        self._save_sessions()
        return sess

    def _launch(self, sess: PickerSession) -> None:
        # Lazy import to avoid circular dependency
        from src.analytics.sports_analytics import Sports2DRunner

        # If on Windows/MacOS or if no Xvfb is available, we might want a direct local window.
        # However, for a web API, we usually want the virtual buffer.
        is_linux = os.name == 'posix'
        
        if is_linux:
            env = os.environ.copy()
            env["DISPLAY"] = sess.display

            # Xvfb (virtual framebuffer)
            try:
                xvfb = subprocess.Popen(
                    ["Xvfb", sess.display, "-screen", "0", "1280x720x24", "-ac"],
                    env=env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                sess.pids["xvfb"] = xvfb.pid
            except FileNotFoundError:
                print("[PICKER] Xvfb not found. Falling back to local display if available.")
                is_linux = False

        if is_linux:
            env = os.environ.copy()
            env["DISPLAY"] = sess.display
            # fluxbox (window manager)
            try:
                fb = subprocess.Popen(["fluxbox"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                sess.pids["fluxbox"] = fb.pid
            except: pass

            # x11vnc (VNC server)
            try:
                vnc = subprocess.Popen(
                    ["x11vnc", "-display", sess.display, "-nopw", "-listen", "localhost", "-rfbport", str(sess.vnc_port), "-forever"],
                    env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
                sess.pids["vnc"] = vnc.pid
            except: pass

            # websockify (noVNC gateway)
            try:
                # noVNC is usually in /usr/share/novnc or similar
                novnc_path = "/usr/share/novnc"
                ws = subprocess.Popen(
                    ["websockify", "--web", novnc_path, str(sess.ws_port), f"localhost:{sess.vnc_port}"],
                    env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
                sess.pids["websockify"] = ws.pid
            except: pass

        # Finally, start the Sports2D Picker in its own PROCESS via a helper script.
        # This is the most robust way on Windows to avoid the threading error
        # and ensure it doesn't interfere with the FastAPI/Uvicorn process.
        worker_script = os.path.join(os.path.dirname(__file__), "picker_worker.py")
        
        # On Windows, we should let stdout/stderr go to the main console so we can see what's wrong.
        # In a real production server we'd log to a file, but for local debugging this is better.
        log_file = os.path.join(os.path.dirname(sess.result_dir), "picker_worker.log")
        lf = open(log_file, "w")
        p = subprocess.Popen(
            [sys.executable, worker_script, sess.video_path, sess.result_dir, sess.session_id],
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0,
            stdout=lf,
            stderr=lf,
        )
        # We don't close lf here because Popen needs the handle.
        # It will be closed when the parent process exits or we can track it.
        # But for now, this ensures the handle stays open for the child.
        sess.pids["sports2d"] = p.pid

    def poll(self, session_id: str) -> Optional[PickerSession]:
        sess = self._sessions.get(session_id)
        if not sess:
            return None

        if sess.status not in ("creating", "loading", "running"):
            return sess

        # If Sports2D process finished, parse outputs on disk.
        sports2d_pid = sess.pids.get("sports2d")
        if not sports2d_pid:
            return sess

        # Check sentinel files for status updates
        try:
            init_started = os.path.join(sess.result_dir, ".init_started")
            gui_ready = os.path.join(sess.result_dir, ".gui_ready")
            done_sentinel = os.path.join(sess.result_dir, ".done")
            error_file = os.path.join(sess.result_dir, ".error")

            if os.path.exists(error_file):
                with open(error_file, "r") as f:
                    sess.error = f.read()
                sess.status = "failed"
                self._stop_gui_stack(sess, keep_outputs=True)
                return sess

            if os.path.exists(done_sentinel):
                sess.selection = self._derive_selection_from_disk(sess)
                if sess.selection:
                    sess.status = "finished"
                else:
                    sess.status = "failed"
                    sess.error = "Picker finished but selection was not captured."
                self._stop_gui_stack(sess, keep_outputs=True)
                return sess

            if os.path.exists(gui_ready):
                sess.status = "running"
            elif os.path.exists(init_started):
                sess.status = "loading"
            
            # Check if the process is still alive.
            if not _pid_alive(sports2d_pid):
                # If it's not alive and didn't finish normally via sentinel, it might have crashed.
                # Check outputs anyway.
                sess.selection = self._derive_selection_from_disk(sess)
                if sess.selection:
                    sess.status = "finished"
                else:
                    sess.status = "failed"
                    if not sess.error:
                        sess.error = "Sports2D process terminated unexpectedly."
                self._stop_gui_stack(sess, keep_outputs=True)
                
                self._save_sessions()
                return sess
        except Exception as e:
            print(f"[PICKER] Poll error: {e}")
            return sess

        self._save_sessions()
        return sess

    def finalize(self, session_id: str) -> Optional[PickerSession]:
        sess = self._sessions.get(session_id)
        if not sess:
            return None
        # Poll once to update status and attempt to extract selection.
        res = self.poll(session_id)
        self._save_sessions()
        return res

    def cancel(self, session_id: str) -> Optional[PickerSession]:
        sess = self._sessions.get(session_id)
        if not sess:
            return None
        sess.status = "cancelled"
        self._stop_gui_stack(sess, keep_outputs=True)
        self._save_sessions()
        return sess

    def cleanup(self, session_id: str) -> bool:
        sess = self._sessions.get(session_id)
        if not sess:
            return False
        self._stop_gui_stack(sess, keep_outputs=False)
        # Remove directory
        try:
            base_dir = os.path.dirname(sess.result_dir)
            shutil.rmtree(base_dir, ignore_errors=True)
        except Exception:
            pass
        del self._sessions[session_id]
        self._save_sessions()
        return True

    def public_payload(self, sess: PickerSession, request_base_url: str) -> Dict[str, Any]:
        """
        Returns a JSON-serializable payload for frontend.
        NOTE: noVNC is served by websockify on a dedicated port.
        """
        base_host = request_base_url.split("://", 1)[-1].split("/", 1)[0].split(":")[0]
        novnc_url = f"http://{base_host}:{sess.ws_port}/vnc.html?autoconnect=1&resize=scale"
        payload = asdict(sess)
        payload["novnc_url"] = novnc_url
        # Drop internal-only paths from frontend
        payload.pop("video_path", None)
        payload.pop("result_dir", None)
        return payload

    def _derive_selection_from_disk(self, sess: PickerSession) -> Optional[Dict[str, Any]]:
        try:
            # Give the filesystem a moment to settle on Windows
            time.sleep(1.0)
            from src.analytics.sports_analytics import Sports2DRunner
            r = Sports2DRunner(
                video_path=sess.video_path,
                result_dir=sess.result_dir,
                person_ordering="greatest_displacement",
                show_realtime=False,
            )
            r.outputs = r._collect_outputs()
            seed = r.get_seed_from_trc()
            if not seed:
                return None
            return {"seed_bbox": seed.get("seed_bbox"), "seed_frame": seed.get("seed_frame")}
        except Exception as e:
            sess.error = str(e)
            return None

    def _stop_gui_stack(self, sess: PickerSession, keep_outputs: bool) -> None:
        # Stop child processes (best-effort)
        for key in ("sports2d", "websockify", "x11vnc", "fluxbox", "xvfb"):
            pid = sess.pids.get(key)
            if pid:
                _kill_pid(pid)
        if not keep_outputs:
            try:
                base_dir = os.path.dirname(sess.result_dir)
                shutil.rmtree(base_dir, ignore_errors=True)
            except Exception:
                pass


def _pid_alive(pid: int) -> bool:
    try:
        import psutil
        return psutil.pid_exists(pid)
    except ImportError:
        try:
            os.kill(pid, 0)
            return True
        except Exception:
            return False


def _kill_pid(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGTERM)
    except Exception:
        return
    # Give it a moment; then SIGKILL if still alive.
    for _ in range(10):
        time.sleep(0.05)
        if not _pid_alive(pid):
            return
    try:
        os.kill(pid, signal.SIGKILL)
    except Exception:
        pass

