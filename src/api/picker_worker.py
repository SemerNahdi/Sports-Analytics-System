
import sys
import os
import json
import signal

# Ensure project root is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def main():
    if len(sys.argv) < 4:
        print("Usage: picker_worker.py <video_path> <result_dir> <session_id>")
        sys.exit(1)

    video_path = sys.argv[1]
    result_dir = sys.argv[2]
    session_id = sys.argv[3]

    # Sentinel paths
    sentinel_init = os.path.join(result_dir, ".init_started")
    sentinel_ready = os.path.join(result_dir, ".gui_ready")
    sentinel_done = os.path.join(result_dir, ".done")
    error_file = os.path.join(result_dir, ".error")

    # Indicate initialization started
    try:
        with open(sentinel_init, "w") as f: f.write("1")
    except: pass

    try:
        # For Windows, force an interactive backend for matplotlib BEFORE importing sports2d
        # This is critical for the on_click picker window to appear.
        if os.name == 'nt':
            try:
                import matplotlib
                # Try common interactive backends
                backends = ['TkAgg', 'Qt5Agg', 'QtAgg', 'WXAgg']
                success = False
                for b in backends:
                    try:
                        matplotlib.use(b, force=True)
                        success = True
                        break
                    except: continue
                if not success:
                    print("[PICKER WORKER] Warning: No interactive matplotlib backend found.")
            except ImportError:
                pass

        # Import inside the function to keep the outer script light
        from src.analytics.sports_analytics import Sports2DRunner
        
        # Indicate we are actually running (GUI ready)
        try:
            with open(sentinel_ready, "w") as f: f.write("1")
        except: pass

        runner = Sports2DRunner(
            video_path=video_path,
            result_dir=result_dir,
            person_ordering="on_click",
            show_realtime=True,
            show_realtime_results=False,
        )
        runner.run()

        # Indicate we finished successfully
        try:
            with open(sentinel_done, "w") as f: f.write("1")
        except: pass
    except (KeyboardInterrupt, SystemExit):
        pass
    except Exception as e:
        print(f"[PICKER WORKER] Failed: {e}")
        try:
            with open(error_file, "w") as f: f.write(str(e))
        except: pass
        sys.exit(1)
    finally:
        # Clean up sentinels (but NOT .done or .error)
        for s in [sentinel_init, sentinel_ready]:
            try:
                if os.path.exists(s): os.remove(s)
            except: pass

if __name__ == "__main__":
    main()
