import http.server
import socketserver
import webbrowser
import threading
import os
import sys

# Define port and directory
PORT = 8000
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def log_message(self, format, *args):
        # Suppress log messages for cleaner output
        pass

def start_server():
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"Serving dashboard at http://localhost:{PORT}/dashboard.html")
        httpd.serve_forever()

if __name__ == '__main__':
    print("Starting local server for Juventus Analytics Dashboard...")
    # Change to current dir just to be safe
    os.chdir(DIRECTORY)
    
    # Start server in a separate thread so we can open the browser immediately
    server_thread = threading.Thread(target=start_server)
    server_thread.daemon = True
    server_thread.start()
    
    # Open default browser
    webbrowser.open_new_tab(f'http://localhost:{PORT}/dashboard.html')
    
    try:
        print("\nServer is running. Press Ctrl+C to stop.")
        # Keep main thread alive
        while True:
            pass
    except KeyboardInterrupt:
        print("\nStopping server...")
        sys.exit(0)
