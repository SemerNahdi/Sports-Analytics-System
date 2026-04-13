import http.server
import socketserver
import os
import sys

# Define port and directory
# Use PORT environment variable for Render compatibility
PORT = int(os.environ.get("PORT", 8000))
DIRECTORY = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "dashboard")

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def log_message(self, format, *args):
        # Redirect log messages to stdout for Render logs
        sys.stdout.write("%s - - [%s] %s\n" %
                         (self.address_string(),
                          self.log_date_time_string(),
                          format % args))

def start_server():
    # Use 0.0.0.0 to bind to all interfaces for cloud hosting
    with socketserver.TCPServer(("0.0.0.0", PORT), Handler) as httpd:
        print(f"Server started globally (needed for Render).")
        print(f"For local testing, go to: http://localhost:{PORT}/dashboard.html")
        httpd.serve_forever()

if __name__ == '__main__':
    print("Starting Mitus AI Sports Analytics Dashboard Server...")
    # Change to current dir
    os.chdir(DIRECTORY)
    
    try:
        start_server()
    except KeyboardInterrupt:
        print("\nStopping server...")
        sys.exit(0)
