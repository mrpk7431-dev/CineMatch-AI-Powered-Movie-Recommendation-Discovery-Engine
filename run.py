import sys
import os
import threading
import webbrowser

# Add app directory to Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app.app import app
from waitress import serve

def open_browser(port):
    try:
        webbrowser.open(f"http://127.0.0.1:{port}")
    except Exception as e:
        pass

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    # Only open browser when running locally (not on Render cloud)
    if "RENDER" not in os.environ and "PORT" not in os.environ:
        threading.Timer(1.2, open_browser, args=(port,)).start()

    print(f"Starting server on 0.0.0.0:{port}...")
    serve(app, host='0.0.0.0', port=port)


