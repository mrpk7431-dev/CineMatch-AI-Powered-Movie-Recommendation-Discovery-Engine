import sys
import os
import threading
import webbrowser

# Add app directory to Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app.app import app
from waitress import serve

def open_browser():
    try:
        webbrowser.open("http://127.0.0.1:5000")
    except Exception as e:
        print(f"Could not open browser automatically: {e}")

if __name__ == "__main__":
    url = "http://127.0.0.1:5000"
    print(f"Starting server on {url}")
    threading.Timer(1.2, open_browser).start()
    serve(app, host='127.0.0.1', port=5000)

