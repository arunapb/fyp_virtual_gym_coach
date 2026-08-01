"""
app.py — start the web front end.

    python app.py                 # http://127.0.0.1:8000
    python app.py --port 9000
    python app.py --host 0.0.0.0  # reachable from another device

Upload a recorded video, get the full form analysis in a browser: annotated
playback, live risk zones, per-rep breakdown, coaching audio at the right
moments, and the CSVs the evaluation scripts consume.

This is one of two front ends over the same pipeline in `src/`.  The other is
`desktop_app.py`, the live OpenCV window driven by a webcam or the sample clip.
Neither replaces the other, and both produce the same numbers.
"""

import argparse
import os
import sys
import webbrowser


def main():
    parser = argparse.ArgumentParser(description="Module 2 pose-correction web app")
    parser.add_argument("--host", default="127.0.0.1",
                        help="interface to bind (default: 127.0.0.1, local only)")
    parser.add_argument("--port", type=int, default=8000, help="port (default: 8000)")
    parser.add_argument("--reload", action="store_true",
                        help="auto-reload on source changes (development)")
    parser.add_argument("--no-browser", action="store_true",
                        help="do not open a browser window on startup")
    args = parser.parse_args()

    # Imported here so `--help` works without the web dependencies installed.
    try:
        import uvicorn
    except ImportError:
        sys.exit("FastAPI/uvicorn are not installed.\n"
                 "Install them with:  pip install -r requirements.txt")

    # uvicorn imports "api" by name, so the repository root has to be importable
    # regardless of where this script was launched from.
    root = os.path.dirname(os.path.abspath(__file__))
    if root not in sys.path:
        sys.path.insert(0, root)

    url = f"http://{'127.0.0.1' if args.host == '0.0.0.0' else args.host}:{args.port}"
    print(f"\n  Module 2 — Pose Correction\n  {url}\n  Ctrl+C to stop\n")
    if not args.no_browser and not args.reload:
        # Skipped under --reload: the reloader re-executes this module, which
        # would open a new tab on every code change.
        webbrowser.open(url)

    uvicorn.run("api:app", host=args.host, port=args.port,
                reload=args.reload, log_level="info", app_dir=root)


if __name__ == "__main__":
    main()
