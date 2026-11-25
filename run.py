"""Launch the dashboard using installed packages. Never install dependencies."""

import argparse
import importlib.metadata
from pathlib import Path
import sys
import threading
import webbrowser

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "models" / "yolov8n-f2.pt"
HAND_MODELS = (ROOT / "models" / "hands" / "palm_detection_mediapipe_2023feb.onnx",
               ROOT / "models" / "hands" / "handpose_estimation_mediapipe_2023feb.onnx")


def check_setup():
    """Inspect the current interpreter without installing or loading the model."""
    ready = True
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    for name in ("Flask", "numpy", "opencv-python", "ultralytics", "matplotlib"):
        try:
            print(f"  {name}: {importlib.metadata.version(name)}")
        except importlib.metadata.PackageNotFoundError:
            print(f"  {name}: missing")
            ready = False
    print(f"Model: {'found' if MODEL.is_file() else 'MISSING'}")
    for path in HAND_MODELS:
        print(f"Hand model {path.name}: {'found' if path.is_file() else 'MISSING'}")
        ready = ready and path.is_file()
    try:
        print(f"Hand accelerator: ONNX Runtime {importlib.metadata.version('onnxruntime')} (already installed)")
    except importlib.metadata.PackageNotFoundError:
        print("Hand accelerator: OpenCV DNN fallback (no installation needed)")
    print("Setup looks ready." if ready and MODEL.is_file() else "Some requirements are missing in this interpreter. No packages were installed or changed.")
    return 0 if ready and MODEL.is_file() else 1


def serve(open_browser):
    import config
    from app import app
    from werkzeug.serving import make_server

    host = config.HOST
    browser_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    if ":" in browser_host:
        browser_host = f"[{browser_host}]"
    url = f"http://{browser_host}:{config.PORT}"
    # Bind before opening the browser so a busy port never opens a different app.
    server = make_server(host, config.PORT, app, threaded=True)
    timer = None
    if open_browser:
        timer = threading.Timer(0.8, webbrowser.open, args=(url,))
        timer.daemon = True
        timer.start()
    print(f"\nDashboard: {url}\nKeep this window open. Press Ctrl+C here to stop.\n", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping the tracker...", flush=True)
    finally:
        if timer:
            timer.cancel()
        server.server_close()
        registry = app.extensions["tracker_sessions"]
        with registry.lock:
            for browser in registry.sessions.values():
                browser.close()
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check packages and model without starting or installing anything")
    parser.add_argument("--no-browser", action="store_true", help="Start without opening a browser tab")
    parser.add_argument("--serve", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if sys.version_info < (3, 11):
        print("Python 3.11 or later is required. This project was verified with Python 3.11.")
        return 1
    if args.check:
        return check_setup()
    try:
        return serve(not args.no_browser)
    except KeyboardInterrupt:
        return 0
    except ModuleNotFoundError as exc:
        print(f"This Python environment is missing {exc.name}. Use the interpreter where the project's packages are already installed.")
        return 1
    except (OSError, ValueError) as exc:
        print(f"Could not start: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
