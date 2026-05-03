"""Development watcher to auto-restart the Gradio app on Python file changes.

Usage:
  1. Install dependency: `pip install watchdog`
  2. From the `src` folder run: `python dev_reload.py`

This script watches the `src` directory recursively and restarts
`src/ui/gradio_app.py` whenever a `.py` file changes. It's intentionally
small and dependency-light for local development only.
"""
from pathlib import Path
import subprocess
import sys
import time
import threading

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / "ui" / "gradio_app.py"
DEBOUNCE_SECS = 0.4


class RestartHandler(FileSystemEventHandler):
    def __init__(self, restart_cb):
        self.restart_cb = restart_cb
        self._timer = None

    def _debounce(self):
        if self._timer:
            self._timer.cancel()
        self._timer = threading.Timer(DEBOUNCE_SECS, self.restart_cb)
        self._timer.start()

    def on_any_event(self, event):
        if event.is_directory:
            return
        if not event.src_path.endswith('.py'):
            return
        self._debounce()


def run():
    proc = None

    def start_proc():
        nonlocal proc
        if proc and proc.poll() is None:
            return
        print(f"Starting: {SCRIPT}")
        proc = subprocess.Popen([sys.executable, str(SCRIPT)])

    def stop_proc():
        nonlocal proc
        if not proc:
            return
        try:
            proc.terminate()
            proc.wait(5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        proc = None

    def restart():
        nonlocal proc
        print("Change detected — restarting Gradio app...")
        if proc and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        start_proc()

    if not SCRIPT.exists():
        print(f"Error: expected script at {SCRIPT} not found.")
        return

    start_proc()

    event_handler = RestartHandler(restart)
    observer = Observer()
    observer.schedule(event_handler, str(ROOT), recursive=True)
    observer.start()

    try:
        while True:
            time.sleep(1)
            # If process died, restart it
            if proc and proc.poll() is not None:
                print("Gradio process exited; restarting...")
                start_proc()
    except KeyboardInterrupt:
        print("Stopping watcher and app...")
        observer.stop()
        stop_proc()
    observer.join()


if __name__ == '__main__':
    run()
