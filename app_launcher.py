"""
app_launcher.py - Start FastAPI server and open UI in browser.

Use this script as the PyInstaller entry point.
"""

import logging
import os
import socket
import sys
import threading
import time
import traceback
import webbrowser
from pathlib import Path

# Fix for PyInstaller --noconsole and Uvicorn
if sys.stdout is None:
    class DummyStream:
        def write(self, *args, **kwargs): pass
        def flush(self, *args, **kwargs): pass
        def isatty(self): return False
    sys.stdout = DummyStream()
if sys.stderr is None:
    sys.stderr = DummyStream()

import uvicorn
from urllib import error as url_error
from urllib import request as url_request

# Import api at the top level so PyInstaller traces all dependencies
import api
from api import app


def _get_log_path() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).with_name("launcher.log")
    return Path(__file__).with_name("launcher.log")


def _port_available(host: str, port: int) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((host, port))
            return True
    except OSError:
        return False


def _choose_port(host: str, preferred_port: int, max_tries: int = 25) -> int:
    if _port_available(host, preferred_port):
        return preferred_port

    for offset in range(1, max_tries + 1):
        candidate = preferred_port + offset
        if _port_available(host, candidate):
            return candidate

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return sock.getsockname()[1]


def run_server(host: str, port: int, error_state: dict) -> None:
    try:
        config = uvicorn.Config(app, host=host, port=port, log_level="info")
        server = uvicorn.Server(config)
        server.run()
    except BaseException as e:
        error_state["error"] = f"Server failed: {type(e).__name__} - {e}"
        with open("crash_debug.txt", "w") as f:
            f.write("SERVER CRASHED:\n")
            f.write(traceback.format_exc())
        logging.error("Server failed to start.\n%s", traceback.format_exc())


def _wait_for_health(
    url: str,
    timeout_s: float,
    error_state: dict,
    interval_s: float = 0.25
) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if error_state.get("error"):
            return False
        try:
            with url_request.urlopen(url, timeout=0.5) as response:
                if 200 <= response.status < 300:
                    return True
        except (url_error.URLError, url_error.HTTPError):
            time.sleep(interval_s)
    return False


def _notify_startup_timeout(
    url: str,
    timeout_s: float,
    log_path: Path,
    error_state: dict
) -> None:
    error_line = error_state.get("error") or ""
    message = (
        "Server did not start in time.\n\n"
        f"Tried for {timeout_s:.0f} seconds.\n"
        f"You can retry by opening:\n{url}\n\n"
        f"{error_line}\n"
        f"Log file:\n{log_path}"
    )
    if sys.platform.startswith("win"):
        try:
            import ctypes

            ctypes.windll.user32.MessageBoxW(0, message, "Bubble OCR Studio", 0x10)
            return
        except Exception:
            pass
    print(message)


def main() -> None:
    host = os.environ.get("APP_HOST", "127.0.0.1")
    preferred_port = int(os.environ.get("APP_PORT", "8000"))

    log_path = _get_log_path()
    logging.basicConfig(
        filename=str(log_path),
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        filemode="a"
    )

    error_state = {}
    port = _choose_port(host, preferred_port)
    if port != preferred_port:
        logging.warning("Port %s is busy. Using %s.", preferred_port, port)

    thread = threading.Thread(target=run_server, args=(host, port, error_state), daemon=True)
    thread.start()

    health_url = f"http://{host}:{port}/api/health"
    ui_url = f"http://{host}:{port}"

    timeout_s = float(os.environ.get("APP_HEALTH_TIMEOUT", "60"))
    if _wait_for_health(health_url, timeout_s=timeout_s, error_state=error_state):
        webbrowser.open(ui_url)
    else:
        _notify_startup_timeout(ui_url, timeout_s, log_path, error_state)

    thread.join()


if __name__ == "__main__":
    main()
