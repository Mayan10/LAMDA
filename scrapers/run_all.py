"""
LAMDA Scraper Launcher — run_all.py
Starts all 6 scraper agents as subprocesses.
Usage: python scrapers/run_all.py
Ctrl+C gracefully terminates all.
"""

import sys
import os
import signal
import subprocess
import time

# Scraper definitions: (script_name, port)
SCRAPERS = [
    ("gscpi_agent.py",     8001),
    ("news_agent.py",      8002),
    ("political_agent.py", 8003),
    ("trade_agent.py",     8004),
    ("weather_agent.py",   8005),
    ("reporter_agent.py",  8006),
]


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    python_exe = sys.executable

    processes = []
    print("=" * 60)
    print("  LAMDA — Starting All Scraper Agents")
    print("=" * 60)

    for script_name, port in SCRAPERS:
        script_path = os.path.join(script_dir, script_name)
        if not os.path.exists(script_path):
            print(f"  [WARN] {script_name} not found — skipping")
            continue

        print(f"  Starting {script_name:25s} on port {port}...")
        proc = subprocess.Popen(
            [python_exe, script_path],
            cwd=script_dir,
            # Forward stdout/stderr so logs are visible
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        processes.append((script_name, port, proc))
        time.sleep(0.5)  # slight stagger to avoid port conflicts

    print("=" * 60)
    print(f"  {len(processes)} scrapers launched. Press Ctrl+C to stop all.")
    print("=" * 60)

    # Wait for Ctrl+C
    try:
        while True:
            # Check if any process has died
            for script_name, port, proc in processes:
                ret = proc.poll()
                if ret is not None:
                    print(f"  [WARN] {script_name} (port {port}) exited with code {ret}")
            time.sleep(5)
    except KeyboardInterrupt:
        print("\n" + "=" * 60)
        print("  Shutting down all scrapers...")
        print("=" * 60)

    # Terminate all
    for script_name, port, proc in processes:
        if proc.poll() is None:
            print(f"  Stopping {script_name} (port {port})...")
            proc.terminate()

    # Wait for graceful shutdown
    for script_name, port, proc in processes:
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            print(f"  Force killing {script_name} (port {port})")
            proc.kill()

    print("  All scrapers stopped.")


if __name__ == "__main__":
    main()
