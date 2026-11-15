"""Read one JSON scientific request from stdin, write one JSON receipt, exit.

Run with the project virtual-environment Python. This entry point accepts no
credentials and starts no long-lived process. The embedding predictor uses
explicit frozen file references; model fitting reads only current known labels.
"""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from protevo.research_session import execute_request


def main():
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    try:
        request = json.loads(sys.stdin.read())
        receipt = execute_request(request)
    except Exception as exc:
        receipt = {"ok": False, "error": {"type": type(exc).__name__, "message": str(exc)}}
    sys.stdout.write(json.dumps(receipt, ensure_ascii=False, allow_nan=False) + "\n")
    return 0 if receipt["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
