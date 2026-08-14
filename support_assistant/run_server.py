#!/usr/bin/env python
"""Startup script for the Zepto Support Assistant API."""
import os
import sys
import traceback

import uvicorn

support_assistant_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(support_assistant_dir)
sys.path.insert(0, support_assistant_dir)

if __name__ == "__main__":
    try:
        print(f"Starting from: {os.getcwd()}")
        print(f"PYTHONPATH includes: {support_assistant_dir}")
        sys.stdout.flush()

        from app.main import app
        uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info", reload=False)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

