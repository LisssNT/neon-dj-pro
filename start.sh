#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ ! -x .venv/bin/python ]]; then
  echo "Virtual environment not found. Run: python3.11 -m venv .venv"
  echo "Then: .venv/bin/pip install -r requirements.txt"
  exit 1
fi
exec .venv/bin/python server.py
