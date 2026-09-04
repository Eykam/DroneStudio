#!/bin/bash
cd /workspace
set -a; . /workspace/.dashboard_env; set +a
exec /workspace/venv/bin/python /workspace/dagger25.py
