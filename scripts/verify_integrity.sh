#!/usr/bin/env bash
set -euo pipefail
python -m compileall -q .
echo "OK: compileall passed"
