#!/bin/bash
set -euo pipefail
cd "${MCL_PROJECT_ROOT:?Set private workspace path}"
exec "${MCL_PYTHON:?Set existing project Python path}" outputs/module_mix50_v1/code/coordinate.py
