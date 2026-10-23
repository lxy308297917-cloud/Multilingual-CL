#!/usr/bin/env bash
set -euo pipefail
cd /data0/users/lixiaoyang/Documents/Codex/2026-10-21/referenced-chatgpt-conversation-this-is-an
exec /data0/users/lixiaoyang/ssu-migration/envs/cl/bin/python outputs/module_lr077_v1/code/worker.py --uuid GPU-38dc0691-ffb0-12c2-0770-84bd5f37be39 --script pipeline.py
