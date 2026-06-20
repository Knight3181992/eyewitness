"""All runtime constants and env-var bindings for Eyewitness."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

# ── model ──────────────────────────────────────────────────────────────────────
YOLO_MODEL    = os.getenv("YOLO_MODEL",   "yolo11x.pt")
CLAUDE_MODEL  = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
MODEL_VERSION = "eyewitness-v1"

# ── CV ─────────────────────────────────────────────────────────────────────────
VEHICLE_CLASSES  = [1, 2, 3, 5, 7]
INPUT_W          = 1280
HORIZON_SEC      = 2.0
COLL_THRESH_PX   = 120
TRAIL_LEN        = 20
SAMPLE_N         = 2
MIN_TRACK_FRAMES = 5

# ── LLM cost (claude-sonnet-4-6 rates) ────────────────────────────────────────
# Update these if pricing changes; cost tracking reads from here.
COST_INPUT_PER_1K_USD  = 0.003   # $3 / M input tokens
COST_OUTPUT_PER_1K_USD = 0.015   # $15 / M output tokens

# ── LLM retry (tenacity) ──────────────────────────────────────────────────────
LLM_MAX_ATTEMPTS  = 3
LLM_WAIT_MIN_S    = 1
LLM_WAIT_MAX_S    = 10

# ── Butterbase (Option A — locked endpoints) ───────────────────────────────────
BB_APP_ID  = "app_46yxrt8czo59"
BB_API_URL = f"https://api.butterbase.ai/v1/{BB_APP_ID}"
BB_API_KEY = os.getenv("BUTTERBASE_API_KEY", "")

# Deferred post-demo: route VLM through Butterbase AI gateway.
# Flip to "true" after hardening — zero other code changes needed.
USE_BUTTERBASE_GATEWAY = os.getenv("USE_BUTTERBASE_GATEWAY", "false").lower() == "true"

# ── Anthropic ──────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# ── Gradio / Render ────────────────────────────────────────────────────────────
# Render injects PORT; fall back to GRADIO_PORT for local dev.
GRADIO_PORT  = int(os.getenv("PORT", os.getenv("GRADIO_PORT", "7860")))
GRADIO_SHARE = os.getenv("GRADIO_SHARE", "false").lower() == "true"
