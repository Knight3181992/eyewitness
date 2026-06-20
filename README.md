# Eyewitness — AI Accident Analyst

Beta Hacks 2026 · 90-second demo · isolated from PlanGEN

## What it does

Upload a dashcam clip → YOLO11x tracks vehicles → extracts per-vehicle speed,
heading, TTC, and braking facts → Claude analyses 4 keyframes → fault hypothesis
in structured JSON → all evidence appended to Butterbase → human can override
without overwriting the original VLM analysis.

## Stack

| Layer | Tech |
|-------|------|
| Tracking | YOLO11x via ultralytics |
| CV facts | OpenCV + NumPy (two-pass, memory-safe) |
| VLM | Claude claude-sonnet-4-6 via Anthropic SDK |
| Persistence | Butterbase REST (append-only, 4 tables) |
| UI | Gradio Blocks |

## Setup

```bash
pip install -r requirements.txt
```

### Required env vars

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export BUTTERBASE_API_KEY=<your Butterbase service key>   # get from butterbase.ai dashboard
```

### Optional env vars

| Var | Default | Purpose |
|-----|---------|---------|
| `YOLO_MODEL` | `yolo11x.pt` | Switch to `yolo11n.pt` for faster CPU inference |
| `CLAUDE_MODEL` | `claude-sonnet-4-6` | Override VLM model |
| `GRADIO_PORT` | `7860` | UI port |
| `GRADIO_SHARE` | `false` | Set `true` for a public Gradio link |
| `USE_BUTTERBASE_GATEWAY` | `false` | **Deferred post-demo.** Route VLM calls through Butterbase AI gateway instead of direct Anthropic API |

## Butterbase backend

App: `eyewitness` (`app_46yxrt8czo59`)  
API: `https://api.butterbase.ai/v1/app_46yxrt8czo59`

Tables (all append-only):
- `claims` — one row per analysis run
- `facts` — one row per tracked vehicle
- `frames` — 4 keyframe rows per run (base64 JPEG inline)
- `fault_analyses` — VLM hypothesis + human overrides as separate rows

## Demo clip

Download a dashcam near-miss video and name it `clip.mp4`:

```bash
yt-dlp -o clip.mp4 "https://www.youtube.com/watch?v=<near-miss-video-id>"
```

## Run

```bash
python app.py
# open http://localhost:7860
# upload clip.mp4 → Analyze → review report → submit override
```

## Architecture notes

- **Two-pass CV**: pass-1 tracks without storing frames (memory-safe for long clips);
  pass-2 seeks to the 4 keyframe positions.
- **VLM fallback**: if Claude output cannot be parsed or is missing required fields,
  `FALLBACK_HYPOTHESIS` is returned and `fallback_used=True` is recorded.
- **Append-only evidence trail**: human overrides write a new `fault_analyses` row
  with `override_reason` set; the original VLM row is never modified.
- **Butterbase writes are async**: a daemon thread handles persistence so the UI
  returns immediately.
- **PlanGEN untouched**: this package lives entirely under `eyewitness/` with no
  shared imports or side-effects on other projects in this workspace.
