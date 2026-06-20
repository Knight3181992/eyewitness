"""
Eyewitness — Gradio demo app.
Flow: upload clip → analyze → 4 keyframes + full report card → human override.
Report shows three independent panels: CV facts, avoidability physics, VLM hypothesis.
"""

import io
from typing import Optional

import gradio as gr
from PIL import Image

import config
from butterbase_client import ButterbaseClient
from schemas import AnalysisResult
from video_pipeline import analyze_clip

# ── helpers ────────────────────────────────────────────────────────────────────

def _to_pil(image_bytes: bytes) -> Optional[Image.Image]:
    if not image_bytes:
        return None
    try:
        return Image.open(io.BytesIO(image_bytes))
    except Exception:
        return None


def _format_report(result: AnalysisResult) -> str:
    lines = [f"## Eyewitness Report `{result.run_id[:8]}`"]

    if result.error:
        lines.append(f"\n> **Error:** {result.error}")

    # ── panel 1: CV facts ────────────────────────────────────────────────────
    if result.vehicle_facts:
        lines.append("\n### 📐 Vehicle Facts  *(deterministic CV)*")
        lines.append("| ID | Speed est. | Heading | TTC | Safe Stop? |")
        lines.append("|:--:|:----------:|:-------:|:---:|:----------:|")
        for f in result.vehicle_facts:
            ttc  = f"{f.ttc_ms:.0f} ms" if f.ttc_ms >= 0 else "—"
            stop = "✅" if f.had_safe_stop else "❌"
            lines.append(
                f"| #{f.vehicle_id} | {f.speed_kph_est:.1f} kph | "
                f"{f.heading_deg:.0f}° | {ttc} | {stop} |"
            )
    else:
        lines.append("\n*No vehicles tracked.*")

    # ── panel 2: avoidability ─────────────────────────────────────────────────
    if result.avoidability:
        lines.append("\n### ⚖️ Avoidability Analysis  *(Field of Safe Motion)*")
        lines.append(
            "| ID | Speed | React dist | Stop dist | Total needed | Gap | Verdict |"
        )
        lines.append("|:--:|:-----:|:----------:|:---------:|:------------:|:---:|:-------:|")
        for a in result.avoidability:
            lines.append(
                f"| #{a.vehicle_id} | {a.speed_kph} kph "
                f"| {a.react_dist_m} m | {a.stop_dist_m} m "
                f"| {a.total_needed_m} m | {a.available_gap_m} m | {a.verdict} |"
            )
        unavoidable = [a for a in result.avoidability if not a.avoidable]
        avoidable   = [a for a in result.avoidability if a.avoidable]
        if unavoidable:
            ids = ", ".join(f"#{a.vehicle_id}" for a in unavoidable)
            lines.append(f"\n> ⛔ **Physics exculpates:** {ids} — no safe stop existed.")
        if avoidable:
            ids = ", ".join(f"#{a.vehicle_id}" for a in avoidable)
            lines.append(f"> ✅ **Physics implicates:** {ids} — avoidance was physically possible.")

    # ── panel 3: VLM hypothesis ───────────────────────────────────────────────
    h = result.hypothesis
    if h:
        fault = (
            f"Vehicle #{h.fault_vehicle_id}"
            if h.fault_vehicle_id is not None
            else "Shared / indeterminate"
        )
        sev_emoji = {"minor": "🟡", "moderate": "🟠", "severe": "🔴",
                     "critical": "⛔", "unknown": "⚪"}.get(h.severity, "⚪")
        lines.append("\n### 🤖 VLM Fault Hypothesis  *(Claude corroboration)*")
        lines.append(f"**Fault:** {fault}  ")
        lines.append(f"**Reason:** {h.fault_reason}  ")
        lines.append(
            f"**Confidence:** {h.confidence * 100:.0f}%  "
            f"**Severity:** {sev_emoji} {h.severity.upper()}  "
        )
        if h.fallback_used:
            lines.append("\n> ⚠ Fallback used — VLM output could not be parsed.")
        lines.append("\n**Contributing factors:**")
        for factor in h.contributing_factors:
            lines.append(f"- {factor}")

        # ── MLOps monitoring panel ────────────────────────────────────────────
        if h.input_tokens or h.cost_usd:
            lines.append("\n### 📊 MLOps Metrics")
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            lines.append(f"| Input tokens  | {h.input_tokens:,} |")
            lines.append(f"| Output tokens | {h.output_tokens:,} |")
            lines.append(f"| Cost          | ${h.cost_usd:.5f} |")
            lines.append(f"| Fallback      | {'Yes ⚠' if h.fallback_used else 'No ✅'} |")
            lines.append(f"| FPS           | {result.fps:.0f} |")
    else:
        lines.append("\n*VLM hypothesis not available.*")

    lines.append(
        f"\n---\n*Model: `{result.model_version}` · Run: `{result.run_id}`*"
    )
    return "\n".join(lines)


# ── Gradio handlers ───────────────────────────────────────────────────────────

def on_analyze(video_path: Optional[str]):
    if not video_path:
        return None, None, None, None, "Upload a clip and click Analyze.", None

    result  = analyze_clip(video_path)
    kf_imgs = [_to_pil(kf.image_bytes) for kf in result.keyframes]
    while len(kf_imgs) < 4:
        kf_imgs.append(None)

    return (*kf_imgs[:4], _format_report(result), result)


def on_override(result: Optional[AnalysisResult], override_reason: str):
    if result is None:
        return "Run an analysis first."
    reason = override_reason.strip()
    if not reason:
        return "Enter a reason before submitting."

    bb = ButterbaseClient()
    ok = bb.append_override(
        result.run_id, result.model_version, result.hypothesis, reason
    )
    tag = result.run_id[:8]
    return (
        f"Override recorded for run `{tag}`. Original VLM row untouched."
        if ok else
        f"Saved locally for run `{tag}` (Butterbase write skipped — check key)."
    )


# ── UI ─────────────────────────────────────────────────────────────────────────

with gr.Blocks(theme=gr.themes.Monochrome(), title="Eyewitness") as demo:
    gr.Markdown("# 🎥 EYEWITNESS — AI Accident Analyst")
    gr.Markdown(
        "**Facts layer → Avoidability physics → VLM corroboration → Evidence trail.**  \n"
        "Fault is determined by physics first, AI second."
    )

    with gr.Row():
        video_input = gr.Video(label="Dashcam clip", scale=3)
        analyze_btn = gr.Button("⚡ Analyze", variant="primary", scale=1, min_width=120)

    gr.Markdown("### Keyframes")
    with gr.Row():
        frame_overview = gr.Image(label="Scene Overview")
        frame_pre      = gr.Image(label="Pre-Impact")
        frame_impact   = gr.Image(label="Impact")
        frame_post     = gr.Image(label="Post-Impact")

    report_md    = gr.Markdown("*Analysis will appear here.*")
    result_state = gr.State(value=None)

    gr.Markdown("---\n### Human Override")
    gr.Markdown(
        "Append a correction — the original VLM analysis is never overwritten."
    )
    with gr.Row():
        override_input = gr.Textbox(
            label="Override reason",
            placeholder="e.g. Vehicle #3 had right-of-way; stop sign obscured by tree.",
            scale=4,
        )
        override_btn = gr.Button("Submit Override", variant="secondary", scale=1, min_width=160)
    override_status = gr.Textbox(label="Override status", interactive=False)

    analyze_btn.click(
        fn      = on_analyze,
        inputs  = [video_input],
        outputs = [frame_overview, frame_pre, frame_impact, frame_post,
                   report_md, result_state],
    )
    override_btn.click(
        fn      = on_override,
        inputs  = [result_state, override_input],
        outputs = [override_status],
    )


if __name__ == "__main__":
    demo.launch(server_port=config.GRADIO_PORT, share=config.GRADIO_SHARE)
