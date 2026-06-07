
#!/usr/bin/env python3
# app.py — Professional ICH Detection Demo UI
# Run: python app.py → http://localhost:7860

import os
from pathlib import Path
import numpy as np
import torch
import gradio as gr
from PIL import Image

from src.config import (LABEL_NAMES, LABEL_COLS, THRESHOLD,
                        BEST_VAL_F1, BEST_VAL_AUC, BEST_EPOCH, MODEL_NAME)
from src.model import load_model
from src.inference import predict_with_gradcam

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL  = None

def get_model():
    global MODEL
    if MODEL is None:
        for p in [Path("models") / MODEL_NAME,
                  Path("models") / "best_rsna_multilabel_resnet50.pth"]:
            if p.exists():
                MODEL = load_model(str(p), DEVICE)
                return MODEL
        raise FileNotFoundError("Model not found in models/ folder")
    return MODEL

LABEL_COLORS = {
    'Epidural'        : '#FF6B6B',
    'Intraparenchymal': '#FF9F43',
    'Intraventricular': '#A29BFE',
    'Subarachnoid'    : '#74B9FF',
    'Subdural'        : '#55EFC4',
}

LABEL_DESCRIPTIONS = {
    'Epidural'        : 'Blood accumulates between skull and dura mater. Often arterial — rapid progression.',
    'Intraparenchymal': 'Bleeding within brain tissue itself. Associated with high mortality risk.',
    'Intraventricular': 'Blood enters brain ventricles. Can obstruct CSF flow causing hydrocephalus.',
    'Subarachnoid'    : 'Blood in subarachnoid space. Classic presentation: sudden severe headache.',
    'Subdural'        : 'Blood between dura and brain surface. Common in elderly and trauma patients.',
}

def prob_bar_html(prob, label, color):
    pct      = int(prob * 100)
    detected = prob >= THRESHOLD
    status   = f'<span style="color:{color};font-weight:700;font-size:11px">● DETECTED</span>' if detected else '<span style="color:#4a5568;font-size:11px">○ Clear</span>'
    return f"""
    <div style="margin:8px 0">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px">
        <span style="font-size:12px;color:#e2e8f0;font-weight:500">{label}</span>
        <div style="display:flex;align-items:center;gap:12px">
          <span style="font-size:12px;color:#94a3b8;font-family:monospace">{prob:.3f}</span>
          {status}
        </div>
      </div>
      <div style="background:#1e293b;border-radius:4px;height:6px;overflow:hidden">
        <div style="background:{'linear-gradient(90deg,'+color+','+color+'99)' if detected else '#334155'};width:{pct}%;height:100%;border-radius:4px"></div>
      </div>
    </div>
    """

def run_inference(uploaded_file):
    if uploaded_file is None:
        return None, None, "<p style='color:#94a3b8;text-align:center;padding:40px'>Upload a DICOM or PNG file to begin analysis</p>"
    try:
        model  = get_model()
        result = predict_with_gradcam(uploaded_file, model, DEVICE, THRESHOLD)
    except Exception as e:
        return None, None, f"<p style='color:#fc8181'>Error: {e}</p>"

    probs        = result['probs']
    pred_labels  = result['pred_labels']
    gradcam_maps = result['gradcam_maps']
    base_img     = result['base_img']
    is_normal    = result['is_normal']

    ct_display = Image.fromarray(base_img)
    gcam_img   = Image.fromarray(list(gradcam_maps.values())[0]) if gradcam_maps else ct_display
    gcam_label = list(gradcam_maps.keys())[0] if gradcam_maps else "N/A"

    if is_normal:
        verdict_html = """
        <div style="background:linear-gradient(135deg,#064e3b,#065f46);border:1px solid #10b981;border-radius:12px;padding:20px 24px;margin-bottom:16px">
          <div style="display:flex;align-items:center;gap:12px">
            <div style="width:12px;height:12px;border-radius:50%;background:#10b981;box-shadow:0 0 8px #10b981"></div>
            <span style="font-size:18px;font-weight:700;color:#10b981;letter-spacing:0.5px">NO HEMORRHAGE DETECTED</span>
          </div>
          <p style="color:#6ee7b7;margin:8px 0 0;font-size:13px">All five subtype probabilities are below the detection threshold of 0.5</p>
        </div>"""
    else:
        labels_html = "".join([
            f'<span style="background:{LABEL_COLORS.get(l,"#FF6B6B")}22;color:{LABEL_COLORS.get(l,"#FF6B6B")};border:1px solid {LABEL_COLORS.get(l,"#FF6B6B")}55;padding:3px 12px;border-radius:20px;font-size:12px;font-weight:600;margin-right:6px">{l}</span>'
            for l in pred_labels
        ])
        desc_html = "".join([
            f'<div style="margin:6px 0;padding:10px 14px;background:#1e293b;border-left:3px solid {LABEL_COLORS.get(l,"#FF6B6B")};border-radius:0 6px 6px 0"><span style="color:{LABEL_COLORS.get(l,"#FF6B6B")};font-weight:600;font-size:12px">{l}:</span><span style="color:#94a3b8;font-size:12px;margin-left:8px">{LABEL_DESCRIPTIONS.get(l,"")}</span></div>'
            for l in pred_labels
        ])
        verdict_html = f"""
        <div style="background:linear-gradient(135deg,#450a0a,#7f1d1d);border:1px solid #ef4444;border-radius:12px;padding:20px 24px;margin-bottom:16px">
          <div style="display:flex;align-items:center;gap:12px;margin-bottom:10px">
            <div style="width:12px;height:12px;border-radius:50%;background:#ef4444;box-shadow:0 0 8px #ef4444"></div>
            <span style="font-size:18px;font-weight:700;color:#ef4444;letter-spacing:0.5px">HEMORRHAGE DETECTED</span>
          </div>
          <div style="margin-bottom:12px">{labels_html}</div>
          {desc_html}
        </div>"""

    bars_html = "".join([
        prob_bar_html(p, n, LABEL_COLORS.get(n, '#94a3b8'))
        for n, p in zip(LABEL_NAMES, probs)
    ])

    result_html = f"""
    <div style="font-family:system-ui,sans-serif;color:#e2e8f0">
      {verdict_html}
      <div style="background:#0f172a;border:1px solid #1e293b;border-radius:10px;padding:16px;margin-bottom:14px">
        <p style="font-size:11px;font-weight:700;color:#475569;letter-spacing:1px;margin:0 0 12px;text-transform:uppercase">Per-Label Probabilities</p>
        {bars_html}
        <p style="font-size:11px;color:#334155;margin:10px 0 0">Detection threshold: 0.5 — sigmoid probabilities above this are flagged as detected</p>
      </div>
      <div style="background:#0f172a;border:1px solid #1e293b;border-radius:10px;padding:16px">
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:10px">
          <span style="font-size:15px">🔥</span>
          <span style="font-size:13px;font-weight:700;color:#f8fafc">GradCAM Explanation</span>
          <span style="background:#1e293b;color:#64748b;font-size:10px;padding:2px 8px;border-radius:10px;font-family:monospace">layer4[-1].conv3</span>
        </div>
        <p style="color:#64748b;font-size:12px;margin:0 0 8px">Attention map for: <span style="color:{LABEL_COLORS.get(gcam_label,'#f97316')};font-weight:600">{gcam_label}</span> — brighter regions indicate stronger model focus</p>
        <div style="display:flex;gap:8px;font-size:11px">
          <span style="background:#ef444422;color:#ef4444;padding:2px 8px;border-radius:4px">Red = High attention</span>
          <span style="background:#3b82f622;color:#3b82f6;padding:2px 8px;border-radius:4px">Blue = Low attention</span>
        </div>
      </div>
    </div>
    """
    return ct_display, gcam_img, result_html


CSS = """
body, .gradio-container { background:#020817 !important; }
.gradio-container { max-width:1200px !important; margin:0 auto !important; }
button.primary {
    background: linear-gradient(135deg,#3b82f6,#6366f1) !important;
    border: none !important; font-weight:700 !important;
    font-size:15px !important; height:52px !important;
    border-radius:10px !important;
}
"""

# ── Gradio 6.0: theme & css go to launch(), NOT gr.Blocks() ──────────────────
with gr.Blocks() as demo:

    gr.HTML(f"""
    <div style="text-align:center;padding:36px 20px 20px;border-bottom:1px solid #0f172a;margin-bottom:24px">
      <h1 style="font-size:2em;font-weight:800;color:#f8fafc;margin:0 0 8px;letter-spacing:-0.5px">
        🧠 Explainable AI for Reliable ICH Detection
      </h1>
      <p style="color:#475569;font-size:12px;font-family:monospace;letter-spacing:1px;margin:0">
        RESNET-50 MULTILABEL &nbsp;·&nbsp; RSNA 2019 &nbsp;·&nbsp; TRIPLE-WINDOW CT &nbsp;·&nbsp; GRADCAM XAI
      </p>
    </div>
    """)

    with gr.Row():
        gr.HTML(f'<div style="background:#0f172a;border:1px solid #1e293b;border-radius:10px;padding:16px;text-align:center"><div style="font-size:1.6em;font-weight:800;color:#f8fafc">{BEST_VAL_F1}</div><div style="font-size:10px;color:#475569;letter-spacing:1px;text-transform:uppercase;margin-top:4px">Val Macro-F1</div></div>')
        gr.HTML(f'<div style="background:#0f172a;border:1px solid #1e293b;border-radius:10px;padding:16px;text-align:center"><div style="font-size:1.6em;font-weight:800;color:#f8fafc">{BEST_VAL_AUC}</div><div style="font-size:10px;color:#475569;letter-spacing:1px;text-transform:uppercase;margin-top:4px">Val ROC-AUC</div></div>')
        gr.HTML(f'<div style="background:#0f172a;border:1px solid #1e293b;border-radius:10px;padding:16px;text-align:center"><div style="font-size:1.6em;font-weight:800;color:#f8fafc">{BEST_EPOCH}</div><div style="font-size:10px;color:#475569;letter-spacing:1px;text-transform:uppercase;margin-top:4px">Best Epoch</div></div>')
        gr.HTML( '<div style="background:#0f172a;border:1px solid #1e293b;border-radius:10px;padding:16px;text-align:center"><div style="font-size:1.6em;font-weight:800;color:#f8fafc">ResNet-50</div><div style="font-size:10px;color:#475569;letter-spacing:1px;text-transform:uppercase;margin-top:4px">Architecture</div></div>')
        gr.HTML( '<div style="background:#0f172a;border:1px solid #1e293b;border-radius:10px;padding:16px;text-align:center"><div style="font-size:1.6em;font-weight:800;color:#f8fafc">5</div><div style="font-size:10px;color:#475569;letter-spacing:1px;text-transform:uppercase;margin-top:4px">ICH Subtypes</div></div>')
        gr.HTML( '<div style="background:#0f172a;border:1px solid #1e293b;border-radius:10px;padding:16px;text-align:center"><div style="font-size:1.6em;font-weight:800;color:#f8fafc">53.9K</div><div style="font-size:10px;color:#475569;letter-spacing:1px;text-transform:uppercase;margin-top:4px">Training Images</div></div>')

    gr.HTML("<div style='height:20px'></div>")

    with gr.Row(equal_height=False):
        with gr.Column(scale=1):
            gr.HTML('<p style="font-size:11px;font-weight:700;color:#475569;letter-spacing:1px;text-transform:uppercase;margin-bottom:8px">Upload CT Scan</p>')
            file_input = gr.File(label="DICOM (.dcm) or PNG", file_types=[".dcm",".dicom",".png",".jpg"])
            run_btn    = gr.Button("Analyse Scan →", variant="primary")
            gr.HTML("""
            <div style="margin-top:16px;padding:16px;background:#0f172a;border:1px solid #1e293b;border-radius:10px">
              <p style="font-size:11px;font-weight:700;color:#475569;letter-spacing:1px;text-transform:uppercase;margin:0 0 10px">Detectable Subtypes</p>
              <div style="display:flex;flex-direction:column;gap:7px">
                <div style="display:flex;align-items:center;gap:8px"><div style="width:8px;height:8px;border-radius:50%;background:#FF6B6B"></div><span style="font-size:12px;color:#94a3b8">Epidural</span></div>
                <div style="display:flex;align-items:center;gap:8px"><div style="width:8px;height:8px;border-radius:50%;background:#FF9F43"></div><span style="font-size:12px;color:#94a3b8">Intraparenchymal</span></div>
                <div style="display:flex;align-items:center;gap:8px"><div style="width:8px;height:8px;border-radius:50%;background:#A29BFE"></div><span style="font-size:12px;color:#94a3b8">Intraventricular</span></div>
                <div style="display:flex;align-items:center;gap:8px"><div style="width:8px;height:8px;border-radius:50%;background:#74B9FF"></div><span style="font-size:12px;color:#94a3b8">Subarachnoid</span></div>
                <div style="display:flex;align-items:center;gap:8px"><div style="width:8px;height:8px;border-radius:50%;background:#55EFC4"></div><span style="font-size:12px;color:#94a3b8">Subdural</span></div>
              </div>
            </div>
            <div style="margin-top:12px;padding:16px;background:#0f172a;border:1px solid #1e293b;border-radius:10px">
              <p style="font-size:11px;font-weight:700;color:#475569;letter-spacing:1px;text-transform:uppercase;margin:0 0 10px">Pipeline</p>
              <div style="display:flex;flex-direction:column;gap:7px">
                <div style="display:flex;gap:8px"><span style="color:#3b82f6;font-weight:700;font-size:11px;min-width:14px">1</span><span style="color:#64748b;font-size:11px">DICOM → HU conversion</span></div>
                <div style="display:flex;gap:8px"><span style="color:#3b82f6;font-weight:700;font-size:11px;min-width:14px">2</span><span style="color:#64748b;font-size:11px">Triple-window stacking</span></div>
                <div style="display:flex;gap:8px"><span style="color:#3b82f6;font-weight:700;font-size:11px;min-width:14px">3</span><span style="color:#64748b;font-size:11px">ResNet-50 → 5 sigmoid outputs</span></div>
                <div style="display:flex;gap:8px"><span style="color:#3b82f6;font-weight:700;font-size:11px;min-width:14px">4</span><span style="color:#64748b;font-size:11px">GradCAM on layer4[-1].conv3</span></div>
              </div>
            </div>
            """)

        with gr.Column(scale=2):
            with gr.Row():
                # ── Gradio 6.0: show_download_button removed from gr.Image ──
                ct_out   = gr.Image(label="CT Scan — Brain Window", type="pil")
                gcam_out = gr.Image(label="GradCAM — Model Attention Map", type="pil")
            result_html = gr.HTML("<p style='color:#334155;text-align:center;padding:40px;font-size:13px'>Upload a CT scan and click Analyse →</p>")

    run_btn.click(fn=run_inference, inputs=[file_input], outputs=[ct_out, gcam_out, result_html])

    gr.HTML("""
    <div style="border-top:1px solid #0f172a;margin-top:32px;padding:20px;text-align:center">
      <p style="font-size:11px;color:#1e293b;font-family:monospace;margin:0">
        EXPLAINABLE AI FOR RELIABLE ICH DETECTION · RESNET-50 · RSNA 2019 · VAL F1: 0.729 · AUC: 0.929
      </p>
    </div>
    """)

if __name__ == "__main__":
    print(f"\nDevice: {DEVICE}")
    print("Starting at http://localhost:7860\n")
    # ── Gradio 6.0: theme & css passed here, not in gr.Blocks() ──────────────
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=gr.themes.Base(),
        css=CSS,
    )
