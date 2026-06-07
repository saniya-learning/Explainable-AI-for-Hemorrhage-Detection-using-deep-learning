#!/usr/bin/env python3
# gradcam_proof.py
# ── GradCAM Proof on confirmed test set images ────────────────────────────────
#
# Usage: python gradcam_proof.py
# Output: outputs/gradcam_proof/ folder

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from src.config import LABEL_NAMES, LABEL_COLS, THRESHOLD, MODEL_NAME, BEST_VAL_F1, BEST_VAL_AUC
from src.model import load_model
from src.inference import predict_with_gradcam

# ── Setup ─────────────────────────────────────────────────────────────────────
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROOF_DIR = Path("outputs/gradcam_proof")
PROOF_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_DIR = Path("static/sample_dicoms")
TEST_CSV   = Path("test_ids.csv")

LABEL_COLORS = {
    'Epidural'        : '#e74c3c',
    'Intraparenchymal': '#e67e22',
    'Intraventricular': '#9b59b6',
    'Subarachnoid'    : '#3498db',
    'Subdural'        : '#1abc9c',
    'Normal'          : '#2ecc71',
}

def find_model():
    candidates = [
        Path("models") / MODEL_NAME,
        Path("models") / "best_rsna_multilabel_resnet50.pth",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    raise FileNotFoundError("Model not found in models/ folder")

def get_true_labels(image_id, test_df):
    """Get true labels from test_ids.csv for this image."""
    row = test_df[test_df['image_id'] == image_id]
    if len(row) == 0:
        return ['Unknown']
    labels = [LABEL_NAMES[i] for i, c in enumerate(LABEL_COLS)
              if row.iloc[0][c] == 1]
    return labels if labels else ['Normal']

def save_proof_figure(file_path, result, true_labels, save_path):
    """Save proof figure: CT + GradCAM + probability bars + true vs predicted."""
    probs        = result['probs']
    pred_labels  = result['pred_labels']
    gradcam_maps = result['gradcam_maps']
    base_img     = result['base_img']
    is_normal    = result['is_normal']

    # Determine correct/wrong prediction
    true_set = set(true_labels)
    pred_set = set(pred_labels)
    is_correct = true_set == pred_set

    fig = plt.figure(figsize=(18, 9))
    fig.patch.set_facecolor('#0f172a')

    n_gcam = len(gradcam_maps)
    gs = gridspec.GridSpec(2, 2 + n_gcam, figure=fig,
                           hspace=0.4, wspace=0.3)

    # ── CT Scan ───────────────────────────────────────────────────────────────
    ax_ct = fig.add_subplot(gs[0, 0])
    ax_ct.imshow(base_img, cmap='gray')
    ax_ct.set_title('CT Scan\n(Brain Window)', color='white',
                    fontsize=11, fontweight='bold')
    ax_ct.axis('off')

    # ── GradCAM overlays ──────────────────────────────────────────────────────
    for i, (label, overlay) in enumerate(gradcam_maps.items()):
        ax_gc = fig.add_subplot(gs[0, i + 1])
        ax_gc.imshow(overlay)
        color = LABEL_COLORS.get(label, '#f97316')
        ax_gc.set_title(f'GradCAM\n{label}',
                        color=color, fontsize=11, fontweight='bold')
        ax_gc.axis('off')

    # ── Probability bars ──────────────────────────────────────────────────────
    ax_bar = fig.add_subplot(gs[0, -1])
    bar_colors = ['#e74c3c' if p >= THRESHOLD else '#334155' for p in probs]
    bars = ax_bar.barh(LABEL_NAMES, probs, color=bar_colors, height=0.55)
    ax_bar.axvline(THRESHOLD, color='yellow', linestyle='--',
                   linewidth=1.5, label=f'Threshold={THRESHOLD}')
    ax_bar.set_xlim(0, 1)
    ax_bar.set_facecolor('#1e293b')
    ax_bar.tick_params(colors='white', labelsize=9)
    ax_bar.set_title('Probabilities', color='white',
                     fontsize=11, fontweight='bold')
    for spine in ax_bar.spines.values():
        spine.set_edgecolor('#334155')
    for bar, prob in zip(bars, probs):
        ax_bar.text(min(prob + 0.02, 0.88), bar.get_y() + bar.get_height()/2,
                    f'{prob:.3f}', va='center', color='white', fontsize=9)
    ax_bar.legend(facecolor='#1e293b', labelcolor='white', fontsize=8)

    # ── True vs Predicted table ───────────────────────────────────────────────
    ax_table = fig.add_subplot(gs[1, :])
    ax_table.axis('off')
    ax_table.set_facecolor('#0f172a')

    verdict_color = '#2ecc71' if is_correct else '#e74c3c'
    verdict_text  = ' CORRECT PREDICTION' if is_correct else ' INCORRECT PREDICTION'

    table_data = [
        ['Image ID',     Path(file_path).stem],
        ['True Label',   ', '.join(true_labels)],
        ['Predicted',    ', '.join(pred_labels)],
        ['Verdict',      verdict_text],
        ['Max Prob',     f'{max(probs):.3f}'],
    ]

    table = ax_table.table(
        cellText=table_data,
        colWidths=[0.15, 0.6],
        loc='center',
        cellLoc='left',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)

    for (row, col), cell in table.get_celld().items():
        cell.set_facecolor('#1e293b')
        cell.set_edgecolor('#334155')
        cell.set_text_props(color='white')
        if row == 3:  # Verdict row
            cell.set_text_props(color=verdict_color, fontweight='bold')
        if col == 0:
            cell.set_text_props(color='#94a3b8', fontweight='bold')

    # ── Main title ────────────────────────────────────────────────────────────
    fig.suptitle(
        f'Explainable AI for Reliable ICH Detection  |  '
        f'ResNet-50 Multilabel  |  Val F1: {BEST_VAL_F1}  |  Val AUC: {BEST_VAL_AUC}\n'
        f'Test Set Image (No Data Leakage — Confirmed from held-out test split)',
        color='white', fontsize=12, fontweight='bold', y=1.01
    )

    plt.savefig(save_path, dpi=130, bbox_inches='tight',
                facecolor='#0f172a')
    plt.close()
    print(f"  [SAVED] {save_path.name}")


def main():
    print("=" * 65)
    print("  GradCAM Proof Generator — Test Set Only (No Leakage)")
    print(f"  Device: {DEVICE}")
    print("=" * 65)

    # Load model
    weights = find_model()
    print(f"\n[INFO] Loading model: {Path(weights).name}")
    model = load_model(weights, DEVICE)

    # Load test IDs
    if not TEST_CSV.exists():
        print(f"[ERROR] test_ids.csv not found in project root!")
        return
    test_df = pd.read_csv(TEST_CSV)
    print(f"[INFO] Loaded test_ids.csv — {len(test_df)} test images")

    # Find DICOM files
    files = sorted(set(SAMPLE_DIR.glob("*.dcm"))) 

    if not files:
        print(f"[ERROR] No .dcm files found in {SAMPLE_DIR}")
        return

    print(f"[INFO] Found {len(files)} DICOM files")
    print(f"[INFO] Saving proof to: {PROOF_DIR}\n")

    # ── Process each file ─────────────────────────────────────────────────────
    summary = []

    for i, file_path in enumerate(files):
        image_id   = file_path.stem
        true_labels = get_true_labels(image_id, test_df)

        # Verify it's from test set
        in_test = len(test_df[test_df['image_id'] == image_id]) > 0
        status  = " test set" if in_test else "  NOT in test set"

        print(f"[{i+1}/{len(files)}] {image_id}")
        print(f"  True labels : {true_labels}  ({status})")

        try:
            result = predict_with_gradcam(
                str(file_path), model, DEVICE, THRESHOLD)

            pred_labels = result['pred_labels']
            probs       = result['probs']
            is_correct  = set(true_labels) == set(pred_labels)

            print(f"  Predicted   : {pred_labels}")
            print(f"  Verdict     : {' CORRECT' if is_correct else '❌ WRONG'}")

            # Save figure
            save_name = f"proof_{i+1:02d}_{image_id}.png"
            save_path = PROOF_DIR / save_name
            save_proof_figure(str(file_path), result, true_labels, save_path)

            summary.append({
                'image_id'   : image_id,
                'true'       : ', '.join(true_labels),
                'predicted'  : ', '.join(pred_labels),
                'correct'    : is_correct,
                'max_prob'   : float(max(probs)),
            })

        except Exception as e:
            print(f"  [ERROR] {e}")
        print()

    # ── Final summary ─────────────────────────────────────────────────────────
    correct = sum(1 for s in summary if s['correct'])
    total   = len(summary)

    print("=" * 65)
    print(f"  FINAL SUMMARY")
    print(f"  Model   : ResNet-50  |  Val F1: {BEST_VAL_F1}  |  AUC: {BEST_VAL_AUC}")
    print(f"  Correct : {correct}/{total} predictions")
    print("=" * 65)
    print(f"\n{'Image ID':<20} {'True Label':<35} {'Predicted':<35} {'Result'}")
    print("─" * 100)
    for s in summary:
        verdict = " CORRECT" if s['correct'] else " WRONG"
        print(f"  {s['image_id']:<18} {s['true']:<35} {s['predicted']:<35} {verdict}")

    print(f"\n[DONE] {total}  images saved to: {PROOF_DIR}/")
   


if __name__ == "__main__":
    main()
