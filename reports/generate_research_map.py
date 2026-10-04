"""
Generate a presentation-quality Research Roadmap / Methodology Map (4 stages)
for the ADHD EEG Classification project.
300 DPI, 16:9 ratio, white background.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

OUT_DIR = Path(r"d:\ADHD_Project\reports")
OUT_DIR.mkdir(parents=True, exist_ok=True)
DPI = 300

# ─── 4 Research Stages ────────────────────────────────────────────────
stages = [
    {
        "number": "01",
        "title": "Data Acquisition",
        "icon": "~",  # wave icon placeholder
        "color": "#2563eb",
        "bg": "#eff6ff",
        "border": "#93c5fd",
        "bullets": [
            "TDBRAIN V3.1 Dataset",
            "26-channel EEG (10-20 system)",
            "Eyes-closed resting state",
            "ADHD + Healthy Controls",
            "220 adult subjects",
        ],
    },
    {
        "number": "02",
        "title": "Preprocessing",
        "color": "#7c3aed",
        "bg": "#f5f3ff",
        "border": "#c4b5fd",
        "bullets": [
            "Band-pass filter (1-45 Hz)",
            "FastICA artifact removal",
            "Common Avg Reference (CAR)",
            "Epoching (2s windows)",
            "Bad channel interpolation",
        ],
    },
    {
        "number": "03",
        "title": "Feature Engineering",
        "color": "#059669",
        "bg": "#ecfdf5",
        "border": "#6ee7b7",
        "bullets": [
            "Power Spectral Density (Welch)",
            "Band powers (delta-gamma)",
            "Theta/Beta Ratio (TBR)",
            "Frontal Alpha Asymmetry",
            "289 spectral features total",
        ],
    },
    {
        "number": "04",
        "title": "Classification",
        "color": "#dc2626",
        "bg": "#fef2f2",
        "border": "#fca5a5",
        "bullets": [
            "2D CNN + Transformer encoder",
            "5-Fold Stratified CV",
            "Subject-level split",
            "Binary: ADHD vs Healthy",
            "AUC = 0.924 | Acc = 88.9%",
        ],
    },
]

# ─── Layout ───────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 9))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

TEXT_DARK = "#0f172a"
TEXT_MED = "#334155"
TEXT_MUTED = "#64748b"
BORDER_LIGHT = "#e2e8f0"

# ─── Title ────────────────────────────────────────────────────────────
ax.text(0.50, 0.94, "Research Methodology",
        fontsize=30, fontweight="bold", color=TEXT_DARK,
        ha="center", va="center", family="sans-serif")
ax.text(0.50, 0.895,
        "EEG-Based ADHD Classification Using Deep Learning  |  End-to-End Pipeline",
        fontsize=13, color=TEXT_MUTED, ha="center", va="center", family="sans-serif")

# Accent bar
accent = FancyBboxPatch((0.08, 0.97), 0.84, 0.008,
                         boxstyle="round,pad=0.003",
                         facecolor="#2563eb", edgecolor="none", zorder=3)
ax.add_patch(accent)

# Divider
ax.plot([0.08, 0.92], [0.865, 0.865], color=BORDER_LIGHT, linewidth=1.5, zorder=1)

# ─── Card dimensions ─────────────────────────────────────────────────
n = len(stages)
card_w = 0.19
card_h = 0.58
gap = (0.84 - n * card_w) / (n - 1)
x_start = 0.08
y_top = 0.82

for idx, stage in enumerate(stages):
    x = x_start + idx * (card_w + gap)
    y = y_top

    # ── Card background ──
    card = FancyBboxPatch(
        (x, y - card_h), card_w, card_h,
        boxstyle="round,pad=0.012",
        facecolor=stage["bg"], edgecolor=stage["border"],
        linewidth=2.2, zorder=2
    )
    ax.add_patch(card)

    # ── Number circle ──
    circle_r = 0.022
    cx = x + card_w / 2
    cy = y - 0.04
    circle = plt.Circle((cx, cy), circle_r, facecolor=stage["color"],
                         edgecolor="white", linewidth=3, zorder=4,
                         transform=ax.transData)
    ax.add_patch(circle)
    ax.text(cx, cy, stage["number"],
            fontsize=13, fontweight="bold", color="white",
            ha="center", va="center", family="sans-serif", zorder=5)

    # ── Title ──
    ax.text(cx, cy - 0.055, stage["title"],
            fontsize=15, fontweight="bold", color=stage["color"],
            ha="center", va="center", family="sans-serif", zorder=3)

    # ── Divider line under title ──
    ax.plot([x + 0.015, x + card_w - 0.015],
            [cy - 0.085, cy - 0.085],
            color=stage["border"], linewidth=1.5, zorder=3)

    # ── Bullet points ──
    bullet_y_start = cy - 0.115
    for b_idx, bullet in enumerate(stage["bullets"]):
        by = bullet_y_start - b_idx * 0.072
        # Bullet dot
        ax.plot(x + 0.022, by, marker="o", markersize=5,
                color=stage["color"], zorder=4, clip_on=False)
        # Text
        ax.text(x + 0.038, by, bullet,
                fontsize=9.5, color=TEXT_MED, va="center",
                family="sans-serif", zorder=3,
                fontweight="normal")

    # ── Arrow between cards ──
    if idx < n - 1:
        arrow_x_start = x + card_w + 0.005
        arrow_x_end = x + card_w + gap - 0.005
        arrow_y = y - card_h / 2

        ax.annotate(
            "", xy=(arrow_x_end, arrow_y), xytext=(arrow_x_start, arrow_y),
            arrowprops=dict(
                arrowstyle="-|>",
                color="#94a3b8",
                lw=2.5,
                mutation_scale=18,
                connectionstyle="arc3,rad=0"
            ),
            zorder=5
        )

# ─── Bottom bar with dataset info ─────────────────────────────────────
ax.plot([0.08, 0.92], [0.13, 0.13], color=BORDER_LIGHT, linewidth=1, zorder=1)

footer_items = [
    ("Dataset", "TDBRAIN V3.1 (Brainclinics)"),
    ("Subjects", "220 Adults (ADHD + HC)"),
    ("Channels", "26 EEG (International 10-20)"),
    ("Validation", "5-Fold Stratified CV"),
]

for i, (label, value) in enumerate(footer_items):
    fx = 0.12 + i * 0.215
    ax.text(fx, 0.085, label, fontsize=10, fontweight="bold",
            color=TEXT_MUTED, ha="center", va="center", family="sans-serif")
    ax.text(fx, 0.05, value, fontsize=10.5, fontweight="bold",
            color=TEXT_DARK, ha="center", va="center", family="sans-serif")

# ─── Save ─────────────────────────────────────────────────────────────
out = OUT_DIR / "slide6_research_map.png"
fig.savefig(out, dpi=DPI, bbox_inches="tight", pad_inches=0.2, facecolor="white")
plt.close(fig)
print(f"[OK] Saved -> {out}")
