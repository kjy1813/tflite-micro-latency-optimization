"""Regenerate the README charts from the reported benchmark figures.

The values below are the 30-image batch averages quoted in the final project
report (raw Serial logs were not retained). Run from the repository root:

    python scripts/plot_results.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ASSETS = Path(__file__).resolve().parent.parent / "assets"
ASSETS.mkdir(exist_ok=True)

BASELINE_MS = 239.60
FINAL_MS = 95.17

# (label, avg latency ms, accuracy correct out of 30, kept)
HISTORY = [
    ("Stage 0\nBaseline", 239.60, 30, True),
    ("Stage 1\nCMSIS-NN, op set,\noverhead, -O3", 136.0, 30, True),
    ("Stage 5\nAggressive\n(rejected)", 92.51, 22, False),
    ("Stage 6\nFinal valid", 95.17, 30, True),
]


def latency_comparison() -> None:
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    bars = ax.bar(["Baseline", "Final"], [BASELINE_MS, FINAL_MS],
                  color=["#9e9e9e", "#1f77b4"], width=0.55)
    for bar, value in zip(bars, [BASELINE_MS, FINAL_MS]):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 4, f"{value:.2f} ms",
                ha="center", va="bottom", fontsize=11)
    reduction = (BASELINE_MS - FINAL_MS) / BASELINE_MS * 100
    ax.set_ylabel("Average inference latency (ms)")
    ax.set_title(f"Arduino Nano 33 BLE, 30-image batch\n"
                 f"-{reduction:.2f}%  ({BASELINE_MS / FINAL_MS:.2f}x), accuracy 30/30 both")
    ax.set_ylim(0, BASELINE_MS * 1.18)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(ASSETS / "latency_comparison.png")
    plt.close(fig)


def optimization_history() -> None:
    labels = [h[0] for h in HISTORY]
    values = [h[1] for h in HISTORY]
    colors = ["#1f77b4" if h[3] else "#d62728" for h in HISTORY]
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    bars = ax.bar(labels, values, color=colors, width=0.6)
    for bar, h in zip(bars, HISTORY):
        approximate = h[0].startswith("Stage 1")
        text = f"~{h[1]:.0f} ms" if approximate else f"{h[1]:.2f} ms"
        ax.text(bar.get_x() + bar.get_width() / 2, h[1] + 4,
                f"{text}\n{h[2]}/30", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Average inference latency (ms)")
    ax.set_title("Optimization history (blue = kept, red = rejected for accuracy regression)")
    ax.set_ylim(0, BASELINE_MS * 1.2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(ASSETS / "optimization_history.png")
    plt.close(fig)


if __name__ == "__main__":
    latency_comparison()
    optimization_history()
    print(f"wrote {ASSETS / 'latency_comparison.png'}")
    print(f"wrote {ASSETS / 'optimization_history.png'}")
