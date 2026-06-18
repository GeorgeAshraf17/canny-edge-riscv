import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from timing_parser import parse_timing_file, parse_speedup_file, STAGES

COLOR_SCALAR = "#4C72B0"
COLOR_RVV    = "#DD8452"
COLOR_COLD   = "#AAAAAA"


def generate(out_dir=".",
             padded_file="timing_padded.txt",
             speedup_file="speedup_target.txt"):
    sc_dict, rv_dict, hot_stages = parse_speedup_file(speedup_file)
    sc_base = parse_timing_file(padded_file, col=0)
    if sc_dict is None or rv_dict is None:
        print("[plot3] Skipping: missing data.")
        return

    baseline = sc_base if sc_base else sc_dict
    stages = [s for s in STAGES if s in baseline and s in rv_dict]
    sc_v   = [baseline[s] for s in stages]
    rv_v   = [rv_dict[s]  for s in stages]
    x      = np.arange(len(stages))
    w      = 0.35

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(x - w/2, sc_v, w, label="Scalar", color=COLOR_SCALAR)

    for i, (stage, rv_t) in enumerate(zip(stages, rv_v)):
        color = COLOR_RVV if stage in hot_stages else COLOR_COLD
        ax.bar(x[i] + w/2, rv_t, w, color=color)

    ax.bar([], [], color=COLOR_RVV,  label="RVV (vectorised)")
    ax.bar([], [], color=COLOR_COLD, label="RVV (scalar fallback)")

    max_t = max(sc_v) if sc_v else 1.0
    for i, (stage, sc_t, rv_t) in enumerate(zip(stages, sc_v, rv_v)):
        if rv_t <= 0:
            continue
        if stage in hot_stages:
            speedup = sc_t / rv_t
            label = f"x{speedup:.1f}"
            y_pos = (sc_t + rv_t) / 2
            ax.text(x[i], y_pos, label,
                    ha="center", va="center", fontsize=10, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.85))
        else:
            ax.text(x[i] + w/2, rv_t + max_t * 0.01, "scalar",
                    ha="center", va="bottom", fontsize=7, color="#666666")

    ax.set_xlabel("Stage")
    ax.set_ylabel("Time (us)")
    ax.set_title("Canny Stages: Scalar vs canny_rvv (default build, VLEN=256, 256x256)\n"
                 "Grey bars = stages that fall back to scalar in this build")
    ax.set_xticks(x)
    ax.set_xticklabels(stages, rotation=15, ha="right")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    out = os.path.join(out_dir, "before_after.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print("[plot3] Saved:", out)


if __name__ == "__main__":
    generate()
