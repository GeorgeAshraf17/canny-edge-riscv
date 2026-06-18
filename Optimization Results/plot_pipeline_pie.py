import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from timing_parser import parse_timing_file, parse_speedup_file, STAGES

COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]


def generate(out_path="pipeline_pie.png",
             padded_file="timing_padded.txt",
             speedup_file="speedup_target.txt"):
    sc_dict, rv_dict, _hot = parse_speedup_file(speedup_file)
    sc_base = parse_timing_file(padded_file, col=0)
    if sc_dict is None or rv_dict is None:
        print("[pie] Skipping: missing data.")
        return

    baseline = sc_base if sc_base else sc_dict
    stages = [s for s in STAGES if s in baseline and s in rv_dict]
    sc_v = [baseline[s] for s in stages]
    rv_v = [rv_dict[s] for s in stages]
    colors = COLORS[:len(stages)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle("Pipeline Bottleneck \u2014 Time Distribution", fontsize=13)

    def explode_for(values):
        biggest = values.index(max(values))
        return [0.06 if i == biggest else 0 for i in range(len(values))]

    ax1.pie(sc_v, labels=stages, autopct="%1.1f%%", colors=colors,
            explode=explode_for(sc_v), startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 1})
    ax1.set_title("Scalar Baseline")

    ax2.pie(rv_v, labels=stages, autopct="%1.1f%%", colors=colors,
            explode=explode_for(rv_v), startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 1})
    ax2.set_title("RVV Pipeline (default build)\nscalar fallback for non-vectorised stages")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print("[pie] Saved:", out_path)


if __name__ == "__main__":
    generate()
