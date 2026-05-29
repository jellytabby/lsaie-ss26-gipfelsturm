import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

WARMUP_ITERS = 2

LOG_PATTERN = re.compile(
    r"iteration\s+(\d+)/\s*\d+.*?tokens/sec/GPU:\s*(\d+)"
)


def parse_filename(name: str) -> tuple[int, str] | None:
    mbs_m = re.search(r"mbs(\d+)", name)
    if not mbs_m:
        return None
    mbs = int(mbs_m.group(1))
    offload = "nooffload" if "nooffload" in name else "offload"
    return mbs, offload


def parse_log(path: Path) -> tuple[list[int], list[int]]:
    iters, tokens = [], []
    for line in path.read_text().splitlines():
        m = LOG_PATTERN.search(line)
        if m:
            iters.append(int(m.group(1)))
            tokens.append(int(m.group(2)))
    return iters, tokens


def main():
    parser = argparse.ArgumentParser(
        description="Plot averaged tokens/sec/GPU per iteration for all runs in a log directory."
    )
    parser.add_argument("--logs-dir", default="logs", help="Directory of log files (default: logs/)")
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    if not logs_dir.is_dir():
        sys.exit(f"Logs directory not found: {logs_dir}")

    groups: dict[tuple[int, str], list[tuple[list[int], list[int]]]] = defaultdict(list)
    for log_path in sorted(logs_dir.glob("*.log")):
        key = parse_filename(log_path.name)
        if key is None:
            continue
        iters, tokens = parse_log(log_path)
        if iters:
            groups[key].append((iters, tokens))

    if not groups:
        sys.exit("No parseable log files found.")

    # ── colour palette: one colour per unique mbs value ──────────────────────
    all_mbs = sorted({mbs for mbs, _ in groups})
    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    mbs_color = {mbs: palette[i % len(palette)] for i, mbs in enumerate(all_mbs)}

    # ── figure setup ──────────────────────────────────────────────────────────
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })
    fig, ax = plt.subplots(figsize=(7, 5))

    # ── console stats header ──────────────────────────────────────────────────
    col_w = (30, 12, 12, 10, 10)
    header = (
        f"{'Config':<{col_w[0]}}"
        f"{'Mean':>{col_w[1]}}"
        f"{'Std':>{col_w[2]}}"
        f"{'Min':>{col_w[3]}}"
        f"{'Max':>{col_w[4]}}"
    )
    print(f"\n  tokens/sec/GPU  (warmup={WARMUP_ITERS} iters excluded)\n")
    print("  " + header)
    print("  " + "-" * sum(col_w))

    for (mbs, offload), runs in sorted(groups.items()):
        linestyle = "--" if offload == "offload" else "-"
        color = mbs_color[mbs]
        label = f"MBS {mbs} / {offload}"

        # align all runs to common iteration axis
        common_iters = runs[0][0]
        arrays = []
        for iters, tokens in runs:
            arr = np.full(len(common_iters), np.nan)
            it_map = {it: tok for it, tok in zip(iters, tokens)}
            for j, it in enumerate(common_iters):
                if it in it_map:
                    arr[j] = it_map[it]
            arrays.append(arr)

        matrix = np.array(arrays)          # shape: (n_runs, n_iters)
        mean = np.nanmean(matrix, axis=0)

        ax.plot(common_iters, mean, linestyle=linestyle, color=color,
                linewidth=1.8, label=label)

        # console stats (skip warmup)
        stable = mean[WARMUP_ITERS:]
        stable_all = matrix[:, WARMUP_ITERS:].flatten()
        print(
            f"  {label:<{col_w[0]}}"
            f"{stable.mean():>{col_w[1]}.0f}"
            f"{stable_all.std():>{col_w[2]}.0f}"
            f"{stable_all.min():>{col_w[3]}.0f}"
            f"{stable_all.max():>{col_w[4]}.0f}"
        )

    print()

    # ── axes formatting ───────────────────────────────────────────────────────
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Tokens / sec / GPU")
    ax.set_title("Training Throughput")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.6)
    ax.legend(framealpha=0.9, edgecolor="0.8")

    fig.tight_layout()
    out = Path("plot.pdf")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
