

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

HEALTHY = "#2a78d6"   # blue
DISEASED = "#eb6834"  # orange (blue/orange stays distinguishable for colorblind readers)
INK, MUTED, GRID = "#222222", "#666666", "#e5e5e5"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False,
})


def is_healthy(name: str) -> bool:
    n = str(name).lower()
    return "healthy" in n or n.endswith(" leaf")


def legend_handles():
    return [plt.Rectangle((0, 0), 1, 1, color=HEALTHY), plt.Rectangle((0, 0), 1, 1, color=DISEASED)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eda", default="reports/eda/plantwild")
    args = ap.parse_args()
    eda = Path(args.eda)

    counts = pd.read_csv(eda / "class_counts.csv")
    counts.columns = ["class", "images"]
    counts = counts.sort_values("images", ascending=False).reset_index(drop=True)
    counts["healthy"] = counts["class"].map(is_healthy)
    colors = counts["healthy"].map({True: HEALTHY, False: DISEASED})
    n_h, n_d = int(counts["healthy"].sum()), int((~counts["healthy"]).sum())

    # ---------- Figure 1: compact sorted bars ----------
    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    ax.bar(range(len(counts)), counts["images"], color=colors, width=0.8)
    med = counts["images"].median()
    med_line = ax.axhline(med, color=MUTED, linewidth=1, linestyle="--")
    top, bot = counts.iloc[0], counts.iloc[-1]
    ax.set_xticks([])
    ax.set_xlabel(f"{len(counts)} classes, sorted by size  |  largest: {top['class']} ({top['images']}),"
                  f"  smallest: {bot['class']} ({bot['images']})", fontsize=8)
    ax.set_ylabel("Images per class")
    ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
    ax.legend(legend_handles() + [med_line], [f"Healthy ({n_h})", f"Diseased ({n_d})", f"Median ({med:.0f})"],
              frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(eda / "fig1_class_distribution_compact.png", dpi=200)
    plt.close(fig)

    # ---------- Figure 1b: 10 largest vs 10 smallest ----------
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.2), sharex=True)
    for ax, sub, title in [(axes[0], counts.head(10), "10 largest classes"),
                           (axes[1], counts.tail(10), "10 smallest classes")]:
        sub = sub.iloc[::-1]  # largest at the top of each panel
        ax.barh(sub["class"], sub["images"], color=sub["healthy"].map({True: HEALTHY, False: DISEASED}), height=0.7)
        for y, v in enumerate(sub["images"]):
            ax.text(v + 6, y, str(v), va="center", fontsize=7, color=MUTED)
        ax.set_title(title, fontsize=9, color=INK)
        ax.tick_params(axis="y", labelsize=7)
        ax.xaxis.grid(True, color=GRID); ax.set_axisbelow(True)
    axes[0].set_xlabel("Images"); axes[1].set_xlabel("Images")
    fig.legend(legend_handles(), ["Healthy", "Diseased"], frameon=False, loc="lower center", ncol=2)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(eda / "fig1b_top_bottom_classes.png", dpi=200)
    plt.close(fig)

    # ---------- Figure 2: compact samples ----------
    index_path = eda / "image_index.csv"
    if not index_path.exists():
        print(f"{index_path} not found; skipping sample figure")
        return
    idx = pd.read_csv(index_path)
    idx["healthy"] = idx["class"].map(is_healthy)
    rows = [("Healthy", idx[idx["healthy"]]), ("Diseased", idx[~idx["healthy"]])]
    fig, axes = plt.subplots(2, 4, figsize=(6.5, 3.6))
    for r, (label, pool) in enumerate(rows):
        # one image per class, so the row shows variety
        picks = pool.groupby("class").sample(1, random_state=1).sample(min(4, pool["class"].nunique()), random_state=3)
        for c in range(4):
            ax = axes[r, c]; ax.axis("off")
            if c >= len(picks):
                continue
            row = picks.iloc[c]
            with Image.open(row["path"]) as im:
                im = im.convert("RGB")
                s = min(im.size)
                left, topy = (im.width - s) // 2, (im.height - s) // 2
                ax.imshow(im.crop((left, topy, left + s, topy + s)).resize((256, 256)))
            ax.set_title(row["class"], fontsize=7, color=INK)
        axes[r, 0].text(-0.08, 0.5, label, transform=axes[r, 0].transAxes, rotation=90,
                        va="center", ha="right", fontsize=9, weight="bold",
                        color=HEALTHY if label == "Healthy" else DISEASED)
    fig.tight_layout()
    fig.savefig(eda / "fig2_sample_images_compact.png", dpi=200)
    plt.close(fig)

    print(f"Saved compact figures to {eda}/")


if __name__ == "__main__":
    main()