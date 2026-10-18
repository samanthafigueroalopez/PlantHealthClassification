
import argparse
import os
import random
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save figures without opening windows
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from tqdm import tqdm

try:
    import imagehash
except ImportError:
    imagehash = None

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_images(root: Path):
    """Return a list of (path, class_name) for every image under root."""
    rows = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith(".") and d != "__MACOSX"]
        for name in filenames:
            if Path(name).suffix.lower() in IMG_EXTS:
                p = Path(dirpath) / name
                rows.append((p, p.parent.name))
    return rows


def is_healthy(class_name: str) -> bool:
    return "healthy" in class_name.lower() or class_name.lower().endswith(" leaf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw/PlantWild", help="folder containing the extracted dataset")
    ap.add_argument("--out", default="reports/eda/plantwild", help="where to save figures and tables")
    ap.add_argument("--min-size", type=int, default=64, help="flag images smaller than this (pixels)")
    ap.add_argument("--hash-threshold", type=int, default=4, help="max hash distance to call two images near-duplicates")
    args = ap.parse_args()

    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    images = find_images(root)
    if not images:
        raise SystemExit(f"No images found under {root}. Did you extract plantwild.zip there?")
    print(f"Found {len(images):,} images in {root}")

    # ---------- Pass over every image: size, corruption, hash ----------
    records, corrupt = [], []
    for path, cls in tqdm(images, desc="Reading images"):
        try:
            with Image.open(path) as im:
                im = im.convert("RGB")
                w, h = im.size
                ph = str(imagehash.phash(im)) if imagehash else None
            records.append({"path": str(path), "class": cls, "width": w, "height": h,
                            "healthy": is_healthy(cls), "phash": ph,
                            "bytes": path.stat().st_size})
        except Exception as e:
            corrupt.append({"path": str(path), "class": cls, "error": str(e)})

    df = pd.DataFrame(records)
    df.to_csv(out / "image_index.csv", index=False)
    pd.DataFrame(corrupt, columns=["path", "class", "error"]).to_csv(out / "corrupt_images.csv", index=False)

    # ---------- 1. Class distribution ----------
    counts = df["class"].value_counts()
    counts.rename("images").to_csv(out / "class_counts.csv", header=True)

    fig, ax = plt.subplots(figsize=(14, max(6, len(counts) * 0.18)))
    colors = ["#2e8b57" if is_healthy(c) else "#c0504d" for c in counts.index]
    ax.barh(counts.index[::-1], counts.values[::-1], color=colors[::-1])
    ax.set_xlabel("Number of images")
    ax.set_title(f"PlantWild class distribution ({len(counts)} classes, green = healthy, red = diseased)")
    ax.tick_params(axis="y", labelsize=6)
    fig.tight_layout()
    fig.savefig(out / "fig1_class_distribution.png", dpi=150)
    plt.close(fig)

    # ---------- 2. Sample healthy vs. diseased ----------
    random.seed(0)
    healthy = df[df["healthy"]].sample(min(6, df["healthy"].sum()), random_state=0)
    diseased = df[~df["healthy"]].sample(min(6, (~df["healthy"]).sum()), random_state=0)
    fig, axes = plt.subplots(2, 6, figsize=(16, 6))
    for row, (label, subset) in enumerate([("Healthy", healthy), ("Diseased", diseased)]):
        for col in range(6):
            ax = axes[row, col]
            ax.axis("off")
            if col < len(subset):
                r = subset.iloc[col]
                with Image.open(r["path"]) as im:
                    ax.imshow(im.convert("RGB"))
                ax.set_title(r["class"][:28], fontsize=8)
        axes[row, 0].text(-0.15, 0.5, label, transform=axes[row, 0].transAxes,
                          rotation=90, va="center", ha="right", fontsize=12, weight="bold")
    fig.tight_layout()
    fig.savefig(out / "fig2_sample_images.png", dpi=150)
    plt.close(fig)

    # ---------- 3. Image sizes ----------
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df["width"], df["height"], s=3, alpha=0.3)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Width (px)"); ax.set_ylabel("Height (px)")
    ax.set_title("PlantWild image sizes")
    fig.tight_layout()
    fig.savefig(out / "fig3_image_sizes.png", dpi=150)
    plt.close(fig)

    small = df[(df["width"] < args.min_size) | (df["height"] < args.min_size)]
    small.to_csv(out / "small_images.csv", index=False)

    # ---------- 4. Duplicates ----------
    exact_groups, near_pairs = [], []
    if imagehash:
        by_hash = defaultdict(list)
        for r in df.itertuples():
            by_hash[r.phash].append(r.path)
        exact_groups = [g for g in by_hash.values() if len(g) > 1]

        # near-duplicates: compare hashes that share a 4-char prefix (fast approximation)
        buckets = defaultdict(list)
        for h, paths in by_hash.items():
            buckets[h[:4]].append(h)
        for hashes in buckets.values():
            for i in range(len(hashes)):
                for j in range(i + 1, len(hashes)):
                    d = imagehash.hex_to_hash(hashes[i]) - imagehash.hex_to_hash(hashes[j])
                    if d <= args.hash_threshold:
                        near_pairs.append({"path_a": by_hash[hashes[i]][0],
                                           "path_b": by_hash[hashes[j]][0], "distance": d})
        rows = [{"group": gi, "path": p} for gi, g in enumerate(exact_groups) for p in g]
        pd.DataFrame(rows, columns=["group", "path"]).to_csv(out / "duplicate_groups.csv", index=False)
        pd.DataFrame(near_pairs, columns=["path_a", "path_b", "distance"]).to_csv(out / "near_duplicates.csv", index=False)
    else:
        print("imagehash not installed: skipping duplicate check (pip install imagehash)")

    # ---------- Summary ----------
    dup_images = sum(len(g) - 1 for g in exact_groups)
    summary = [
        f"Images readable:        {len(df):,}",
        f"Corrupt/unreadable:     {len(corrupt):,}",
        f"Classes:                {len(counts)}",
        f"  Healthy classes:      {sum(is_healthy(c) for c in counts.index)}",
        f"  Diseased classes:     {sum(not is_healthy(c) for c in counts.index)}",
        f"Largest class:          {counts.index[0]} ({counts.iloc[0]})",
        f"Smallest class:         {counts.index[-1]} ({counts.iloc[-1]})",
        f"Imbalance ratio:        {counts.iloc[0] / counts.iloc[-1]:.1f}x",
        f"Median size (WxH):      {int(df['width'].median())} x {int(df['height'].median())}",
        f"Images < {args.min_size}px:          {len(small):,}",
        f"Exact duplicate extras: {dup_images:,} (in {len(exact_groups):,} groups)",
        f"Near-duplicate pairs:   {len(near_pairs):,}",
    ]
    text = "\n".join(summary)
    (out / "summary.txt").write_text(text)
    print("\n" + text)
    print(f"\nSaved figures and tables to {out}/")


if __name__ == "__main__":
    main()
