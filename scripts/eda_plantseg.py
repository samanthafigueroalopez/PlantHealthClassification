
import argparse
import os
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

try:
    import imagehash
except ImportError:
    imagehash = None

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MASK_DIR_WORDS = ("annot", "mask", "label")
SPLIT_WORDS = {"train": "train", "training": "train", "val": "val", "valid": "val",
               "validation": "val", "test": "test", "testing": "test"}
GENERIC_DIRS = {"images", "image", "imgs", "img", "jpegimages", "plantseg", "data"}


def walk_files(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith(".") and d != "__MACOSX"]
        for name in filenames:
            yield Path(dirpath) / name


def is_mask_path(p: Path, root: Path) -> bool:
    parts = [s.lower() for s in p.relative_to(root).parts[:-1]]
    return any(w in part for part in parts for w in MASK_DIR_WORDS)


def split_of(p: Path) -> str:
    for part in p.parts:
        if part.lower() in SPLIT_WORDS:
            return SPLIT_WORDS[part.lower()]
    return "unknown"


def load_metadata(root: Path, stems: set):
    """Find a CSV whose rows describe the images; return (DataFrame, filename_col) or (None, None)."""
    for csv in sorted(root.rglob("*.csv")):
        try:
            meta = pd.read_csv(csv)
        except Exception:
            continue
        for col in meta.columns:
            vals = meta[col].astype(str).map(lambda s: Path(s).stem)
            if vals.isin(stems).mean() > 0.5:
                print(f"Using metadata file {csv} (columns: {list(meta.columns)})")
                meta["_stem"] = vals
                return meta, col
    return None, None


def pick_col(columns, words):
    for w in words:
        for c in columns:
            if w in c.lower():
                return c
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw/PlantSeg")
    ap.add_argument("--out", default="reports/eda/plantseg")
    ap.add_argument("--plantwild-index", default="reports/eda/plantwild/image_index.csv")
    ap.add_argument("--hash-threshold", type=int, default=4)
    args = ap.parse_args()

    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    images, masks = [], {}
    for p in walk_files(root):
        if p.suffix.lower() not in IMG_EXTS:
            continue
        if is_mask_path(p, root):
            masks[p.stem] = p
        else:
            images.append(p)
    if not images:
        raise SystemExit(f"No images found under {root}. Did you extract plantseg.zip there?")
    print(f"Found {len(images):,} images and {len(masks):,} masks in {root}")

    # ---------- Labels ----------
    stems = {p.stem for p in images}
    meta, _ = load_metadata(root, stems)
    label_of, plant_of = {}, {}
    if meta is not None:
        cls_col = pick_col(meta.columns, ["disease", "label", "class", "category"])
        plant_col = pick_col(meta.columns, ["plant", "host", "crop"])
        for d in meta.to_dict("records"):
            if cls_col:
                label_of[d["_stem"]] = str(d[cls_col])
            if plant_col:
                plant_of[d["_stem"]] = str(d[plant_col])

    def label_for(p):
        if p.stem in label_of:
            return label_of[p.stem]
        parent = p.parent.name
        if parent.lower() not in GENERIC_DIRS and parent.lower() not in SPLIT_WORDS:
            return parent
        return "unknown"

    # ---------- Pass over images ----------
    records, corrupt = [], []
    for p in tqdm(images, desc="Reading images"):
        rec = {"path": str(p), "class": label_for(p), "plant": plant_of.get(p.stem, ""),
               "split": split_of(p), "has_mask": p.stem in masks}
        try:
            with Image.open(p) as im:
                im = im.convert("RGB")
                rec["width"], rec["height"] = im.size
                rec["phash"] = str(imagehash.phash(im)) if imagehash else None
            if rec["has_mask"]:
                with Image.open(masks[p.stem]) as m:
                    arr = np.array(m.convert("L"))
                rec["disease_fraction"] = float((arr > 0).mean())
            records.append(rec)
        except Exception as e:
            corrupt.append({"path": str(p), "error": str(e)})

    df = pd.DataFrame(records)
    df.to_csv(out / "image_index.csv", index=False)
    pd.DataFrame(corrupt, columns=["path", "error"]).to_csv(out / "corrupt_images.csv", index=False)

    # ---------- Class distribution ----------
    counts = df["class"].value_counts()
    counts.rename("images").to_csv(out / "class_counts.csv", header=True)
    fig, ax = plt.subplots(figsize=(14, max(6, len(counts) * 0.16)))
    ax.barh(counts.index[::-1], counts.values[::-1], color="#c0504d")
    ax.set_xlabel("Number of images")
    ax.set_title(f"PlantSeg class distribution ({len(counts)} classes)")
    ax.tick_params(axis="y", labelsize=5)
    fig.tight_layout()
    fig.savefig(out / "fig1_class_distribution.png", dpi=150)
    plt.close(fig)

    # ---------- Disease area ----------
    if "disease_fraction" in df and df["disease_fraction"].notna().any():
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.hist(df["disease_fraction"].dropna() * 100, bins=40, color="#c0504d")
        ax.set_xlabel("Diseased area (% of image)")
        ax.set_ylabel("Number of images")
        ax.set_title("How much of each PlantSeg image is diseased")
        fig.tight_layout()
        fig.savefig(out / "fig2_disease_area.png", dpi=150)
        plt.close(fig)

    # ---------- Samples with mask overlay ----------
    sample = df[df["has_mask"]].sample(min(6, int(df["has_mask"].sum())), random_state=0)
    if len(sample):
        fig, axes = plt.subplots(2, len(sample), figsize=(3 * len(sample), 6), squeeze=False)
        for i in range(len(sample)):
            r = sample.iloc[i]
            with Image.open(r["path"]) as im:
                img = np.array(im.convert("RGB"))
            with Image.open(masks[Path(r["path"]).stem]) as m:
                mask = np.array(m.convert("L").resize((img.shape[1], img.shape[0]))) > 0
            overlay = img.copy()
            overlay[mask] = (0.5 * overlay[mask] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)
            axes[0, i].imshow(img); axes[0, i].set_title(str(r["class"])[:26], fontsize=8)
            axes[1, i].imshow(overlay)
            axes[0, i].axis("off"); axes[1, i].axis("off")
        axes[0, 0].text(-0.1, 0.5, "Image", transform=axes[0, 0].transAxes, rotation=90, va="center", ha="right", weight="bold")
        axes[1, 0].text(-0.1, 0.5, "Disease mask", transform=axes[1, 0].transAxes, rotation=90, va="center", ha="right", weight="bold")
        fig.tight_layout()
        fig.savefig(out / "fig3_samples_with_masks.png", dpi=150)
        plt.close(fig)

    # ---------- Duplicates within PlantSeg, and overlap with PlantWild ----------
    dup_extra, overlap_exact, overlap_near = 0, 0, 0
    if imagehash:
        by_hash = df.groupby("phash")["path"].apply(list)
        dup_extra = int(sum(len(v) - 1 for v in by_hash if len(v) > 1))

        pw_path = Path(args.plantwild_index)
        if pw_path.exists():
            pw = pd.read_csv(pw_path)
            pw_hashes = defaultdict(list)
            for h, p in zip(pw["phash"].astype(str), pw["path"]):
                pw_hashes[h[:4]].append((h, p))
            rows = []
            for r in df.itertuples():
                for h, p in pw_hashes.get(r.phash[:4], []):
                    d = imagehash.hex_to_hash(r.phash) - imagehash.hex_to_hash(h)
                    if d <= args.hash_threshold:
                        rows.append({"plantseg_path": r.path, "plantwild_path": p, "distance": d})
                        break
            ov = pd.DataFrame(rows, columns=["plantseg_path", "plantwild_path", "distance"])
            ov.to_csv(out / "overlap_with_plantwild.csv", index=False)
            overlap_exact = int((ov["distance"] == 0).sum())
            overlap_near = len(ov)
        else:
            print(f"{pw_path} not found: run eda_plantwild.py first to check PlantWild overlap")

    # ---------- Summary ----------
    split_counts = df["split"].value_counts().to_dict()
    lines = [
        f"Images readable:           {len(df):,}",
        f"Corrupt/unreadable:        {len(corrupt):,}",
        f"Masks found:               {len(masks):,}",
        f"Images missing a mask:     {int((~df['has_mask']).sum()):,}",
        f"Classes:                   {len(counts)}",
        f"Plants:                    {df['plant'].replace('', np.nan).nunique() or 'n/a (no metadata)'}",
        f"Largest class:             {counts.index[0]} ({counts.iloc[0]})",
        f"Smallest class:            {counts.index[-1]} ({counts.iloc[-1]})",
        f"Imbalance ratio:           {counts.iloc[0] / counts.iloc[-1]:.1f}x",
        f"Split counts:              {split_counts}",
        f"Median size (WxH):         {int(df['width'].median())} x {int(df['height'].median())}",
    ]
    if "disease_fraction" in df and df["disease_fraction"].notna().any():
        lines.append(f"Median diseased area:      {df['disease_fraction'].median() * 100:.1f}% of image")
    lines += [
        f"Duplicate extras (within): {dup_extra:,}",
        f"Overlap with PlantWild:    {overlap_near:,} images ({overlap_exact:,} exact)",
    ]
    text = "\n".join(lines)
    (out / "summary.txt").write_text(text)
    print("\n" + text)
    print(f"\nSaved figures and tables to {out}/")


if __name__ == "__main__":
    main()
