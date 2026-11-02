
import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw/FieldPlant")
    ap.add_argument("--out", default="reports/eda/fieldplant")
    args = ap.parse_args()

    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ann_files = sorted(root.rglob("*.json"))
    ann_files = [f for f in ann_files if "coco" in f.name.lower() or "annotation" in f.name.lower()]
    if not ann_files:
        raise SystemExit(f"No COCO annotation files found under {root}. "
                         "Did you extract the Roboflow COCO JSON zip there?")

    image_rows, box_rows, categories = [], [], {}
    for f in ann_files:
        split = f.parent.name
        coco = json.loads(f.read_text())
        for c in coco.get("categories", []):
            categories[c["id"]] = c["name"]
        for im in coco.get("images", []):
            path = f.parent / im["file_name"]
            image_rows.append({"split": split, "image_id": im["id"], "path": str(path),
                               "width": im.get("width"), "height": im.get("height"),
                               "exists": path.exists()})
        for a in coco.get("annotations", []):
            x, y, w, h = a["bbox"]
            box_rows.append({"split": split, "image_id": a["image_id"],
                             "class": categories.get(a["category_id"], str(a["category_id"])),
                             "x": x, "y": y, "w": w, "h": h})
        print(f"Read {f}")

    images = pd.DataFrame(image_rows)
    boxes = pd.DataFrame(box_rows)
    # map class names after all categories are known
    boxes = boxes.merge(images[["split", "image_id", "width", "height", "path"]], on=["split", "image_id"], how="left")
    boxes["rel_area"] = (boxes["w"] * boxes["h"]) / (boxes["width"] * boxes["height"])

    images.to_csv(out / "image_index.csv", index=False)
    boxes.to_csv(out / "box_index.csv", index=False)

    # ---------- Boxes per class ----------
    counts = boxes["class"].value_counts()
    counts.rename("leaves").to_csv(out / "class_counts.csv", header=True)
    fig, ax = plt.subplots(figsize=(10, max(5, len(counts) * 0.3)))
    ax.barh(counts.index[::-1], counts.values[::-1], color="#4f81bd")
    ax.set_xlabel("Number of leaves (bounding boxes)")
    ax.set_title(f"FieldPlant leaves per class ({len(counts)} classes)")
    fig.tight_layout()
    fig.savefig(out / "fig1_class_distribution.png", dpi=150)
    plt.close(fig)

    # ---------- Leaves per photo and box size ----------
    per_image = boxes.groupby(["split", "image_id"]).size()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(per_image, bins=range(1, int(per_image.max()) + 2), color="#4f81bd")
    axes[0].set_xlabel("Leaves per photo"); axes[0].set_ylabel("Number of photos")
    axes[0].set_title("How many labeled leaves each photo has")
    axes[1].hist(boxes["rel_area"].dropna() * 100, bins=40, color="#4f81bd")
    axes[1].set_xlabel("Box size (% of photo)"); axes[1].set_ylabel("Number of leaves")
    axes[1].set_title("How large each labeled leaf is")
    fig.tight_layout()
    fig.savefig(out / "fig2_leaves_and_box_sizes.png", dpi=150)
    plt.close(fig)

    # ---------- Samples with boxes ----------
    sample_ids = per_image.sample(min(6, len(per_image)), random_state=0).index
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, (split, image_id) in zip(axes.flat, sample_ids):
        b = boxes[(boxes["split"] == split) & (boxes["image_id"] == image_id)]
        ax.axis("off")
        try:
            with Image.open(b["path"].iloc[0]) as im:
                ax.imshow(im.convert("RGB"))
        except Exception:
            continue
        for r in b.itertuples():
            ax.add_patch(patches.Rectangle((r.x, r.y), r.w, r.h, fill=False, edgecolor="yellow", linewidth=2))
        ax.set_title(", ".join(sorted(set(b["class"])))[:50], fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig3_samples_with_boxes.png", dpi=150)
    plt.close(fig)

    # ---------- Class list for mapping to PlantWild ----------
    pd.DataFrame({"fieldplant_class": counts.index, "leaves": counts.values,
                  "plantwild_class": ""}).to_csv(out / "class_mapping_template.csv", index=False)

    # ---------- Summary ----------
    labeled_ids = set(zip(boxes["split"], boxes["image_id"]))
    no_boxes = sum((s, i) not in labeled_ids for s, i in zip(images["split"], images["image_id"]))
    unused = [n for n in categories.values() if n not in counts.index]
    lines = [
        f"Photos:                  {len(images):,}",
        f"Photos per split:        {images['split'].value_counts().to_dict()}",
        f"Missing image files:     {int((~images['exists']).sum()):,}",
        f"Photos with no boxes:    {no_boxes:,}",
        f"Leaves (boxes):          {len(boxes):,}",
        f"Classes with leaves:     {len(counts)}",
        f"Unused categories:       {unused}",
        f"Largest class:           {counts.index[0]} ({counts.iloc[0]})",
        f"Smallest class:          {counts.index[-1]} ({counts.iloc[-1]})",
        f"Leaves per photo:        median {per_image.median():.0f}, max {per_image.max()}",
        f"Median box size:         {boxes['rel_area'].median() * 100:.1f}% of photo",
        f"Median photo size (WxH): {int(images['width'].median())} x {int(images['height'].median())}",
    ]
    text = "\n".join(lines)
    (out / "summary.txt").write_text(text)
    print("\n" + text)
    print(f"\nSaved figures and tables to {out}/")


if __name__ == "__main__":
    main()