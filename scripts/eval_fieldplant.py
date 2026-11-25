r"""Evaluate the tuned ResNet-18 on FieldPlant (independent test set: never used for training).

Steps: read FieldPlant COCO boxes -> map FieldPlant classes to PlantWild classes ->
crop each leaf from its bounding box -> predict -> report accuracy / macro-F1 and
save predictions, per-class results, a confusion table and sample crops for the slides.

Run from the repo root (PlantHealthClassification) with a venv that has torch + torchvision:
    python scripts\eval_fieldplant.py --list     # show FieldPlant classes and the class mapping only
    python scripts\eval_fieldplant.py            # full evaluation (a few minutes on CPU)

Outputs go to reports/results/fieldplant_eval/.
"""
import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn
from torchvision import models, transforms

PLANTWILD_TEST_ACC = 0.636  # tuned ResNet-18 on the cleaned PlantWild test split


def words(name):
    s = name.lower().replace("maize", "corn").replace("grey", "gray").replace("mould", "mold")
    return set(re.sub(r"[^a-z]+", " ", s).split())


def map_class(name):
    """FieldPlant class name -> (PlantWild class or None, note)."""
    w = words(name)
    if "corn" in w:
        if "healthy" in w:
            return "corn leaf", ""
        if "rust" in w:
            return "corn rust", ""
        if "smut" in w:
            return "corn smut", ""
        if "gray" in w or "cercospora" in w:
            return "corn gray leaf spot", ""
        if "blight" in w:
            return "corn northern leaf blight", "assumes FieldPlant 'leaf blight' = northern leaf blight"
        return None, "no matching PlantWild corn class"
    if "tomato" in w:
        if "healthy" in w:
            return "tomato leaf", ""
        if "early" in w:
            return "tomato early blight", ""
        if "late" in w:
            return "tomato late blight", ""
        if "blight" in w:
            return None, "ambiguous: could be early or late blight"
        if "mold" in w:
            return "tomato leaf mold", ""
        if "septoria" in w:
            return "tomato septoria leaf spot", ""
        if "curl" in w:
            return "tomato yellow leaf curl virus", ""
        if "yellow" in w and "virus" in w:
            return "tomato yellow leaf curl virus", "assumes 'leaf yellow virus' = yellow leaf curl virus"
        if "mosaic" in w:
            return "tomato mosaic virus", ""
        if "bacterial" in w and "spot" in w:
            return "tomato bacterial leaf spot", ""
        return None, "no matching PlantWild tomato class"
    return None, "crop not in PlantWild"


def load_fieldplant(root):
    ann_files = [f for f in sorted(root.rglob("*.json"))
                 if "coco" in f.name.lower() or "annotation" in f.name.lower()]
    if not ann_files:
        raise SystemExit(f"No COCO annotation files under {root}")
    boxes = []
    for f in ann_files:
        coco = json.loads(f.read_text(encoding="utf-8"))
        cats = {c["id"]: c["name"] for c in coco.get("categories", [])}
        imgs = {im["id"]: f.parent / im["file_name"] for im in coco.get("images", [])}
        for a in coco.get("annotations", []):
            boxes.append({"path": imgs[a["image_id"]], "fp_class": cats[a["category_id"]],
                          "bbox": a["bbox"], "split": f.parent.name})
    return boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fieldplant", default="data/raw/FieldPlant")
    ap.add_argument("--plantwild", default="data/raw/PlantWild/plantwild")
    ap.add_argument("--weights", default="reports/results/resnet18_tuned/resnet18_tuned_best.pth")
    ap.add_argument("--out", default="reports/results/fieldplant_eval")
    ap.add_argument("--min-count", type=int, default=15, help="skip mapped classes with fewer leaves")
    ap.add_argument("--min-side", type=int, default=24, help="skip boxes smaller than this (pixels)")
    ap.add_argument("--pad", type=float, default=0.10, help="extra margin around each box")
    ap.add_argument("--limit", type=int, default=0, help="evaluate at most N leaves (quick test)")
    ap.add_argument("--list", action="store_true", help="only print classes and mapping")
    args = ap.parse_args()

    # PlantWild class list in label-id order (same as training)
    names = {}
    for line in open(Path(args.plantwild) / "classes.txt", encoding="utf-8"):
        if line.strip():
            cid, name = line.strip().split(" ", 1)
            names[int(cid)] = name
    labels = [names[i] for i in sorted(names)]
    idx = {n: i for i, n in enumerate(labels)}

    boxes = load_fieldplant(Path(args.fieldplant))
    counts = Counter(b["fp_class"] for b in boxes)
    mapping = {c: map_class(c) for c in counts}

    print(f"FieldPlant: {len(boxes)} leaves (boxes), {len(counts)} classes\n")
    print(f"{'FieldPlant class':38s} {'leaves':>6s}  -> PlantWild class / note")
    kept = {}
    for c, n in counts.most_common():
        pw, note = mapping[c]
        status = pw if pw else "EXCLUDED"
        if pw and n < args.min_count:
            status, note = "EXCLUDED", f"only {n} leaves (< {args.min_count})"
        elif pw:
            kept[c] = pw
        print(f"{c:38s} {n:6d}  -> {status}" + (f"   [{note}]" if note else ""))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "class_mapping.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["fieldplant_class", "leaves", "plantwild_class", "used", "note"])
        for c, n in counts.most_common():
            wr.writerow([c, n, mapping[c][0] or "", c in kept, mapping[c][1]])
    eval_boxes = [b for b in boxes if b["fp_class"] in kept]
    print(f"\nUsing {len(eval_boxes)} leaves from {len(kept)} FieldPlant classes "
          f"({len(set(kept.values()))} PlantWild classes).")
    if args.list:
        return

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(labels))
    model.load_state_dict(torch.load(args.weights, map_location="cpu", weights_only=True))
    model.eval()
    prep = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    random.seed(0)
    if args.limit:
        eval_boxes = random.sample(eval_boxes, min(args.limit, len(eval_boxes)))
    by_image = defaultdict(list)
    for b in eval_boxes:
        by_image[b["path"]].append(b)

    target_ids = sorted({idx[p] for p in kept.values()})  # the PlantWild classes FieldPlant can be
    rows, samples, skipped = [], [], 0
    batch, meta = [], []

    def flush():
        if not batch:
            return
        with torch.inference_mode():
            probs = model(torch.stack(batch)).softmax(dim=1)
        for p, (b, crop) in zip(probs, meta):
            top = int(p.argmax())
            restricted = target_ids[int(p[target_ids].argmax())]
            true = idx[kept[b["fp_class"]]]
            rows.append({"image": b["path"].name, "split": b["split"], "fieldplant_class": b["fp_class"],
                         "true": labels[true], "pred": labels[top], "confidence": round(float(p[top]), 4),
                         "pred_restricted": labels[restricted], "correct": top == true,
                         "correct_restricted": restricted == true})
            if len(samples) < 400:
                samples.append((crop, labels[true], labels[top], float(p[top])))
        batch.clear()
        meta.clear()

    for k, (path, bs) in enumerate(by_image.items(), 1):
        try:
            with Image.open(path) as im:
                im = im.convert("RGB")
                W, H = im.size
                for b in bs:
                    x, y, w, h = b["bbox"]
                    if min(w, h) < args.min_side:
                        skipped += 1
                        continue
                    px, py = w * args.pad, h * args.pad
                    box = (max(0, int(x - px)), max(0, int(y - py)),
                           min(W, int(x + w + px)), min(H, int(y + h + py)))
                    crop = im.crop(box)
                    batch.append(prep(crop))
                    meta.append((b, crop.copy() if len(samples) + len(meta) < 400 else None))
                    if len(batch) >= 64:
                        flush()
        except FileNotFoundError:
            skipped += len(bs)
        if k % 200 == 0:
            print(f"  {k}/{len(by_image)} photos, {len(rows)} leaves classified")
    flush()

    # ---------- metrics ----------
    y = np.array([idx[r["true"]] for r in rows])
    p_all = np.array([idx[r["pred"]] for r in rows])
    p_res = np.array([idx[r["pred_restricted"]] for r in rows])
    present = sorted(set(y.tolist()))

    def per_class(pred):
        out_rows, f1s = [], []
        for c in present:
            tp = int(((y == c) & (pred == c)).sum())
            fp = int(((y != c) & (pred == c)).sum())
            fn = int(((y == c) & (pred != c)).sum())
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
            f1s.append(f1)
            out_rows.append({"class": labels[c], "leaves": int((y == c).sum()),
                             "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)})
        return out_rows, float(np.mean(f1s))

    pc_all, f1_all = per_class(p_all)
    pc_res, f1_res = per_class(p_res)
    metrics = {
        "leaves_evaluated": len(rows), "leaves_skipped": skipped,
        "fieldplant_classes_used": len(kept), "plantwild_classes_used": len(present),
        "accuracy_all_89": round(float((y == p_all).mean()), 4), "macro_f1_all_89": round(f1_all, 4),
        "accuracy_restricted": round(float((y == p_res).mean()), 4), "macro_f1_restricted": round(f1_res, 4),
        "plantwild_test_accuracy_same_model": PLANTWILD_TEST_ACC,
        "majority_class_baseline": round(float(np.bincount(y).max() / len(y)), 4),
        "note": "restricted = prediction limited to the PlantWild classes that FieldPlant can map to",
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    with open(out / "predictions.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    with open(out / "per_class.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["class", "leaves", "precision_all89", "recall_all89", "f1_all89",
                     "precision_restricted", "recall_restricted", "f1_restricted"])
        for a, r in zip(pc_all, pc_res):
            wr.writerow([a["class"], a["leaves"], a["precision"], a["recall"], a["f1"],
                         r["precision"], r["recall"], r["f1"]])
    with open(out / "confusion_restricted.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["true \\ predicted"] + [labels[c] for c in target_ids])
        for c in present:
            wr.writerow([labels[c]] + [int(((y == c) & (p_res == t)).sum()) for t in target_ids])
    top_wrong = Counter((r["true"], r["pred"]) for r in rows if not r["correct"]).most_common(10)

    # ---------- sample grid for the slides ----------
    random.shuffle(samples)
    good = [s for s in samples if s[0] is not None and s[1] == s[2]][:6]
    bad = [s for s in samples if s[0] is not None and s[1] != s[2]][:6]
    tile, cols = 220, 6
    grid = Image.new("RGB", (cols * tile, 2 * (tile + 44)), "white")
    d = ImageDraw.Draw(grid)
    for r, group in enumerate([good, bad]):
        for c, (crop, true, pred, conf) in enumerate(group):
            x0, y0 = c * tile, r * (tile + 44)
            grid.paste(crop.resize((tile - 8, tile - 8)), (x0 + 4, y0 + 4))
            colour = (30, 120, 50) if true == pred else (180, 30, 30)
            d.text((x0 + 4, y0 + tile - 2), f"true: {true}"[:36], fill=(0, 0, 0))
            d.text((x0 + 4, y0 + tile + 14), f"pred: {pred} ({conf:.0%})"[:36], fill=colour)
    grid.save(out / "sample_predictions.jpg", quality=90)

    print("\n=== FieldPlant generalization (tuned ResNet-18) ===")
    print(f"Leaves evaluated: {len(rows)}  (skipped {skipped} tiny/missing)")
    print(f"Accuracy, all 89 classes:      {metrics['accuracy_all_89']:.1%}   macro-F1 {f1_all:.3f}")
    print(f"Accuracy, restricted classes:  {metrics['accuracy_restricted']:.1%}   macro-F1 {f1_res:.3f}")
    print(f"PlantWild test accuracy:       {PLANTWILD_TEST_ACC:.1%}")
    print(f"Majority-class baseline:       {metrics['majority_class_baseline']:.1%}")
    print("\nPer class (restricted):")
    for r in pc_res:
        print(f"  {r['class']:32s} n={r['leaves']:5d}  recall {r['recall']:.2f}  F1 {r['f1']:.2f}")
    print("\nMost common mistakes (true -> predicted, all 89 classes):")
    for (t, p), n in top_wrong:
        print(f"  {n:5d}  {t} -> {p}")
    print(f"\nSaved results to {out}")


if __name__ == "__main__":
    main()
