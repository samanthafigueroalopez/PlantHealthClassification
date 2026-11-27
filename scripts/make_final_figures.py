r"""Build every figure for the final report and slides from the saved results.
Run from the repo root:  python scripts\make_final_figures.py
Writes reports/figures/final/*.png and reports/figures/final/numbers.md
"""
import argparse
import csv
import json
import random
import re
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

# Pink / plum palette (colorblind-checked) and neutral inks
BLUE, ORANGE = "#e85d9e", "#6a3d9a"   # BLUE = pink (main), ORANGE = plum (comparison)
INK, INK2, GRID = "#2b1d26", "#6e5a66", "#f4e4ee"
plt.rcParams.update({
    "font.size": 11, "axes.edgecolor": "#d9bccd", "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.titleweight": "bold", "axes.titlesize": 13, "axes.titlecolor": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "legend.frameon": False, "savefig.dpi": 200,
    "savefig.bbox": "tight",
})
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def read_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def read_csv(p):
    with open(p, newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def parse_report(path):
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\s*(.+?)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s*$", line)
        if m and m.group(1).strip() not in {"accuracy", "macro avg", "weighted avg"}:
            out[m.group(1).strip()] = tuple(float(m.group(i)) for i in range(2, 5)) + (int(m.group(5)),)
    return out


def bar_labels(ax, bars, fmt="{:.1f}%"):
    for b in bars:
        ax.annotate(fmt.format(b.get_height()), (b.get_x() + b.get_width() / 2, b.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, color=INK)


def load_model(weights, n_classes):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    return model.eval()


PREP = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def gradcam(model, image, class_idx):
    store = {}

    def keep(module, inp, out):
        store["a"] = out
        out.register_hook(lambda g: store.__setitem__("g", g))

    h = model.layer4.register_forward_hook(keep)
    try:
        with torch.enable_grad():
            logits = model(PREP(image).unsqueeze(0))
            model.zero_grad(set_to_none=True)
            logits[0, class_idx].backward()
    finally:
        h.remove()
        model.zero_grad(set_to_none=True)
    cam = torch.relu((store["g"].mean(dim=(2, 3), keepdim=True) * store["a"]).sum(1))[0].detach()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return np.asarray(Image.fromarray((cam.numpy() * 255).astype("uint8")).resize(image.size, Image.BILINEAR)) / 255.0


def fix(p):
    return Path(str(p).replace("\\", "/"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="reports/results")
    ap.add_argument("--plantwild", default="data/raw/PlantWild/plantwild")
    ap.add_argument("--out", default="reports/figures/final")
    ap.add_argument("--speed-images", type=int, default=50)
    args = ap.parse_args()
    R, out = Path(args.results), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    numbers = []

    rn_m = read_json(R / "resnet18_tuned" / "resnet18_tuned_metrics.json")
    cn_m = read_json(R / "custom_cnn_v2" / "custom_cnn_v2_metrics.json")
    rn_h = read_json(R / "resnet18_tuned" / "history.json")
    cn_h = read_json(R / "custom_cnn_v2" / "history.json")

    # 1) Model comparison on PlantWild
    keys = [("accuracy", "Accuracy"), ("macro_precision", "Macro precision"),
            ("macro_recall", "Macro recall"), ("macro_f1", "Macro F1")]
    x = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(8, 4.2))
    b1 = ax.bar(x - 0.19, [cn_m[k] * 100 for k, _ in keys], 0.36, color=ORANGE, label="Custom CNN v2 (from scratch)")
    b2 = ax.bar(x + 0.19, [rn_m[k] * 100 for k, _ in keys], 0.36, color=BLUE, label="ResNet-18 (ImageNet, fine-tuned)")
    bar_labels(ax, b1); bar_labels(ax, b2)
    ax.set_xticks(x, [n for _, n in keys]); ax.set_ylim(0, 100); ax.set_ylabel("PlantWild test set (%)")
    ax.set_title("Transfer learning nearly doubles performance on PlantWild")
    ax.legend(loc="upper left"); ax.grid(axis="x", visible=False)
    fig.savefig(out / "fig1_model_comparison.png"); plt.close(fig)
    numbers.append(f"| PlantWild test | Custom CNN v2 | acc {cn_m['accuracy']:.1%}, macro-F1 {cn_m['macro_f1']:.1%} |")
    numbers.append(f"| PlantWild test | ResNet-18 tuned | acc {rn_m['accuracy']:.1%}, macro-F1 {rn_m['macro_f1']:.1%} |")

    # 2) Training curves
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), sharex="row")
    for r, (name, h, m) in enumerate([("ResNet-18", rn_h, rn_m), ("Custom CNN v2", cn_h, cn_m)]):
        ep = np.arange(1, len(h["train_acc"]) + 1)
        for c, (tr, va, lab) in enumerate([("train_acc", "val_acc", "Accuracy (%)"), ("train_loss", "val_loss", "Loss")]):
            ax = axes[r, c]
            scale = 100 if c == 0 else 1
            ax.plot(ep, np.array(h[tr]) * scale, color=BLUE, lw=2, label="Train")
            ax.plot(ep, np.array(h[va]) * scale, color=ORANGE, lw=2, label="Validation")
            ax.axvline(m.get("best_epoch", ep[-1]), color=INK2, lw=1, ls="--")
            ax.set_title(f"{name}: {lab.split(' ')[0].lower()}", loc="left", fontsize=11)
            ax.set_ylabel(lab)
            if r == 1:
                ax.set_xlabel("Epoch")
    axes[0, 0].legend(loc="lower right")
    fig.suptitle("ResNet-18 overfits (train >> validation); the custom CNN underfits", fontweight="bold", color=INK)
    fig.tight_layout(); fig.savefig(out / "fig2_training_curves.png"); plt.close(fig)

    # 3) Generalization drop
    fp_pc = read_csv(R / "fieldplant_eval" / "per_class.csv")
    fp_m = read_json(R / "fieldplant_eval" / "metrics.json")
    pw_rep = parse_report(R / "resnet18_tuned" / "resnet18_tuned_classification_report.txt")
    cls = [r["class"] for r in fp_pc]
    pw_f1 = [pw_rep.get(c, (0, 0, 0, 0))[2] * 100 for c in cls]
    fp_f1 = [float(r["f1_restricted"]) * 100 for r in fp_pc]
    order = np.argsort(pw_f1)
    y = np.arange(len(cls))
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(cls) + 1.6))
    ax.barh(y + 0.2, np.array(pw_f1)[order], 0.38, color=BLUE, label="PlantWild test (web photos)")
    ax.barh(y - 0.2, np.array(fp_f1)[order], 0.38, color=ORANGE, label="FieldPlant (farm photos)")
    for yi, (a, b) in enumerate(zip(np.array(pw_f1)[order], np.array(fp_f1)[order])):
        ax.text(a + 1, yi + 0.2, f"{a:.0f}", va="center", fontsize=8, color=INK)
        ax.text(b + 1, yi - 0.2, f"{b:.0f}", va="center", fontsize=8, color=INK)
    n_of = {r["class"]: r["leaves"] for r in fp_pc}
    ax.set_yticks(y, [f"{cls[i]}  (field n={n_of[cls[i]]})" for i in order])
    ax.set_xlim(0, 105); ax.set_xlabel("F1 score (%)"); ax.grid(axis="y", visible=False)
    ax.set_title("Same model, same classes: F1 falls on real field photos")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    fig.savefig(out / "fig3_generalization_drop.png"); plt.close(fig)
    numbers.append(f"| FieldPlant (independent) | ResNet-18 | acc {fp_m['accuracy_all_89']:.1%}, macro-F1 "
                   f"{fp_m['macro_f1_all_89']:.3f} (all 89); acc {fp_m['accuracy_restricted']:.1%}, macro-F1 "
                   f"{fp_m['macro_f1_restricted']:.3f} (restricted); {fp_m['leaves_evaluated']} leaves |")

    # 4) FieldPlant confusion
    with open(R / "fieldplant_eval" / "confusion_restricted.csv", newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    cols = rows[0][1:]
    true_names = [r[0] for r in rows[1:]]
    mat = np.array([[int(v) for v in r[1:]] for r in rows[1:]], dtype=float)
    keep = [cols.index(t) for t in true_names if t in cols]
    mat = mat[:, keep]
    norm = mat / np.maximum(mat.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(8.5, 7))
    ax.imshow(norm, cmap="RdPu", vmin=0, vmax=1)
    ax.grid(False)
    for i in range(norm.shape[0]):
        for j in range(norm.shape[1]):
            ax.text(j, i, f"{norm[i, j]:.0%}", ha="center", va="center", fontsize=8,
                    color="white" if norm[i, j] > 0.5 else INK)
    ax.set_xticks(range(len(true_names)), true_names, rotation=40, ha="right")
    ax.set_yticks(range(len(true_names)), true_names)
    ax.set_xlabel("Predicted (restricted to FieldPlant classes)"); ax.set_ylabel("True class")
    ax.set_title("FieldPlant: where the field mistakes go (row %)")
    fig.savefig(out / "fig4_fieldplant_confusion.png"); plt.close(fig)

    # 5) Grad-CAM vs PlantSeg masks
    ps = read_csv(R / "plantseg_gradcam" / "plantseg_gradcam_results.csv")
    for r in ps:
        r["correct"] = str(r["correct"]).strip().lower() == "true"
        r["iou"], r["dice"], r["confidence"] = float(r["iou"]), float(r["dice"]), float(r["confidence"])
    good = [r for r in ps if r["correct"]]
    bad = [r for r in ps if not r["correct"]]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1, 1.4]})
    x = np.arange(2)
    g = a1.bar(x - 0.19, [np.mean([r["iou"] for r in good]), np.mean([r["dice"] for r in good])], 0.36,
               color=BLUE, label=f"Correct predictions (n={len(good)})")
    b = a1.bar(x + 0.19, [np.mean([r["iou"] for r in bad]), np.mean([r["dice"] for r in bad])], 0.36,
               color=ORANGE, label=f"Wrong predictions (n={len(bad)})")
    bar_labels(a1, g, "{:.2f}"); bar_labels(a1, b, "{:.2f}")
    a1.set_xticks(x, ["Mean IoU", "Mean Dice"]); a1.set_ylim(0, max(0.6, a1.get_ylim()[1]))
    a1.set_title("Overlap with the real disease mask", loc="left", fontsize=11); a1.grid(axis="x", visible=False)
    a1.legend(loc="upper left", fontsize=9)
    bins = np.linspace(0, 1, 21)
    a2.hist([r["iou"] for r in good], bins=bins, histtype="step", lw=2, color=BLUE, density=True, label="Correct")
    a2.hist([r["iou"] for r in bad], bins=bins, histtype="step", lw=2, color=ORANGE, density=True, label="Wrong")
    a2.set_xlabel("IoU between Grad-CAM (threshold 0.5) and PlantSeg mask"); a2.set_ylabel("Share of images (density)")
    a2.set_title("Distribution per image", loc="left", fontsize=11); a2.legend()
    fig.suptitle(f"Grad-CAM on {len(ps)} PlantSeg images (PlantWild test, never trained on): "
                 "correct answers look at the disease more", fontweight="bold", color=INK, fontsize=12)
    fig.tight_layout(); fig.savefig(out / "fig5_gradcam_iou.png"); plt.close(fig)
    numbers.append(f"| PlantSeg Grad-CAM | ResNet-18 | {len(ps)} images; IoU {np.mean([r['iou'] for r in ps]):.3f}, "
                   f"Dice {np.mean([r['dice'] for r in ps]):.3f}; correct IoU {np.mean([r['iou'] for r in good]):.3f} "
                   f"vs wrong {np.mean([r['iou'] for r in bad]):.3f} |")

    # 6) Grad-CAM example grid
    names = read_json(R / "resnet18_tuned" / "class_names.json")
    model = load_model(R / "resnet18_tuned" / "resnet18_tuned_best.pth", len(names))

    def pick(rows, key, n):
        chosen, used = [], set()
        for r in sorted(rows, key=key):
            if r["true_class"] in used or not fix(r["image"]).is_file() or not fix(r["mask"]).is_file():
                continue
            chosen.append(r); used.add(r["true_class"])
            if len(chosen) == n:
                break
        return chosen

    picks = pick(good, lambda r: -r["iou"], 3) + pick(bad, lambda r: -r["confidence"], 3)
    if picks:
        fig, axes = plt.subplots(len(picks), 3, figsize=(9, 2.9 * len(picks)))
        axes = np.atleast_2d(axes)
        for i, r in enumerate(picks):
            img = Image.open(fix(r["image"])).convert("RGB")
            img.thumbnail((448, 448))
            mask = np.asarray(Image.open(fix(r["mask"])).convert("L").resize(img.size, Image.NEAREST)) > 0
            cam = gradcam(model, img, int(r["predicted_index"]))
            arr = np.asarray(img) / 255.0
            over = arr.copy(); over[mask] = 0.5 * over[mask] + 0.5 * np.array([0.91, 0.36, 0.62])
            axes[i, 0].imshow(arr); axes[i, 1].imshow(over)
            axes[i, 2].imshow(arr); axes[i, 2].imshow(cam, cmap="jet", alpha=0.45)
            ok = "Correct" if r["correct"] else "Wrong"
            axes[i, 0].set_title(f"True: {r['true_class']}", fontsize=9, loc="left")
            axes[i, 1].set_title("Disease mask (PlantSeg)", fontsize=9, loc="left")
            axes[i, 2].set_title(f"{ok}: {r['predicted_class']} ({r['confidence']:.0%}) - IoU {r['iou']:.2f}",
                                 fontsize=9, loc="left", color=BLUE if r["correct"] else ORANGE)
            for a in axes[i]:
                a.axis("off")
        fig.suptitle("Where the model looks: top 3 = correct, bottom 3 = mistakes", fontweight="bold", color=INK)
        fig.tight_layout(); fig.savefig(out / "fig6_gradcam_examples.png"); plt.close(fig)

    # 7) Speed
    test = []
    tv = Path(args.plantwild) / "trainval.txt"
    if tv.is_file():
        for line in open(tv, encoding="utf-8"):
            if line.strip():
                p, _, flag = line.strip().rsplit("=", 2)
                if flag == "0":
                    test.append(Path(args.plantwild) / "images" / p)
    random.seed(0)
    sample = [p for p in random.sample(test, min(len(test), args.speed_images + 5)) if p.is_file()]
    times = []
    with torch.inference_mode():
        for k, p in enumerate(sample):
            t0 = time.perf_counter()
            with Image.open(p) as im:
                xx = PREP(im.convert("RGB")).unsqueeze(0)
            model(xx).softmax(1)
            if k >= 5:
                times.append((time.perf_counter() - t0) * 1000)
    ms = float(np.median(times)) if times else float("nan")
    rn_min, cn_min = rn_m.get("training_minutes"), cn_m.get("training_minutes")
    rn_params = sum(p.numel() for p in model.parameters())
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
    if rn_min and cn_min:
        bb = a1.bar(["Custom CNN v2", "ResNet-18"], [cn_min, rn_min], color=[ORANGE, BLUE], width=0.5)
        bar_labels(a1, bb, "{:.0f} min")
        a1.set_ylabel("Minutes (Colab T4 GPU)")
        a1.set_title("Training time", loc="left", fontsize=11)
    a1.grid(axis="x", visible=False)
    if times:
        a2.hist(times, bins=15, color=BLUE, edgecolor="white")
        a2.axvline(ms, color=INK, lw=1.5, ls="--")
        a2.text(ms, a2.get_ylim()[1] * 0.9, f"  median {ms:.0f} ms", color=INK, fontsize=9)
        a2.set_xlabel("Milliseconds per photo (laptop CPU, incl. preprocessing)")
        a2.set_title(f"Prediction time ({len(times)} PlantWild test photos)", loc="left", fontsize=11)
    fig.tight_layout(); fig.savefig(out / "fig7_speed.png"); plt.close(fig)
    if rn_min:
        numbers.append(f"| Speed | ResNet-18 | {rn_params:,} params; training {rn_min:.0f} min; prediction median {ms:.0f} ms/photo on CPU |")
    else:
        numbers.append(f"| Speed | ResNet-18 | prediction median {ms:.0f} ms/photo on CPU |")
    if cn_min:
        numbers.append(f"| Speed | Custom CNN v2 | training {cn_min:.0f} min |")

    md = ["# Final numbers (generated by scripts/make_final_figures.py)", "",
          "| Evaluation | Model | Result |", "|---|---|---|"] + numbers
    (out / "numbers.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(f"\nSaved figures to {out}")


if __name__ == "__main__":
    main()
