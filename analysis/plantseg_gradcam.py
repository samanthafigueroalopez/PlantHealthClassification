import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from analysis.evaluate_plantseg import evaluate_gradcam_mask
from analysis.gradcam import GradCAM
from models.resnet18 import create_resnet18


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MASK_DIR_WORDS = ("annot", "mask", "label")


def normalize_label(text):
    text = text.lower().replace("_", " ").replace("-", " ")
    return " ".join(text.split())


def infer_true_class(image_stem, class_names):
    """
    Infer a PlantWild class from a PlantSeg filename.

    Example:
    apple_black_rot_143 -> apple black rot
    """
    stem = normalize_label(image_stem)
    matches = []

    for index, class_name in enumerate(class_names):
        normalized = normalize_label(class_name)

        if stem == normalized or stem.startswith(normalized + " "):
            matches.append(
                (len(normalized), index, class_name)
            )

    if not matches:
        return None, None

    _, index, class_name = max(matches)

    return index, class_name


def is_mask_path(path, root):
    parts = [
        part.lower()
        for part in path.relative_to(root).parts[:-1]
    ]

    return any(
        word in part
        for part in parts
        for word in MASK_DIR_WORDS
    )


def find_image_mask_pairs(root):
    """
    Search PlantSeg for image/mask pairs that share the same filename stem.
    """
    root = Path(root)

    images = []
    masks = {}

    for path in root.rglob("*"):
        if not path.is_file():
            continue

        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        if is_mask_path(path, root):
            masks[path.stem] = path
        else:
            images.append(path)

    pairs = []

    for image_path in sorted(images):
        mask_path = masks.get(image_path.stem)

        if mask_path is not None:
            pairs.append(
                (image_path, mask_path)
            )

    return pairs


def load_overlap_stems(csv_path):
    """
    Load PlantSeg image names already found in PlantWild.

    These samples are excluded to reduce data leakage.
    """
    csv_path = Path(csv_path)

    if not csv_path.exists():
        return set()

    stems = set()

    with csv_path.open(
        newline="",
        encoding="utf-8-sig"
    ) as f:

        for row in csv.DictReader(f):
            raw = row.get(
                "plantseg_path",
                ""
            )

            normalized = raw.replace(
                "\\",
                "/"
            )

            if normalized:
                stems.add(
                    Path(normalized).stem
                )

    return stems


def load_class_names(path):
    with Path(path).open(
        encoding="utf-8"
    ) as f:
        names = json.load(f)

    if not isinstance(names, list) or not names:
        raise ValueError(
            "class_names.json must contain a non-empty list."
        )

    return names


def load_resnet_checkpoint(
    checkpoint_path,
    num_classes,
    device
):
    model = create_resnet18(
        num_classes=num_classes,
        pretrained=False
    )

    state = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    if not isinstance(state, dict):
        raise ValueError(
            "Checkpoint does not contain a valid state_dict."
        )

    state = {
        key.removeprefix("module."): value
        for key, value in state.items()
    }

    model.load_state_dict(state)

    model.to(device)
    model.eval()

    return model


def eval_transform():
    return transforms.Compose([
        transforms.Resize(
            (224, 224)
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ])


def load_binary_mask(mask_path):
    with Image.open(mask_path) as image:
        array = np.array(
            image.convert("L"),
            dtype=np.uint8
        )

    binary = (
        array > 0
    ).astype(np.float32)

    return torch.from_numpy(binary)


def analyze_pair(
    model,
    gradcam,
    image_path,
    mask_path,
    class_names,
    device,
    threshold=0.5
):
    true_index, true_class = infer_true_class(
        Path(image_path).stem,
        class_names
    )

    if true_index is None:
        return None

    with Image.open(image_path) as image:
        rgb = image.convert("RGB")

        original = np.array(rgb)

        tensor = (
            eval_transform()(rgb)
            .unsqueeze(0)
            .to(device)
        )

    heatmap, predicted_index, logits = gradcam.generate(
        tensor
    )

    probabilities = torch.softmax(
        logits,
        dim=1
    )

    confidence = probabilities[
        0,
        predicted_index
    ].item()

    true_mask = load_binary_mask(
        mask_path
    ).to(
        heatmap.device
    )

    overlap = evaluate_gradcam_mask(
        heatmap,
        true_mask,
        threshold=threshold
    )

    result = {
        "image": str(image_path),
        "mask": str(mask_path),

        "true_index": true_index,
        "true_class": true_class,

        "predicted_index": predicted_index,
        "predicted_class": class_names[predicted_index],

        "correct": predicted_index == true_index,
        "confidence": confidence,

        "iou": overlap["iou"],
        "dice": overlap["dice"],
    }

    return (
        result,
        original,
        heatmap.detach().cpu().numpy(),
        overlap["true_mask"].detach().cpu().numpy()
    )


def save_example_figure(
    original,
    heatmap,
    true_mask,
    result,
    output_path
):
    heatmap_resized = np.array(
        Image.fromarray(
            (heatmap * 255).astype(np.uint8)
        ).resize(
            (
                original.shape[1],
                original.shape[0]
            )
        )
    ) / 255.0

    mask_resized = np.array(
        Image.fromarray(
            (true_mask * 255).astype(np.uint8)
        ).resize(
            (
                original.shape[1],
                original.shape[0]
            ),
            resample=Image.Resampling.NEAREST,
        )
    ) > 0

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(14, 4)
    )

    axes[0].imshow(original)
    axes[0].set_title(
        "PlantSeg image"
    )

    axes[1].imshow(original)
    axes[1].imshow(
        heatmap_resized,
        cmap="jet",
        alpha=0.45,
        vmin=0,
        vmax=1
    )
    axes[1].set_title(
        "Grad-CAM"
    )

    axes[2].imshow(
        mask_resized,
        cmap="gray"
    )
    axes[2].set_title(
        "Disease mask"
    )

    axes[3].imshow(
        heatmap_resized >= 0.5,
        cmap="gray"
    )
    axes[3].set_title(
        "Thresholded Grad-CAM"
    )

    for ax in axes:
        ax.axis("off")

    fig.suptitle(
        f"True: {result['true_class']} | "
        f"Pred: {result['predicted_class']} | "
        f"IoU: {result['iou']:.3f} | "
        f"Dice: {result['dice']:.3f}",
        fontsize=10,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight"
    )

    plt.close(fig)


def run_analysis(
    checkpoint,
    plantseg_root,
    class_names_path,
    overlap_csv,
    output_dir,
    threshold=0.5,
    limit=None,
    save_examples=12,
    device=None,
):
    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    example_dir = (
        output_dir / "examples"
    )

    example_dir.mkdir(
        exist_ok=True
    )

    if device is None:
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()

            else "mps"
            if torch.backends.mps.is_available()

            else "cpu"
        )
    else:
        device = torch.device(
            device
        )

    class_names = load_class_names(
        class_names_path
    )

    overlap_stems = load_overlap_stems(
        overlap_csv
    )

    pairs = find_image_mask_pairs(
        plantseg_root
    )

    eligible = []

    for image_path, mask_path in pairs:

        if image_path.stem in overlap_stems:
            continue

        true_index, _ = infer_true_class(
            image_path.stem,
            class_names
        )

        if true_index is not None:
            eligible.append(
                (
                    image_path,
                    mask_path
                )
            )

    if limit is not None:
        eligible = eligible[:limit]

    if not eligible:
        raise RuntimeError(
            "No eligible non-overlapping PlantSeg "
            "image/mask pairs were found."
        )

    model = load_resnet_checkpoint(
        checkpoint,
        len(class_names),
        device
    )

    gradcam = GradCAM(
        model,
        model.layer4[-1]
    )

    rows = []

    try:

        for index, (
            image_path,
            mask_path
        ) in enumerate(eligible):

            analyzed = analyze_pair(
                model,
                gradcam,
                image_path,
                mask_path,
                class_names,
                device,
                threshold,
            )

            if analyzed is None:
                continue

            (
                result,
                original,
                heatmap,
                true_mask
            ) = analyzed

            rows.append(result)

            if index < save_examples:
                save_example_figure(
                    original,
                    heatmap,
                    true_mask,
                    result,
                    example_dir
                    / f"{index:03d}_{image_path.stem}.png",
                )

    finally:
        gradcam.close()

    if not rows:
        raise RuntimeError(
            "No PlantSeg samples could be evaluated."
        )

    csv_path = (
        output_dir
        / "plantseg_gradcam_results.csv"
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            )
        )

        writer.writeheader()
        writer.writerows(rows)

    mean_iou = (
        sum(
            row["iou"]
            for row in rows
        )
        / len(rows)
    )

    mean_dice = (
        sum(
            row["dice"]
            for row in rows
        )
        / len(rows)
    )

    accuracy = (
        sum(
            row["correct"]
            for row in rows
        )
        / len(rows)
    )

    correct_rows = [
        row
        for row in rows
        if row["correct"]
    ]

    incorrect_rows = [
        row
        for row in rows
        if not row["correct"]
    ]

    def mean_or_none(
        items,
        key
    ):
        if not items:
            return None

        return (
            sum(
                item[key]
                for item in items
            )
            / len(items)
        )

    summary = {
        "evaluated_samples":
            len(rows),

        "excluded_overlap_stems":
            len(overlap_stems),

        "classification_accuracy_on_evaluated_subset":
            accuracy,

        "mean_iou":
            mean_iou,

        "mean_dice":
            mean_dice,

        "mean_iou_correct_predictions":
            mean_or_none(
                correct_rows,
                "iou"
            ),

        "mean_iou_incorrect_predictions":
            mean_or_none(
                incorrect_rows,
                "iou"
            ),

        "mean_dice_correct_predictions":
            mean_or_none(
                correct_rows,
                "dice"
            ),

        "mean_dice_incorrect_predictions":
            mean_or_none(
                incorrect_rows,
                "dice"
            ),

        "threshold":
            threshold,
    }

    with (
        output_dir
        / "summary.json"
    ).open(
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=2
        )

    return summary


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run Grad-CAM on non-overlapping "
            "PlantSeg images and compare with "
            "disease masks."
        )
    )

    parser.add_argument(
        "--checkpoint",
        required=True
    )

    parser.add_argument(
        "--plantseg-root",
        default="data/raw/PlantSeg"
    )

    parser.add_argument(
        "--class-names",
        default=(
            "reports/results/"
            "resnet18_tuned/"
            "class_names.json"
        ),
    )

    parser.add_argument(
        "--overlap-csv",
        default=(
            "reports/eda/plantseg/"
            "overlap_with_plantwild.csv"
        ),
    )

    parser.add_argument(
        "--out",
        default=(
            "reports/results/"
            "plantseg_gradcam"
        ),
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None
    )

    parser.add_argument(
        "--save-examples",
        type=int,
        default=12
    )

    parser.add_argument(
        "--device",
        default=None
    )

    args = parser.parse_args()

    summary = run_analysis(
        checkpoint=args.checkpoint,
        plantseg_root=args.plantseg_root,
        class_names_path=args.class_names,
        overlap_csv=args.overlap_csv,
        output_dir=args.out,
        threshold=args.threshold,
        limit=args.limit,
        save_examples=args.save_examples,
        device=args.device,
    )

    print(
        json.dumps(
            summary,
            indent=2
        )
    )


if __name__ == "__main__":
    main()
