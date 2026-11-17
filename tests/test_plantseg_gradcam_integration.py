import csv
import json

import torch
from PIL import Image

from analysis.plantseg_gradcam import run_analysis
from models.resnet18 import create_resnet18


def test_run_analysis_writes_outputs(tmp_path):
    plantseg_root = tmp_path / "PlantSeg"

    image_dir = (
        plantseg_root
        / "images"
        / "test"
    )

    mask_dir = (
        plantseg_root
        / "annotations"
        / "test"
    )

    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)

    image_path = (
        image_dir
        / "apple_black_rot_001.jpg"
    )

    mask_path = (
        mask_dir
        / "apple_black_rot_001.png"
    )

    Image.new(
        "RGB",
        (224, 224),
        color=(120, 160, 90)
    ).save(image_path)

    Image.new(
        "L",
        (224, 224),
        color=255
    ).save(mask_path)

    class_names = [
        "apple black rot",
        "tomato leaf"
    ]

    class_names_path = (
        tmp_path
        / "class_names.json"
    )

    class_names_path.write_text(
        json.dumps(class_names),
        encoding="utf-8"
    )

    overlap_csv = (
        tmp_path
        / "overlap.csv"
    )

    with overlap_csv.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "plantseg_path",
                "plantwild_path",
                "distance"
            ]
        )

        writer.writeheader()

    checkpoint = (
        tmp_path
        / "resnet18_test.pth"
    )

    model = create_resnet18(
        num_classes=len(class_names),
        pretrained=False
    )

    torch.save(
        model.state_dict(),
        checkpoint
    )

    output_dir = (
        tmp_path
        / "results"
    )

    summary = run_analysis(
        checkpoint=checkpoint,
        plantseg_root=plantseg_root,
        class_names_path=class_names_path,
        overlap_csv=overlap_csv,
        output_dir=output_dir,
        threshold=0.5,
        limit=1,
        save_examples=1,
        device="cpu",
    )

    assert (
        summary["evaluated_samples"]
        == 1
    )

    assert (
        output_dir
        / "plantseg_gradcam_results.csv"
    ).exists()

    assert (
        output_dir
        / "summary.json"
    ).exists()

    example_files = list(
        (
            output_dir
            / "examples"
        ).glob("*.png")
    )

    assert len(example_files) == 1
