import csv

from PIL import Image

from analysis.plantseg_gradcam import (
    find_image_mask_pairs,
    infer_true_class,
    load_overlap_stems,
    normalize_label,
)


def test_normalize_label():
    result = normalize_label(
        "Tomato_Septoria-Leaf Spot"
    )

    assert result == (
        "tomato septoria leaf spot"
    )


def test_infer_true_class_prefers_longest_match():
    classes = [
        "apple leaf",
        "apple black rot",
        "tomato septoria leaf spot",
    ]

    index, name = infer_true_class(
        "apple_black_rot_google_001",
        classes
    )

    assert index == 1
    assert name == "apple black rot"


def test_infer_true_class_returns_none_for_unmapped_class():
    index, name = infer_true_class(
        "unknown_disease_12",
        ["apple black rot"]
    )

    assert index is None
    assert name is None


def test_find_image_mask_pairs(tmp_path):
    images = (
        tmp_path
        / "plantseg"
        / "images"
        / "test"
    )

    masks = (
        tmp_path
        / "plantseg"
        / "annotations"
        / "test"
    )

    images.mkdir(
        parents=True
    )

    masks.mkdir(
        parents=True
    )

    Image.new(
        "RGB",
        (8, 8)
    ).save(
        images
        / "apple_black_rot_1.jpg"
    )

    Image.new(
        "L",
        (8, 8)
    ).save(
        masks
        / "apple_black_rot_1.png"
    )

    Image.new(
        "RGB",
        (8, 8)
    ).save(
        images
        / "unpaired.jpg"
    )

    pairs = find_image_mask_pairs(
        tmp_path
    )

    assert len(pairs) == 1

    assert (
        pairs[0][0].stem
        == "apple_black_rot_1"
    )

    assert (
        pairs[0][1].stem
        == "apple_black_rot_1"
    )


def test_load_overlap_stems_handles_windows_paths(
    tmp_path
):
    path = (
        tmp_path
        / "overlap.csv"
    )

    with path.open(
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "plantseg_path",
                "plantwild_path",
                "distance",
            ]
        )

        writer.writeheader()

        writer.writerow({
            "plantseg_path":
                r"data\raw\PlantSeg\plantseg\images\test\apple_black_rot_143.jpg",

            "plantwild_path":
                r"data\raw\PlantWild\plantwild\images\apple black rot\143.jpg",

            "distance":
                0,
        })

    stems = load_overlap_stems(
        path
    )

    assert stems == {
        "apple_black_rot_143"
    }
