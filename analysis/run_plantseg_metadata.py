"""PlantSeg Grad-CAM on PlantSeg images that are PlantWild TEST images (never trained on,
PlantWild label known, PlantSeg mask available). Train/val overlaps stay excluded.
Grad-CAM, IoU, Dice and figures are the original analysis/plantseg_gradcam.py code."""
import csv
import sys
from collections import Counter
from pathlib import Path

import analysis.plantseg_gradcam as pg


def arg_value(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def pop_arg(name, default):
    if name in sys.argv:
        i = sys.argv.index(name)
        value = sys.argv[i + 1]
        del sys.argv[i:i + 2]
        return value
    return default


def key(path):
    parts = str(path).replace("\\", "/").split("/")
    return "/".join(parts[-2:]).lower()


pw_root = Path(pop_arg("--plantwild-root", "data/raw/PlantWild/plantwild"))
overlap_csv = arg_value("--overlap-csv")
class_names = pg.load_class_names(arg_value("--class-names", "class_names.json"))

flag_of, cls_of = {}, {}
for line in open(pw_root / "trainval.txt", encoding="utf-8"):
    if line.strip():
        path, cid, flag = line.strip().rsplit("=", 2)
        flag_of[key(path)] = int(flag)
        cls_of[key(path)] = int(cid)

test_label, seen, splits = {}, set(), Counter()
with open(overlap_csv, newline="", encoding="utf-8-sig") as fh:
    for row in csv.DictReader(fh):
        stem = Path(row["plantseg_path"].replace("\\", "/")).stem
        k = key(row["plantwild_path"])
        flag = flag_of.get(k)
        splits[{0: "test", 1: "train", 2: "val"}.get(flag, "not found")] += 1
        if flag == 0:
            test_label[stem] = cls_of[k]
        else:
            seen.add(stem)
test_label = {s: c for s, c in test_label.items() if s not in seen}

print(f"PlantSeg images matching PlantWild, by PlantWild split: {dict(splits)}")
print(f"Evaluating {len(test_label)} PlantSeg images that are PlantWild test images "
      f"({len(set(test_label.values()))} classes); excluding {len(seen)} train/val matches.")
print(flush=True)


def infer_from_plantwild(image_stem, names):
    c = test_label.get(image_stem)
    return (c, names[c]) if c is not None else (None, None)


pg.infer_true_class = infer_from_plantwild
pg.load_overlap_stems = lambda path: set(seen)

if __name__ == "__main__":
    pg.main()
