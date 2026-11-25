r"""Sanity-check the LeafLens model outside Streamlit.

Run inside plant_classification_ui with the venv active:
  python check_model.py                    # class-order check + accuracy on 300 PlantWild test images
  python check_model.py path\to\leaf.jpg   # top-5 predictions for one or more images
"""
import json
import random
import sys
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

HERE = Path(__file__).resolve().parent
PW = HERE.parent / "data" / "raw" / "PlantWild" / "plantwild"

labels = json.loads((HERE / "class_names.json").read_text(encoding="utf-8"))
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(labels))
model.load_state_dict(torch.load(HERE / "resnet18_tuned_best.pth", map_location="cpu", weights_only=True))
model.eval()

# Same preprocessing as Final_app.py and the notebook's eval_transform
prep = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def topk(path, k=5):
    with Image.open(path) as im:
        x = prep(im.convert("RGB")).unsqueeze(0)
    with torch.inference_mode():
        probs = model(x).softmax(dim=1)[0]
    scores, idx = probs.topk(k)
    return [(labels[i], float(s)) for i, s in zip(idx.tolist(), scores.tolist())]


if len(sys.argv) > 1:
    for p in sys.argv[1:]:
        print(p)
        for name, conf in topk(p):
            print(f"  {conf:6.1%}  {name}")
    sys.exit()

# 1) class_names.json must be in the same order as classes.txt (the training label ids)
names = {}
for line in open(PW / "classes.txt", encoding="utf-8"):
    if line.strip():
        cid, name = line.strip().split(" ", 1)
        names[int(cid)] = name
assert [names[i] for i in sorted(names)] == labels, "class_names.json order differs from classes.txt!"
print(f"Class order OK ({len(labels)} classes)")

# 2) accuracy on a random sample of the official PlantWild test split (flag 0)
test = []
for line in open(PW / "trainval.txt", encoding="utf-8"):
    path, cid, flag = line.strip().rsplit("=", 2)
    if flag == "0":
        test.append((PW / "images" / path, int(cid)))
random.seed(0)
sample = random.sample(test, min(300, len(test)))
correct = 0
for i, (p, cid) in enumerate(sample, 1):
    correct += topk(p, 1)[0][0] == labels[cid]
    if i % 50 == 0:
        print(f"  {i}/{len(sample)} checked")
print(f"Accuracy on {len(sample)} PlantWild test images: {correct / len(sample):.1%}  (expected about 60-65%)")

# 3) the class from the web photo
print("\nPlantWild 'maple tar spot' test images:")
for p, cid in [t for t in test if labels[t[1]] == "maple tar spot"][:5]:
    name, conf = topk(p, 1)[0]
    print(f"  {p.name}: {name} ({conf:.0%})")
