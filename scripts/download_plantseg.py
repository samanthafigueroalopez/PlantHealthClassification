import os
import requests

RECORD_ID = "17719108"  # from DOI 10.5281/zenodo.17719108
OUT_DIR = "data/raw/PlantSeg"
os.makedirs(OUT_DIR, exist_ok=True)

record = requests.get(f"https://zenodo.org/api/records/{RECORD_ID}").json()
for f in record["files"]:
    name, url = f["key"], f["links"]["self"]
    path = os.path.join(OUT_DIR, name)
    if os.path.exists(path):
        print("Already have", name)
        continue
    print("Downloading", name, "...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(path, "wb") as out:
            for chunk in r.iter_content(chunk_size=1 << 20):
                out.write(chunk)
print("PlantSeg downloaded to", OUT_DIR)