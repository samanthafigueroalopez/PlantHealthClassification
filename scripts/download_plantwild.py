from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="uqtwei2/PlantWild",
    repo_type="dataset",
    local_dir="data/raw/PlantWild",
)
print("PlantWild downloaded to data/raw/PlantWild")