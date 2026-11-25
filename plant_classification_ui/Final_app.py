"""LeafLens plant classification dashboard. Run with python -m streamlit run app.py."""
import csv
import hashlib
import html
import json
import time
from datetime import datetime
from io import BytesIO, StringIO
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

st.set_page_config(page_title="LeafLens · Research workspace", page_icon="🌱", layout="wide")
st.markdown('''<style>
.stApp{background:#f5f7fa;color:#20332d}
[data-testid="stHeader"]{background:rgba(245,247,250,.92)}
.block-container{max-width:1420px;padding-top:2.2rem;padding-bottom:2rem}
[data-testid="stSidebar"]{background:#112f27}
[data-testid="stSidebar"] *{color:#e8f0eb}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p{color:#a8c0b3}
[data-testid="stSidebar"] hr{border-color:#476155}
h1,h2,h3{letter-spacing:-.5px;color:#173c2e}
h1{font-size:2.3rem!important} h3{font-size:1.17rem!important}
[data-testid="stMetric"]{background:#fff;padding:19px 20px;border:1px solid #e0e7e3;border-radius:14px}
[data-testid="stMetricLabel"] p{color:#718077;font-size:13px}
[data-testid="stMetricValue"]{font-size:1.85rem;font-weight:700}
[data-testid="stVerticalBlockBorderWrapper"]{background:#ffffff;border-radius:16px;border-color:#e0e7e3!important}
.stButton button,.stDownloadButton button{border-radius:9px;min-height:42px}
.stButton button[kind="primary"]{background:#24734f;border:0}
[data-baseweb="tab-list"]{gap:22px}
.eyebrow{font-size:11px;letter-spacing:2px;font-weight:750;color:#718578;margin-bottom:8px}
.banner{background:linear-gradient(110deg,#153d2e,#2f6d51);border-radius:18px;padding:29px 34px;margin:16px 0 25px;color:#fff;position:relative;overflow:hidden}
.banner:after{content:"";position:absolute;border:1px solid #ffffff20;width:230px;height:230px;border-radius:50%;right:-35px;top:-80px;box-shadow:0 0 0 40px #ffffff08,0 0 0 80px #ffffff05;pointer-events:none}
.banner h2{color:white;font-size:27px;margin:0;padding:0 0 9px;letter-spacing:-.6px}
.banner p{color:#d3e4d8;font-size:14px;margin:0;max-width:670px}
.pill{font-size:10px;letter-spacing:1px;color:#22583f;background:#e2efe5;padding:6px 10px;border-radius:30px;display:inline-block;font-weight:700}
.empty{height:210px;display:flex;flex-direction:column;justify-content:center;align-items:center;background:#f6f8f6;border:1px dashed #cbd8ce;border-radius:12px;color:#809186;text-align:center;margin:14px 0}
.empty strong{font-size:17px;color:#4e6957;margin-bottom:8px}
.step{background:#edf3ef;border-radius:9px;padding:12px 16px;font-size:13px;color:#466252;margin:0 0 20px}
.result{background:#eaf4ed;border:1px solid #c9dfcf;border-radius:13px;padding:20px;margin:14px 0}
.result small{font-size:10px;letter-spacing:1.5px;color:#59806a}.result h2{font-size:25px;padding:6px 0;color:#204b34}
.footer{font-size:11px;color:#809086;border-top:1px solid #dfe7e1;padding-top:15px;margin-top:26px}
@media(max-width:700px){.banner{padding:22px}.banner h2{font-size:23px}.block-container{padding-top:1.5rem}}
</style>''', unsafe_allow_html=True)

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "resnet18_tuned_best.pth"
LABELS_PATH = APP_DIR / "class_names.json"
# Previously reported tuned-run test metrics; not measured on UI photos.
METRICS = {"Accuracy":63.60,"Precision":60.70,"Recall":59.96,"F1-score":59.89}


@st.cache_resource(show_spinner=False)
def load_classifier():
    try:
        import torch
        from torch import nn
        from torchvision import models, transforms
    except ImportError as exc:
        raise RuntimeError("Install torch and torchvision using requirements.txt, then restart the app.") from exc
    if not MODEL_PATH.is_file():
        raise FileNotFoundError("Place resnet18_tuned_best.pth beside app.py.")
    labels_path = LABELS_PATH
    if not labels_path.is_file():
        labels_path = APP_DIR / "class_names(1).json"
    if not labels_path.is_file():
        raise FileNotFoundError("Place class_names.json beside app.py.")
    with labels_path.open(encoding="utf-8") as file:
        labels = json.load(file)
    if not isinstance(labels, list) or len(labels) != 89 or not all(isinstance(x, str) and x for x in labels):
        raise ValueError("Class labels must be an ordered JSON list of 89 names.")
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(labels))
    state = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    # Matches the supplied project's evaluation pipeline. No training augmentation.
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return model, labels, preprocess


def predict_leaf(image):
    import torch
    model, labels, preprocess = load_classifier()
    with torch.inference_mode():
        probabilities = model(preprocess(image.convert("RGB")).unsqueeze(0)).softmax(dim=1)[0]
        scores, indices = probabilities.topk(3)
    return {labels[index]: float(score) for index, score in zip(indices.tolist(), scores.tolist())}


st.session_state.setdefault("history", [])


def decode_image(raw):
    """Validate local upload/camera bytes; use bounded images for previews."""
    if len(raw)>10*1024*1024:
        raise ValueError("Choose an image under 10 MB.")
    with Image.open(BytesIO(raw)) as original:
        if original.width*original.height>25_000_000:
            raise ValueError("Choose an image under 25 megapixels.")
        dimensions=original.size
        image=ImageOps.exif_transpose(original).convert("RGB")
        image.load()
    return image, dimensions


def export_csv(rows):
    output=StringIO()
    fields=["id","created_at","scenario","confidence","mode","filename","source","model"]
    writer=csv.DictWriter(output,fieldnames=fields,extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


# UI state is explicit so a captured/uploaded photo is processed once per attempt.
st.session_state.setdefault("stage", "input")
st.session_state.setdefault("attempt", 0)
st.session_state.setdefault("next_record_id", 1)


def start_over():
    st.session_state.stage = "input"
    st.session_state.attempt += 1
    st.session_state.pop("active_result", None)
    st.session_state.pop("preview", None)
    st.session_state.pop("processing_error", None)


def loading_animation():
    """Browser-native SVG/CSS animation; no extra files or packages required."""
    st.markdown('''<style>
.water-loading{padding:32px 12px 36px;text-align:center;min-height:350px}
.water-loading svg{width:240px;max-width:100%;height:220px;display:block;margin:auto}
.water-loading h3{font-size:22px!important;margin:8px 0}
.water-loading p{font-size:14px;color:#6b8072;margin:4px 0}
.water-can{transform-origin:171px 65px;animation:can-tilt 2s ease-in-out infinite}
.water-drop{opacity:0;animation:water-fall 1s linear infinite}
.water-drop.two{animation-delay:.32s}.water-drop.three{animation-delay:.65s}
.water-sprout{transform-origin:102px 153px;animation:sprout-sway 2s ease-in-out infinite}
.loading-dot{animation:dot-pulse 1.2s ease-in-out infinite;display:inline-block}
.loading-dot:nth-child(2){animation-delay:.2s}.loading-dot:nth-child(3){animation-delay:.4s}
@keyframes can-tilt{0%,100%{transform:rotate(0deg)}35%,75%{transform:rotate(-12deg)}}
@keyframes water-fall{0%{opacity:0;transform:translate(0,0)}20%{opacity:1}85%{opacity:.8}100%{opacity:0;transform:translate(-14px,58px)}}
@keyframes sprout-sway{0%,100%{transform:rotate(-3deg)}50%{transform:rotate(4deg)}}
@keyframes dot-pulse{0%,100%{opacity:.25}50%{opacity:1}}
@media(prefers-reduced-motion:reduce){.water-can,.water-drop,.water-sprout,.loading-dot{animation:none}.water-drop{opacity:.75}}
</style>
<div class="water-loading" role="status" aria-live="polite">
<svg viewBox="0 0 260 220" role="img" aria-label="A watering can watering a small plant">
<circle cx="130" cy="110" r="93" fill="#edf5ed"/>
<ellipse cx="107" cy="190" rx="48" ry="7" fill="#d6e4d7"/>
<g class="water-sprout">
<path d="M104 159V115" stroke="#47754b" stroke-width="5" stroke-linecap="round"/>
<path d="M103 139C77 140 70 124 70 116C89 113 104 121 103 139Z" fill="#79ab67"/>
<path d="M105 127C105 106 122 100 135 101C135 117 125 129 105 127Z" fill="#3f8255"/>
<path d="M103 119C91 110 91 100 96 92C108 99 113 111 103 119Z" fill="#9ac37c"/>
</g>
<path d="M81 158H130L123 187H88Z" fill="#c78e69"/>
<rect x="77" y="151" width="57" height="12" rx="4" fill="#dfaa82"/>
<g class="water-can">
<path d="M188 43C220 37 226 79 196 83" fill="none" stroke="#65867d" stroke-width="9"/>
<path d="M148 49H195L199 91Q175 104 148 89Z" fill="#7fa99b"/>
<ellipse cx="171" cy="49" rx="24" ry="6" fill="#adc9b8"/>
<path d="M150 72L124 57L114 72L149 89Z" fill="#7fa99b"/>
<path d="M120 56L110 73" stroke="#4c7868" stroke-width="7" stroke-linecap="round"/>
<path d="M158 58L158 84" stroke="#c1d8c6" stroke-width="4" stroke-linecap="round" opacity=".65"/>
</g>
<path class="water-drop" d="M116 78Q108 88 116 90Q124 88 116 78Z" fill="#72b7cc"/>
<path class="water-drop two" d="M121 78Q113 88 121 90Q129 88 121 78Z" fill="#72b7cc"/>
<path class="water-drop three" d="M111 78Q103 88 111 90Q119 88 111 78Z" fill="#72b7cc"/>
</svg>
<h3>Preparing your result<span class="loading-dot">.</span><span class="loading-dot">.</span><span class="loading-dot">.</span></h3>
<p>A little care for every leaf.</p>
<p>Analyzing your image with ResNet18</p>
</div>''', unsafe_allow_html=True)


def process_photo(raw, filename, source):
    """Classify a validated photo and save its real model scores."""
    image, dimensions = decode_image(raw)
    # Resolve dependencies and files before inference so errors are actionable.
    load_classifier()
    scores = predict_leaf(image)
    scenario = next(iter(scores))
    record = {
        "id": st.session_state.next_record_id,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "scenario": scenario,
        "confidence": scores[scenario],
        "mode": "model_prediction",
        "model": "ResNet18 (tuned)",
        "is_model_prediction": True,
        "scores": scores,
        "filename": filename,
        "source": source,
        "dimensions": list(dimensions),
    }
    st.session_state.next_record_id += 1
    st.session_state.history.append(record)
    preview = image.copy()
    preview.thumbnail((1400, 1400))
    st.session_state.preview = preview
    st.session_state.active_result = record
    st.session_state.stage = "result"


with st.sidebar:
    st.markdown("## 🌱 LeafLens")
    st.caption("PLANT CLASSIFICATION")
    st.divider()
    page = st.radio("WORKSPACE", ["Classify a leaf", "History & confidence"])
    st.divider()
    st.caption("ResNet18 · 89 PlantWild classes")
    st.caption("Upload or capture a leaf photo to classify it.")

st.markdown('<div class="eyebrow">LEAFLENS / PLANT CLASSIFICATION</div>', unsafe_allow_html=True)

if page == "Classify a leaf":
    st.title("A closer look at your leaf.")
    st.caption("Upload a photo or use your camera to begin.")
    # A placeholder is replaced when processing starts so the uploader gives way
    # to a small loading panel before the app reruns to the result view.
    workspace = st.empty()
    if st.session_state.stage == "input":
        with workspace.container():
            _, center, _ = st.columns([1, 2.5, 1])
            with center:
                with st.container(border=True):
                    st.subheader("Add a leaf photo")
                    source = st.radio("Image source", ["Upload image", "Take a photo"], horizontal=True)
                    if source == "Upload image":
                        photo = st.file_uploader("Choose a leaf photo", type=["png", "jpg", "jpeg"], key=f"upload_{st.session_state.attempt}")
                    else:
                        st.caption("Allow camera access, then capture a clear photo of your leaf.")
                        photo = st.camera_input("Take a leaf photo", key=f"camera_{st.session_state.attempt}")
                    st.caption("PNG / JPG · Up to 10 MB · Keep the leaf in focus.")
                    if st.session_state.get("processing_error"):
                        st.error(st.session_state.processing_error)
                    st.caption("Use a clear photo of a single leaf for the best input.")
        if photo is not None:
            raw, filename = photo.getvalue(), photo.name
            with workspace.container():
                _, center, _ = st.columns([1, 2.5, 1])
                with center:
                    with st.container(border=True):
                        loading_animation()
                        started = time.monotonic()
                        try:
                            process_photo(raw, filename, source)
                            # Keep the watering animation visible briefly for fast CPU predictions.
                            time.sleep(max(0, 1.0 - (time.monotonic() - started)))
                        except (OSError, ValueError, RuntimeError, ImportError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
                            st.session_state.processing_error = f"Could not classify this image: {exc}"
                            st.session_state.attempt += 1
            st.rerun()
    else:
        with workspace.container():
            _, center, _ = st.columns([1, 2.5, 1])
            with center:
                with st.container(border=True):
                    st.subheader("Your result")
                    st.image(st.session_state.preview, width="stretch")
                    row = st.session_state.active_result
                    st.markdown(f'<div class="result"><small>PREDICTED CLASSIFICATION</small><h2>{html.escape(row["scenario"])}</h2></div>', unsafe_allow_html=True)
                    st.caption("Research prediction. Results can be incorrect, especially for plants outside the training classes.")
                    st.caption("View confidence scores and previous results on History & confidence.")
                    st.button("Classify another leaf", type="primary", on_click=start_over, width="stretch")

else:
    st.title("History & confidence")
    st.caption("Review this session's results and inspect confidence scores.")
    history_tab, metrics_tab = st.tabs(["Session results", "Reported model performance"])
    with history_tab:
        records = st.session_state.history
        if not records:
            st.info("No results yet. Upload or capture a photo on Classify a leaf to classify a leaf.")
        else:
            st.caption(f"{len(records)} model prediction(s) this session")
            options = {r["id"]: r for r in records}
            selected = st.selectbox("Choose a result", list(reversed(options)), format_func=lambda key: f"#{key} · {options[key]['filename']} · {options[key]['created_at']}")
            record = options[selected]
            left, right = st.columns([1, 1.3])
            with left:
                with st.container(border=True):
                    st.subheader(record["scenario"])
                    st.metric("Model confidence", f"{record['confidence']:.0%}")
                    st.caption(f"Source: {record['source']}")
                    st.caption(f"Image size: {record['dimensions'][0]} × {record['dimensions'][1]} px")
            with right:
                with st.container(border=True):
                    st.subheader("Top 3 model scores")
                    for label, score in record["scores"].items():
                        st.write(f"{label} · {score:.0%}")
                        st.progress(score)
            st.warning("Confidence is the model’s softmax score, not a guaranteed probability of correctness. The top 3 scores may total less than 100%.")
            st.dataframe(pd.DataFrame([{"Record":r["id"], "File":r["filename"], "Predicted class":r["scenario"], "Model confidence":f"{r['confidence']:.0%}", "Time":r["created_at"]} for r in reversed(records)]), hide_index=True, width="stretch")
            a,b = st.columns(2)
            a.download_button("Export selected result", json.dumps(record, indent=2), "leaflens_result.json", "application/json")
            b.download_button("Export session history", export_csv(records), "leaflens_history.csv", "text/csv")
            confirm = st.checkbox("Confirm clearing session history")
            if st.button("Clear history", disabled=not confirm):
                st.session_state.history = []
                st.rerun()
        st.caption("History stays in session memory and may disappear when the session ends. Times include the server timezone offset.")
    with metrics_tab:
        st.subheader("ResNet18 test results")
        for col,(name,value) in zip(st.columns(4), METRICS.items()):
            col.metric(name, f"{value:.2f}%")
        st.caption("Previously reported tuned ResNet18 PlantWild test metrics; precision, recall, and F1 are macro averages. These are separate from per-image confidence and have not been remeasured here.")

st.markdown('<div class="footer">LEAFLENS · Plant classification research · Python + Streamlit</div>', unsafe_allow_html=True)
