# LeafLens — Python UI prototype

## Run locally

Use Python 3.10 or newer. Open a terminal inside this extracted folder:

```bash
python -m venv .venv
```

Activate on Windows:

```powershell
.venv\Scripts\activate
```

Or macOS/Linux:

```bash
source .venv/bin/activate
```

Install and run:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the localhost URL printed in the terminal.

## Included

- PNG/JPEG upload, validation, and preview
- Clearly labeled illustrative scenarios, independent of uploaded images
- Example confidence bars and JSON export
- ResNet18 performance dashboard using previously reported metrics
- Project information and consistent green styling

No model weights, GPU, API key, or dataset is required. Confidence examples are hand-authored, not actual model outputs. Overall test metrics came from the project discussion; averaging methods were not provided.

## Connect a model later

In `app.py`, replace `EXAMPLES` and the sample-results button logic with a prediction function. Match the training architecture, preprocessing, image dimensions, and class-index mapping exactly. Update the demo labels only after inference works.

Streamlit references: https://docs.streamlit.io/develop/api-reference/widgets/st.file_uploader and https://docs.streamlit.io/develop/api-reference/widgets/st.download_button
