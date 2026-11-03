# Machine Learning Framework for Microplastic Detection in Water

Completed software project for Sachin, based on `Microplastic_Detection_PRD.pdf`.

The application uses a **trained YOLOv8n object detector** to locate suspected microplastic fragments in microscope images. It supplies a React dashboard, FastAPI API, confidence and calibration controls, class counts, bounding-box measurements, persistent SQLite history and PDF reports. Training and application Python code are also stored in `.ipynb` notebooks.

## Start on Suhas's laptop

Double-click **START_PROJECT.bat**, then open **http://127.0.0.1:8000**. The launcher uses the existing `new_env` Conda environment when available. The dashboard is already built: Node.js is not required to demonstrate the delivered app.

1. Click **Try a real test sample**, or upload a JPG/PNG microscopy image.
2. Leave confidence at 25% initially. Set microscope calibration only if known from your equipment.
3. Click **Analyze sample**.
4. Inspect bounding boxes, counts, confidence, dimensions and the class chart.
5. Click **Download analysis PDF**.
6. Open **Analysis history** to revisit saved analyses.
7. Open **Model & evaluation** to show measured validation and test metrics.

Stop the server with Ctrl+C in its terminal. History remains in `backend/storage`.

## Run on Sachin's Windows laptop

1. Extract the shared ZIP completely. Do not run files directly inside the ZIP.
2. Install **Python 3.10 or newer** from python.org, with the Python launcher / PATH options enabled.
3. Double-click **SETUP_WINDOWS.bat**. Internet is needed for this first dependency installation. It creates a project-local `.venv`; no Conda installation is required.
4. Double-click **START_PROJECT.bat**.
5. Open **http://127.0.0.1:8000**.

Inference runs on CPU by default and does not require an NVIDIA GPU. The trained model, frontend build, sample images and processed dataset are included. No Roboflow API key or paid account is needed.

If Windows SmartScreen marks a downloaded batch file, inspect it or run the equivalent commands below. If Python is missing or dependency installation fails, the setup script stops and displays the error.

### Manual setup / macOS / Linux

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux instead:
# source .venv/bin/activate
python -m pip install -r backend/requirements.txt
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Run commands in the extracted project's root directory. If port 8000 is occupied, replace it with 8001 and visit that URL.

## Python notebooks

**`ml/notebooks/microplastic_yolov8_training.ipynb`** contains setup, dataset download/conversion source, annotation checks, class distribution, ground-truth visualization, training source, actual evaluation metrics, precision-recall/confusion-matrix figures, held-out inference and a detection DataFrame.

**`ml/notebooks/microplastic_python_implementation.ipynb`** stores every application/ML/testing Python source file with optional export cells. `.py` files accompany these notebooks because FastAPI needs importable modules for the double-click launcher.

Open either notebook in VS Code with the Jupyter extension and select Python (`new_env` or the project's `.venv`). Alternatively:

```bash
python -m pip install jupyterlab
python -m jupyter lab
```

The training notebook defaults to the supplied trained model. Set `RUN_TRAINING=True` to train again and `RUN_EVALUATION=True` to recompute validation/test metrics. It does not retrain on every Run All.

### Google Colab

Upload the project ZIP to Colab, extract it under `/content`, and open/upload the training notebook. Uncomment its `%pip install` cell and select a GPU runtime. The included processed dataset avoids API keys and another data download. If only the notebook is uploaded, set `DOWNLOAD_DATA=True` to fetch and recreate the dataset (about 1.3 GB).

```python
import zipfile
with zipfile.ZipFile('/content/Microplastic_Detection_Sachin_COMPLETE.zip') as archive:
    archive.extractall('/content')
```

If you used another extraction location, set `PROJECT_ROOT` to the project folder in the notebook's setup cell. Training time depends on GPU availability. Source images, splits and training seed are fixed, but exact results can vary across hardware/software.

## Dataset and scope

Dataset: [andreusv / Microplastic-Fragments-WWTP-Dataset](https://huggingface.co/datasets/andreusv/Microplastic-Fragments-WWTP-Dataset), **CC BY 4.0**. See `DATASET_ATTRIBUTION.md` and `ml/data/SOURCE_DATASET_CARD.md`.

- Train: **1,406 images / 2,915 fragment boxes**.
- Validation: **302 images / 790 fragment boxes**.
- Test: **302 images / 852 fragment boxes**.
- One class: `microplastic_fragment`, read dynamically from `data.yaml` and model metadata.
- COCO foreground category 1 is mapped to YOLO class 0. Bounding boxes are converted from absolute xywh to normalized center xywh.
- The publisher's train/validation/test divisions are preserved. Exact source-image SHA-256 hashes were checked across splits; no exact duplicates were found.
- Images are proportionally resized to longest side <=1280 for distribution; YOLO trains at 640. Normalized annotations are preserved.
- Original Parquet downloads remain on Suhas's computer under `ml/data/raw`; the ZIP includes the processed images/labels rather than duplicate raw archives.

This is a **YOLO-compatible dataset rather than a Roboflow-hosted export**. It meets the annotation and class-configuration format requirement without needing credentials. The acquisition domain is prepared wastewater/sludge stereomicroscopy, not arbitrary photographs of water. Other shapes (e.g. fibers) are not annotated target classes. The project does not claim to detect them.

## Training and measured evaluation

Model: pretrained **YOLOv8n**, fine-tuned with up to 40 epochs, input size 640, batch 12, seed 42, early-stopping patience 12 and validation-selected `best.pt`. Training uses the training split only. Test data is used only for final evaluation and demonstration.

Actual values are in **`MODEL_EVALUATION.md`**, **`ml/evaluation/metrics.json`**, and the dashboard. Full training curves/logs and validation/test figures are included. `best.pt` and `last.pt` are under `ml/models`.

To reproduce:

```bash
python ml/scripts/train.py --epochs 40 --imgsz 640 --batch 12
python ml/scripts/evaluate.py
python ml/scripts/predict.py sample_data/test_sample_01.jpg --confidence 0.25
python tests/verify_project.py
```

Preparation can be rerun with `python ml/scripts/prepare_data.py` when you need to recreate data from the original public downloads. GPU PyTorch installation is only needed for accelerated retraining; application inference works with CPU PyTorch.

## Interpreting results

- **Confidence** is a per-box detector score, not the project's overall accuracy.
- **Pixel width/height/area** describe the bounding rectangle, not the exact contour area of a particle.
- **Physical width/height/area** are calculated only when a finite positive micrometers-per-pixel calibration is supplied.
- Calibration must apply to the uploaded image's pixel resolution. Resizing an image changes its micrometers-per-pixel factor; do not reuse an original-image factor without adjustment.
- Low / Moderate / High are **project-defined image count bands**, configured in `backend/config.json`: Low <10, Moderate 10-29, High >=30. They do not establish safe water, laboratory concentration or regulatory compliance.
- Image recognition alone **does not chemically confirm plastics or polymer composition**.
- An unseen microscope or different sample-preparation protocol can reduce detection quality. Evaluate on your own images before interpreting scientific results.

## API

`GET /health`, `GET /model-info`, `POST /predict`, `POST /generate-report`, `GET /history`, `GET /samples`.

`/predict` accepts a multipart `file`, `confidence` (0.01-1), and optional `calibration` (>0). `/generate-report` accepts JSON `{"analysis_id":"saved_analysis_id"}` and returns a PDF from the server's saved results.

Uploads are limited to 15 MB and 24 megapixels, validated by image decoder, and stored with generated UUID names. Unsupported/corrupt images, oversized inputs, invalid thresholds, missing models and unknown analysis IDs return readable errors. The server binds to localhost by default. History is local; the application is designed for a student demonstration, not a public multi-user service.

## Development frontend

```bash
cd frontend
npm install
npm run dev
```

Run FastAPI on port 8000 alongside it. Vite forwards API calls to the backend. `npm run build` creates the `frontend/dist` folder served by FastAPI. Recharts supplies the distribution visualization. The shipped build works without Node.js.

## Project files

| Folder / file | Purpose |
|---|---|
| `ml/notebooks` | Requested `.ipynb` Python work |
| `ml/scripts` | Download, training, evaluation and inference entry points |
| `ml/data` | Processed YOLO dataset, source card and audit |
| `ml/models` | Trained `best.pt` and `last.pt` |
| `ml/evaluation` | Actual validation/test metrics and figures |
| `ml/runs` | Training results and curves |
| `backend` | FastAPI, detector, analysis, report module, configuration |
| `frontend` | React + Vite source and ready-built dashboard |
| `sample_data` | Three real held-out microscopy images |
| `output/verification` | Tested annotated image, JSON and PDF report |
| `output/playwright` | Browser verification screenshots |
| `TESTING_REPORT.md` | Final verification evidence |
| `START_PROJECT.bat` | Start the app |
| `SETUP_WINDOWS.bat` | First-time setup on another Windows computer |

## Licensing

Project-authored source is provided under AGPL-3.0-or-later, consistent with the Ultralytics dependency. Dataset images/annotations retain CC BY 4.0 attribution; third-party libraries retain their own licenses. See `THIRD_PARTY_NOTICES.md`. Do not represent the dataset, pretrained YOLO architecture or original PRD as your own creation.
