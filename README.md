# TradeVision

TradeVision is a TensorFlow/Keras chart-pattern classification project for the SE 3231 Deep Learning Application final project. It uses transfer learning with an EfficientNetB0 backbone, a custom classification head, and a Streamlit UI for image upload and inference.

## What It Does

- Classifies chart screenshots into known technical pattern classes
- Shows the predicted class, confidence score, and probability distribution
- Flags low-confidence outputs instead of overstating certainty
- Ships with a ready-to-run Streamlit interface

## Submission Quick Start

This submission already includes the prepared datasets and the trained model. No credentials or dataset download steps are required for normal use.

### 1. Create and activate a virtual environment

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows CMD**
```bat
python -m venv .venv
.venv\Scripts\activate
```

**Windows PowerShell**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py
```

The app opens at [http://localhost:8501](http://localhost:8501).

## Included Artifacts

The submission bundle is expected to include:

- `data/raw/` — original YOLOv8 dataset export
- `data/processed/` — cropped class-organized dataset used by the classifier
- `models/tradevision_best.keras` — trained model checkpoint
- `models/class_names.json` — class label metadata
- `models/training_curves.png` — training/validation curves
- `models/confusion_matrix.png` — evaluation confusion matrix

Because these assets are bundled, the default grader flow is just: install dependencies and run the app.

## Project Structure

```text
tradevision/
├── app.py
├── config.py
├── requirements.txt
├── Makefile
├── setup.sh
├── setup.bat
├── setup.ps1
├── README.md
├── data/
│   ├── raw/
│   ├── processed/
│   ├── download_dataset.py
│   └── reorganize_yolov8.py
├── models/
├── tradevision/
│   ├── data/
│   ├── inference/
│   ├── model/
│   └── utils/
├── ui/
└── tests/
```

## Model Approach

TradeVision follows the **transfer learning** path from the course specification.

- Backbone: `EfficientNetB0`
- Input size: `224 x 224`
- Custom head:
  - `GlobalAveragePooling2D`
  - `Dense`
  - `BatchNormalization`
  - `Dropout`
  - final `Dense(..., activation="softmax")`
- Optimizer: `Adam`
- Loss: `SparseCategoricalCrossentropy`

The architecture, optimizer, and loss are defined directly in code in `tradevision/model/builder.py`.

## Training and Evaluation Outputs

Training writes the following artifacts to `models/`:

- `tradevision_best.keras`
- `class_names.json`
- `training_curves.png`
- `confusion_matrix.png`

These support the project evaluation criteria for model performance, confidence reporting, and reproducibility.

## Optional: Retrain From the Bundled Dataset

If the bundled `data/processed/` directory is already present, you can retrain directly with:

```bash
python -m tradevision.model.trainer
```

This trains the classifier from the processed class-folder dataset and overwrites the model artifacts in `models/`.

## Optional: Rebuild `data/processed` From the Raw YOLO Export

Use this only if you need to regenerate the classifier-ready dataset.

1. Ensure the YOLOv8 export is present at:

```text
data/raw/Chart-pattern.v2i.yolov8
```

2. Rebuild the processed dataset:

```bash
python data/reorganize_yolov8.py
```

3. Retrain the classifier:

```bash
python -m tradevision.model.trainer
```

The reorganization script converts the YOLO annotations into cropped class images under `data/processed/<class-name>/...`.

## Optional: Acquire the Dataset Yourself

This is a maintainer or fresh-clone workflow, not a required submission step.

If you do not have the bundled raw dataset, you can optionally download it from Roboflow:

```bash
ROBOFLOW_API_KEY=your_key python data/download_dataset.py
```

You may also override the dataset source:

```bash
ROBOFLOW_API_KEY=your_key \
ROBOFLOW_WORKSPACE=your-workspace \
ROBOFLOW_PROJECT=your-project \
ROBOFLOW_VERSION=your-version \
python data/download_dataset.py
```

After download, rebuild the processed dataset and retrain:

```bash
python data/reorganize_yolov8.py
python -m tradevision.model.trainer
```

## Makefile Shortcuts

With the project venv active:

```bash
make setup
make run
make train
make test
make fetch-data
```

- `make run` launches the Streamlit UI
- `make train` retrains from `data/processed`
- `make fetch-data` is optional and downloads the raw dataset from Roboflow

## Run Tests

```bash
pytest tests/
```

The test suite covers dataset loading, preprocessing, predictor behavior, validation, and training callbacks.

## Notes and Limits

- TradeVision is a pattern classifier, not a trading signal generator.
- It does not predict future prices, returns, or risk.
- Low-confidence outputs should be treated cautiously.
- Results depend on dataset quality, labeling quality, and similarity between uploaded charts and the training distribution.

## Disclaimer

TradeVision is an academic machine learning project. It is for pattern-recognition demonstration and evaluation only, and it should not be used as financial advice.
