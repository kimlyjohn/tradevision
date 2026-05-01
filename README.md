# 📈 TradeVision

> **Deep learning–powered technical chart pattern recognition using Transfer Learning with TensorFlow/Keras and Streamlit.**

---

## What TradeVision Does

TradeVision analyses candlestick / OHLC chart images and classifies them into one of the supported technical analysis patterns (Head and Shoulders, Double Top, Cup and Handle, etc.) using a fine-tuned EfficientNetB0 backbone.

### ✅ What it does
- Classifies uploaded chart images into one of the trained pattern categories
- Returns a confidence score and a full probability distribution over all classes
- Flags results with confidence below the configured threshold as "low confidence"
- Provides a clean Streamlit UI with an image preview, result card, and bar chart

### ❌ What it does NOT do
- **Does not predict future prices or market direction**
- Does not generate trading signals or financial advice
- Is not a replacement for a licensed financial analyst

---

## Project Structure

```
tradevision/
├── app.py                        # Streamlit entry point
├── config.py                     # All constants and config
├── pyproject.toml                # pytest config
├── requirements.txt
├── Makefile                      # Convenience commands
├── setup.sh                      # One-shot setup: macOS / Linux
├── setup.bat                     # One-shot setup: Windows (CMD)
├── setup.ps1                     # One-shot setup: Windows (PowerShell)
├── README.md
│
├── .venv/                        # Virtual environment (gitignored)
│
├── data/
│   ├── raw/                      # Raw dataset (gitignored)
│   ├── processed/                # Preprocessed splits (gitignored)
│   └── download_dataset.py       # Kaggle download script
│
├── tradevision/
│   ├── model/
│   │   ├── builder.py            # Model architecture
│   │   └── trainer.py            # Training loop
│   ├── data/
│   │   ├── loader.py             # Dataset loading + splitting
│   │   └── preprocessor.py      # Image preprocessing pipeline
│   ├── inference/
│   │   └── predictor.py          # Singleton model + inference
│   └── utils/
│       ├── logger.py             # App-wide logging
│       └── validators.py         # Input validation
│
├── ui/
│   ├── components/
│   │   ├── uploader.py
│   │   ├── result_card.py
│   │   └── confidence_chart.py
│   └── pages/
│       ├── home.py
│       └── about.py
│
├── models/                       # Saved .keras files (gitignored)
└── tests/                        # pytest test suite
```

---

## Setup

> **All dependencies are installed inside a project-local `.venv/` folder.**
> Nothing is ever installed into your global Python environment.

---

### Venv Activation — Quick Reference

| Platform | Shell | Activate command |
|---|---|---|
| macOS / Linux | bash / zsh | `source .venv/bin/activate` |
| Windows | Command Prompt | `.venv\Scripts\activate` |
| Windows | PowerShell | `.venv\Scripts\Activate.ps1` |
| Windows | Git Bash | `source .venv/Scripts/activate` |

> **Deactivate** (all platforms): type `deactivate`

---

### Option A — One-shot automated setup (recommended)

**macOS / Linux**
```bash
git clone https://github.com/youruser/tradevision.git
cd tradevision
bash setup.sh
```

**Windows — Command Prompt**
```bat
git clone https://github.com/youruser/tradevision.git
cd tradevision
setup.bat
```

**Windows — PowerShell**
```powershell
git clone https://github.com/youruser/tradevision.git
cd tradevision
.\setup.ps1
```

> If PowerShell blocks the script, run this once to allow local scripts:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

Each script:
1. Creates `.venv/` in the project root
2. Upgrades `pip` inside the venv
3. Installs all `requirements.txt` dependencies
4. Copies `.env.example` → `.env` if it doesn't already exist
5. Prints platform-specific activation instructions

---

### Option B — Manual step-by-step

```bash
# 1. Clone
git clone https://github.com/youruser/tradevision.git
cd tradevision

# 2. Create the venv inside the project (never touches global Python)
python3 -m venv .venv          # macOS / Linux
# python -m venv .venv         # Windows

# 3. Activate (pick your platform)
source .venv/bin/activate      # macOS / Linux
# .venv\Scripts\activate       # Windows CMD
# .venv\Scripts\Activate.ps1  # Windows PowerShell

# 4. Upgrade pip + install deps — all goes into the venv only
pip install --upgrade pip
`pip install -r requirements.txt`

# 5. When finished, deactivate
deactivate
```

> The `.venv/` directory is already in `.gitignore` — it will never be committed.

---

### 5. Configure Kaggle credentials

```bash
# macOS / Linux
cp .env.example .env

# Windows CMD
copy .env.example .env

# Edit .env and fill in:
#   KAGGLE_USERNAME=your_username
#   KAGGLE_KEY=your_api_key
# Get your key at: https://www.kaggle.com/account
```

> Alternatively, place `{"username":"…","key":"…"}` in `~/.kaggle/kaggle.json`.

---

### Using the Makefile (macOS / Linux)

With the venv active (`source .venv/bin/activate`), you can use short `make` commands:

```bash
make setup      # Create venv + install deps (same as setup.sh)
make download   # Download dataset from Kaggle
make train      # Train the model
make run        # Launch the Streamlit app
make test       # Run pytest suite
make clean      # Remove .venv/, logs/, __pycache__
```

**Windows users:** Run the equivalent commands directly inside the activated venv:
```bat
python data\download_dataset.py
python -m tradevision.model.trainer
streamlit run app.py
pytest tests\
```

---

## Download the Dataset

1. Open `config.py` and set `KAGGLE_DATASET_SLUG` to the actual Kaggle dataset slug
   (e.g. `"username/chart-patterns-dataset"`).
2. Run:

```bash
python data/download_dataset.py
```

Images will be extracted into `data/raw/` with one sub-folder per pattern class.

---

## Train the Model

```bash
python -m tradevision.model.trainer
```

This will:
1. Load and split images from `data/raw/`
2. Build the EfficientNetB0-based model
3. Train with EarlyStopping, ReduceLROnPlateau, and ModelCheckpoint
4. Save the best model to `models/tradevision_best.keras`
5. Save training curve and confusion matrix plots to `models/`
6. Print final test accuracy and classification report

**Training can be configured in `config.py`:**

| Parameter | Default | Description |
|---|---|---|
| `BACKBONE` | `"EfficientNetB0"` | Backbone architecture |
| `EPOCHS` | `40` | Maximum training epochs |
| `BATCH_SIZE` | `32` | Batch size |
| `LEARNING_RATE` | `0.001` | Initial Adam LR |
| `DROPOUT_RATE` | `0.40` | Dropout probability |
| `DENSE_UNITS` | `256` | Dense head width |

---

## Run the App

```bash
streamlit run app.py
```

The app opens at [http://localhost:8501](http://localhost:8501).

> If no trained model is found, the Home page shows a banner with training instructions.

---

## Run Tests

```bash
pytest tests/
```

Test coverage includes:
- `test_preprocessor.py` — image resize, normalisation, grayscale conversion
- `test_validators.py` — file extension, size, and image integrity validation
- `test_predictor.py` — mocked model inference, output structure, error handling

---

## Configuration Reference (`config.py`)

| Key | Default | Purpose |
|---|---|---|
| `MODEL_PATH` | `models/tradevision_best.keras` | Saved model location |
| `INPUT_SIZE` | `(224, 224)` | Image resize target |
| `SUPPORTED_FORMATS` | `.jpg .jpeg .png .webp` | Allowed upload extensions |
| `MAX_FILE_SIZE_MB` | `10` | Upload size limit |
| `CONFIDENCE_THRESHOLD` | `0.60` | Below this → "low confidence" |
| `TOP_N_CLASSES` | `5` | Bars shown in confidence chart |
| `BACKBONE` | `"EfficientNetB0"` | Swap to `"MobileNetV2"` freely |

---

## Disclaimer

TradeVision is a research and educational tool. It classifies visual patterns only.
**It does not predict prices, generate signals, or constitute financial advice.**
Use at your own risk. Consult a licensed financial professional before making any investment decisions.
