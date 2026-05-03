"""
config.py — TradeVision Central Configuration

All constants, paths, and hyperparameters are defined here.
No magic numbers anywhere else in the codebase.
"""

from pathlib import Path

# ─── Project root ─────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent.resolve()

# ─── Data paths ───────────────────────────────────────────────────────────────
DATA_DIR = ROOT_DIR / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"

# ─── Model paths ──────────────────────────────────────────────────────────────
MODELS_DIR = ROOT_DIR / "models"
MODEL_PATH = MODELS_DIR / "tradevision_best.keras"
TRAINING_CURVES_PATH = MODELS_DIR / "training_curves.png"
CONFUSION_MATRIX_PATH = MODELS_DIR / "confusion_matrix.png"

# ─── Logging ──────────────────────────────────────────────────────────────────
LOGS_DIR = ROOT_DIR / "logs"
LOG_FILE = LOGS_DIR / "tradevision.log"

# ─── Image settings ───────────────────────────────────────────────────────────
INPUT_SIZE: tuple[int, int] = (224, 224)
SUPPORTED_FORMATS: list[str] = [".jpg", ".jpeg", ".png", ".webp"]
MAX_FILE_SIZE_MB: int = 10

# ─── Inference ────────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD: float = 0.60
CONFIDENCE_MARGIN_THRESHOLD: float = 0.15
REJECTION_LABEL: str = "Unrecognized or unsupported chart pattern"
TOP_N_CLASSES: int = 5          # how many bars to show in confidence chart

# ─── Dataset split ratios (must sum to 1.0) ──────────────────────────────────
TRAIN_SPLIT: float = 0.70
VAL_SPLIT: float = 0.15
TEST_SPLIT: float = 0.15

# ─── Model / Training hyperparameters ────────────────────────────────────────
#   Backbone can be swapped to "MobileNetV2" without any other code change
BACKBONE: str = "EfficientNetB0"
BATCH_SIZE: int = 32
EPOCHS: int = 40
LEARNING_RATE: float = 1e-3
DROPOUT_RATE: float = 0.40
DENSE_UNITS: int = 256