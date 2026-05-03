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

## Model Approach & Architecture Design

TradeVision follows the **transfer learning** path from the course specification. The architecture, optimizer, and loss are defined directly in code in `tradevision/model/builder.py`.

### 1. Backbone and Input
- **Backbone**: `EfficientNetB0` (or `MobileNetV2`)
- **Input size**: `224 x 224`
- **Why this over others**: We chose EfficientNetB0 (and MobileNetV2 as a fallback) over heavier architectures like VGG16 or ResNet50 because they offer an optimal balance between accuracy and computational efficiency. They capture complex image features without requiring excessive training time or massive hardware.

### 2. Custom Classification Head Layer Selection
- **`GlobalAveragePooling2D`**: Flattens the feature maps from the backbone while preserving spatial context. 
  - *Why this over others*: Chose over a standard `Flatten` layer because `Flatten` drastically increases the parameter count, which almost inevitably leads to overfitting when training on moderately sized datasets. Global Average Pooling reduces spatial dimensions gracefully and makes the model more robust to spatial translations.
- **`Dense` Layer (with ReLU)**: Learns non-linear combinations of the extracted features. The ReLU activation introduces necessary non-linearity.
- **`BatchNormalization`**: Accelerates training convergence and stabilizes learning by normalizing the inputs to the subsequent layer. 
  - *Why this over others*: Chosen to mitigate internal covariate shifts. Relying on Batch Normalization is generally more robust and effective in training deep networks than solely depending on careful, manual weight initialization schemes.
- **`Dropout`**: Randomly sets a fraction of input units to 0 during training to act as a strict regularizer.
  - *Why this over others*: We chose Dropout as a regularizer over L1/L2 weight decay because it's highly effective at preventing the co-adaptation of neurons in dense layers. It essentially trains an ensemble of sub-networks, fundamentally reducing the risk of overfitting on our specific chart image dataset.
- **final `Dense(..., activation="softmax")`**: The output layer that maps abstractions to class probabilities. Softmax ensures output probabilities sum to 1. 

### 3. Optimizer and Loss Function
- **Optimizer**: `Adam`
  - *Why this over others*: The Adam (Adaptive Moment Estimation) optimizer dynamically scales updates based on the momentum of gradients. We chose this over standard SGD (Stochastic Gradient Descent) with momentum because Adam handles noisy, sparse gradients effectively and usually leads to faster, more robust convergence right out of the box with less manual learning rate tuning.
- **Loss**: `SparseCategoricalCrossentropy`
  - *Why this over others*: For mutually exclusive categories (a chart cannot simultaneously be a primary "Head and Shoulders" and "Double Bottom"), cross-entropy is standard. We chose the *Sparse* variant over standard `CategoricalCrossentropy` because it allows us to provide class labels as integers ($0, 1, 2\dots$) rather than requiring memory-heavy one-hot encoding matrices, making data pipelines simpler and more efficient. We bypassed Mean Squared Error since MSE is designed for continuous regression variables, not categorical probabilities.

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

Use this only if you need to regenerate the classifier-ready dataset. You can unzip the yolov8.zip file first.

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

The reorganization script converts the YOLO annotations into cropped class images under `data/processed/<split>/<class-name>/...` while preserving the original train, validation, and test boundaries.

## Makefile Shortcuts

With the project venv active:

```bash
make setup
make run
make train
make test
```

- `make run` launches the Streamlit UI
- `make train` retrains from `data/processed`


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
