# 🇳🇵 Explainable Fake News Detection in the Nepalese News Domain Using a GPT-BERT Hybrid Model

A deep learning system that detects fake news in Nepali-language articles using a hybrid **BERT + GPT-2 fusion model**, with **LIME-based explainability** to highlight which words influenced the classification decision. Served via a **Flask web application**.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Explainability](#explainability)
- [Configuration](#configuration)
- [Requirements](#requirements)

---

## Overview

Fake news in low-resource languages like Nepali remains an underexplored problem. This project addresses it by combining:

- **BERT (DAPT)** — a domain-adapted BERT model pre-trained on Nepalese text, fine-tuned for contextual understanding
- **GPT-2** — a generative model used for its complementary language representations
- **Fusion classifier** — a neural network head that merges both representations for binary classification (Real / Fake)
- **LIME** — Local Interpretable Model-agnostic Explanations, which surfaces the words most responsible for each prediction

---

## Architecture

```
Input Text
 ├── BERT-DAPT (fully fine-tuned)   → CLS token embedding      [768-dim]
 └── GPT-2 (last 2 layers tuned)   → Masked mean pooling       [768-dim]
                    ↓
           Concatenate              →                           [1536-dim]
                    ↓
     Dropout(0.3) → Linear(1536→256) → ReLU → Linear(256→2)
                    ↓
         Output: [Real, Fake] logits
```

**Key design choices:**
- BERT uses gradient checkpointing to reduce memory usage during training
- GPT-2 has all layers frozen except the last 2, limiting overfitting while leveraging pre-trained representations
- Masked mean pooling on GPT-2 output ignores padding tokens for cleaner features

---

## Project Structure

```
minnorprojd/
├── app.py              # Flask web app with REST API
├── config.py           # Central configuration (paths, hyperparameters, device)
├── models.py           # BertGptFusionClassifier model definition
├── train.py            # Training loop with early stopping
├── evaluate.py         # Evaluation metrics on test set
├── predict.py          # Inference pipeline (single text → prediction + explanation)
├── explain.py          # LIME explainability module
├── data_loader.py      # Dataset loading, cleaning, and DataLoader creation
├── requirements.txt    # Python dependencies
├── data/
│   └── final_dataset.csv   # Labelled Nepali news dataset (see Dataset section)
├── saved_models/
│   ├── bert_nepali_dapt/   # Domain-adapted BERT weights
│   ├── bert_tokenizer/     # Saved BERT tokenizer
│   ├── gpt_tokenizer/      # Saved GPT-2 tokenizer
│   └── fusion_model.pt     # Trained fusion classifier checkpoint
├── static/
│   ├── style.css
│   └── script.js
└── templates/
    └── index.html
```

---

## Dataset

The dataset is a labelled collection of Nepali news articles split into `train`, `val`, and `test` sets, with binary labels: `0 = Real`, `1 = Fake`.

📦 **Download the dataset from Kaggle:**
[https://www.kaggle.com/datasets/shashwatsrma/final-dataset](https://www.kaggle.com/datasets/shashwatsrma/final-dataset)

After downloading, place the file at:
```
data/final_dataset.csv
```

The CSV is expected to have the following columns:

| Column  | Description                          |
|---------|--------------------------------------|
| `title` | The news article title/content       |
| `label` | `0` for Real, `1` for Fake           |
| `split` | `train`, `val`, or `test`            |

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/shashwatsrma/EXPLAINABLE-FAKE-NEWS-DETECTION-IN-THE-NEPALESE-NEWS-DOMAIN-USING-A-GPT-BERT-HYBRID-MODEL.git
cd EXPLAINABLE-FAKE-NEWS-DETECTION-IN-THE-NEPALESE-NEWS-DOMAIN-USING-A-GPT-BERT-HYBRID-MODEL
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
.venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the dataset

Place `final_dataset.csv` in the `data/` folder (see [Dataset](#dataset) section above).

### 5. Download model weights

The trained model weights (`fusion_model.pt`, `bert_nepali_dapt/`, tokenizers) are hosted on HuggingFace Hub:

```bash
# Coming soon — HuggingFace repo link will be added here
```

Place them in the `saved_models/` directory matching the structure shown above.

---

## Usage

### Train the model

```bash
python train.py
```

This will train the BERT + GPT-2 fusion model with early stopping (patience = 2) and save the best checkpoint to `saved_models/fusion_model.pt`.

### Evaluate on test set

```bash
python evaluate.py
```

Prints accuracy, precision, recall, and F1 score on the held-out test set.

### Run a prediction from CLI

```bash
python predict.py
```

Runs predictions on sample texts and prints the label, confidence, and top LIME-identified words.

### Run the web app

```bash
python app.py
```

Then open [http://localhost:5000](http://localhost:5000) in your browser.

#### API Endpoints

| Method | Endpoint       | Description                        |
|--------|----------------|------------------------------------|
| GET    | `/`            | Web interface                      |
| POST   | `/api/predict` | Predict + explain a news article   |
| GET    | `/api/health`  | Health check                       |

**Example request:**

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "काठमाडौंमा नयाँ पुलको उद्घाटन भयो।", "explain": true}'
```

**Example response:**

```json
{
  "prediction": "Real",
  "confidence": 91.43,
  "probabilities": { "real": 91.43, "fake": 8.57 },
  "top_fake_words": [["BREAKING", 0.312], ["secretly", 0.289]],
  "top_real_words": [["inaugurated", 0.198], ["officials", 0.175]],
  "explanation": [["BREAKING", 0.312], ["secretly", 0.289], ...]
}
```

---

## Explainability

This project uses **LIME (Local Interpretable Model-agnostic Explanations)** to make predictions interpretable. For each prediction, LIME:

1. Perturbs the input text by randomly masking words
2. Observes how the model's confidence changes
3. Assigns an importance score to each word

Words with **positive scores** push the model toward **Fake**, while words with **negative scores** push toward **Real**. The top contributing words are returned alongside every prediction.

---

## Configuration

All hyperparameters and paths are centralised in `config.py`:

| Parameter                  | Default         | Description                          |
|----------------------------|-----------------|--------------------------------------|
| `MAX_SEQ_LENGTH`           | 128             | Max tokens for both BERT and GPT-2   |
| `BATCH_SIZE`               | 16              | Training batch size                  |
| `LEARNING_RATE`            | 2e-5            | AdamW learning rate                  |
| `EPOCHS`                   | 5               | Maximum training epochs              |
| `PATIENCE`                 | 2               | Early stopping patience              |
| `FUSION_HIDDEN_DIM`        | 256             | Hidden size of fusion head           |
| `FUSION_DROPOUT`           | 0.3             | Dropout in fusion head               |
| `GPT2_UNFREEZE_LAST_N_LAYERS` | 2           | Number of GPT-2 layers to fine-tune  |
| `LIME_NUM_FEATURES`        | 10              | Top words shown in explanation       |
| `LIME_NUM_SAMPLES`         | 300             | LIME perturbation samples            |

Device is auto-detected: **CUDA → MPS (Apple Silicon) → CPU**.

---

## Requirements

```
flask
torch
transformers
scikit-learn
numpy
pandas
joblib
lime
tqdm
```

Install with:
```bash
pip install -r requirements.txt
```

---

## Author

**Shashwat Sharma**
- GitHub: [@shashwatsrma](https://github.com/shashwatsrma)
- Dataset: [Kaggle](https://www.kaggle.com/datasets/shashwatsrma/final-dataset)
