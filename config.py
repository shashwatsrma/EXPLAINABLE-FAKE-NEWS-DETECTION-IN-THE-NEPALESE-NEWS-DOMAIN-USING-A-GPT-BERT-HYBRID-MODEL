# ============================================================
# CONFIG.PY — Central Configuration
# ============================================================

import torch
import os

# -------------------- Paths --------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "./saved_models")

DATASET_PATH = os.path.join(DATA_DIR, "final_dataset.csv")

# DAPT BERT — domain-adapted BERT pre-trained on Nepalese text
DAPT_BERT_PATH = os.path.join(MODEL_DIR, "bert_nepali_dapt")

# Saved model paths
FUSION_MODEL_PATH = os.path.join(MODEL_DIR, "fusion_model.pt")
BERT_TOKENIZER_PATH = os.path.join(MODEL_DIR, "bert_tokenizer")
GPT_TOKENIZER_PATH = os.path.join(MODEL_DIR, "gpt_tokenizer")

# -------------------- Device --------------------
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

AMP_DTYPE = (
    torch.float16 if DEVICE.type == "cuda"
    else torch.bfloat16 if DEVICE.type == "mps"
    else None
)

# -------------------- Model --------------------
GPT2_MODEL_NAME = "gpt2"
NUM_LABELS = 2

# -------------------- Tokenization --------------------
MAX_SEQ_LENGTH = 128

# -------------------- Training --------------------
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
EPOCHS = 5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
PATIENCE = 2
MAX_GRAD_NORM = 1.0
SEED = 42

# GPT-2: freeze all except last N layers
GPT2_UNFREEZE_LAST_N_LAYERS = 2

# Fusion NN head
FUSION_HIDDEN_DIM = 256
FUSION_DROPOUT = 0.3

# -------------------- LIME Explainability --------------------
LIME_NUM_FEATURES = 10
LIME_NUM_SAMPLES = 300

# -------------------- Labels --------------------
LABEL_MAP = {0: "Real", 1: "Fake"}
LABEL_MAP_REVERSE = {"Real": 0, "Fake": 1}
