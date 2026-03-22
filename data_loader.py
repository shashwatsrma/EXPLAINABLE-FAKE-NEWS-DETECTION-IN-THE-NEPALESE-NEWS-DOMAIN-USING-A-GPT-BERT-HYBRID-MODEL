# ============================================================
# DATA_LOADER.PY — Dataset Loading & Pre-Tokenization
# ============================================================

import re
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, GPT2Tokenizer
from config import *


def clean_text(text):
    """Light cleaning — preserves Nepali proper nouns that DAPT learned."""
    text = str(text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_dataset(path=DATASET_PATH):
    """Load the dataset and return split DataFrames."""
    df = pd.read_csv(path)
    df["content"] = df["title"].fillna("").astype(str).apply(clean_text)
    df["label"] = df["label"].astype(int)

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    print(f"Dataset loaded: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    return train_df, val_df, test_df


class FusionDataset(Dataset):
    """Pre-tokenized dataset for BERT (DAPT) + GPT-2 fusion model."""

    def __init__(self, texts, labels, bert_tokenizer, gpt2_tokenizer):
        self.labels = labels.tolist()
        print(f"    Pre-tokenizing {len(self.labels)} samples...")

        self.bert_encodings = bert_tokenizer(
            texts.tolist(),
            padding="max_length",
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            return_tensors="pt",
        )
        self.gpt_encodings = gpt2_tokenizer(
            texts.tolist(),
            padding="max_length",
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            return_tensors="pt",
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "bert_ids": self.bert_encodings["input_ids"][idx],
            "bert_mask": self.bert_encodings["attention_mask"][idx],
            "gpt_ids": self.gpt_encodings["input_ids"][idx],
            "gpt_mask": self.gpt_encodings["attention_mask"][idx],
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def create_dataloaders(train_df, val_df, test_df):
    """Create DataLoaders with pre-tokenized datasets."""
    # Load DAPT BERT tokenizer
    bert_tokenizer = BertTokenizer.from_pretrained(DAPT_BERT_PATH)

    gpt2_tokenizer = GPT2Tokenizer.from_pretrained(GPT2_MODEL_NAME)
    gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token

    print("  Creating train dataset...")
    train_ds = FusionDataset(train_df["content"], train_df["label"], bert_tokenizer, gpt2_tokenizer)
    print("  Creating val dataset...")
    val_ds = FusionDataset(val_df["content"], val_df["label"], bert_tokenizer, gpt2_tokenizer)
    print("  Creating test dataset...")
    test_ds = FusionDataset(test_df["content"], test_df["label"], bert_tokenizer, gpt2_tokenizer)

    dataloaders = {
        "train": DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True),
        "val": DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False),
        "test": DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False),
    }

    print("  DataLoaders ready.")
    return dataloaders, bert_tokenizer, gpt2_tokenizer
