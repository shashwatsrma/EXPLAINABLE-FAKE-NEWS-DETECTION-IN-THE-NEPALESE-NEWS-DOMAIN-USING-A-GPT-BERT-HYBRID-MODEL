# ============================================================
# PREDICT.PY — Single Text Prediction with Explanation
# ============================================================

import os
import re
import torch
import numpy as np
from transformers import BertTokenizer, GPT2Tokenizer

from config import *
from models import BertGptFusionClassifier
from explain import FakeNewsExplainer


class FakeNewsPredictor:
    """
    Complete prediction pipeline: text → preprocessing →
    unified GBERT-DAPT model → LIME explanation.
    """

    def __init__(self):
        print(f"Loading models (device: {DEVICE})...")

        # Load tokenizers
        self.bert_tokenizer = BertTokenizer.from_pretrained(BERT_TOKENIZER_PATH)
        self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained(GPT_TOKENIZER_PATH)
        self.gpt2_tokenizer.pad_token = self.gpt2_tokenizer.eos_token

        # Load unified model (BERT loads from DAPT path inside the class)
        self.model = BertGptFusionClassifier(bert_path=DAPT_BERT_PATH).to(DEVICE)
        checkpoint = torch.load(FUSION_MODEL_PATH, map_location=DEVICE, weights_only=True)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        # Initialize LIME explainer
        self.explainer = FakeNewsExplainer(
            model=self.model,
            bert_tokenizer=self.bert_tokenizer,
            gpt2_tokenizer=self.gpt2_tokenizer,
        )

        print("All models loaded successfully.")

    def clean_text(self, text):
        """Light cleaning — preserves proper nouns that DAPT learned."""
        text = str(text)
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def predict(self, text, explain=True):
        """Full prediction with optional LIME explanation."""
        cleaned = self.clean_text(text)

        if not cleaned or len(cleaned) < 10:
            return {
                "prediction": "Unknown",
                "confidence": 0.0,
                "error": "Text too short for reliable classification.",
            }

        if explain:
            return self.explainer.explain(cleaned)
        else:
            bert_input = self.bert_tokenizer(
                cleaned, return_tensors="pt", padding="max_length",
                truncation=True, max_length=MAX_SEQ_LENGTH,
            )
            gpt_input = self.gpt2_tokenizer(
                cleaned, return_tensors="pt", padding="max_length",
                truncation=True, max_length=MAX_SEQ_LENGTH,
            )

            with torch.no_grad():
                logits = self.model(
                    bert_input["input_ids"].to(DEVICE),
                    bert_input["attention_mask"].to(DEVICE),
                    gpt_input["input_ids"].to(DEVICE),
                    gpt_input["attention_mask"].to(DEVICE),
                )
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

            pred_label = int(np.argmax(probs))
            return {
                "prediction": LABEL_MAP[pred_label],
                "confidence": round(float(np.max(probs)) * 100, 2),
                "probabilities": {
                    "real": round(float(probs[0]) * 100, 2),
                    "fake": round(float(probs[1]) * 100, 2),
                },
            }


# -------------------- CLI Testing --------------------
if __name__ == "__main__":
    predictor = FakeNewsPredictor()

    test_texts = [
        "Prime Minister inaugurated the new bridge in Kathmandu today, officials confirmed.",
        "BREAKING: Nepal secretly sells Mount Everest to China for 500 billion dollars!",
    ]

    for text in test_texts:
        print(f"\n{'=' * 60}")
        print(f"Input: {text[:80]}...")
        result = predictor.predict(text, explain=True)
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']}%")
        if "top_fake_words" in result:
            print(f"Words suggesting FAKE: {result['top_fake_words'][:5]}")
            print(f"Words suggesting REAL: {result['top_real_words'][:5]}")
