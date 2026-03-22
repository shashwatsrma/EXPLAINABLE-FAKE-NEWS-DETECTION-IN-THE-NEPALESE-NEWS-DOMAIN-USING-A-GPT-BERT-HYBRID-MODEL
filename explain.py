# ============================================================
# EXPLAIN.PY — LIME Explainability Module
# ============================================================

import numpy as np
import torch
from lime.lime_text import LimeTextExplainer
from config import *


class FakeNewsExplainer:
    """
    LIME-based explainability for the unified GBERT-DAPT model.
    Perturbs input text and observes prediction changes to
    identify which words most influence the classification.
    """

    def __init__(self, model, bert_tokenizer, gpt2_tokenizer):
        self.model = model
        self.bert_tokenizer = bert_tokenizer
        self.gpt2_tokenizer = gpt2_tokenizer
        self.model.eval()

        self.explainer = LimeTextExplainer(
            class_names=[LABEL_MAP[0], LABEL_MAP[1]],
            split_expression=r"\W+",
            random_state=SEED,
        )

    def _predict_proba(self, texts):
        """Prediction function called by LIME internally."""
        all_probs = []

        for text in texts:
            bert_input = self.bert_tokenizer(
                text, return_tensors="pt", padding="max_length",
                truncation=True, max_length=MAX_SEQ_LENGTH,
            )
            gpt_input = self.gpt2_tokenizer(
                text, return_tensors="pt", padding="max_length",
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

            all_probs.append(probs)

        return np.array(all_probs)

    def explain(self, text, num_features=LIME_NUM_FEATURES, num_samples=LIME_NUM_SAMPLES):
        """Generate LIME explanation for a single text."""
        probs = self._predict_proba([text])[0]
        pred_label = int(np.argmax(probs))
        confidence = float(np.max(probs)) * 100

        explanation = self.explainer.explain_instance(
            text, self._predict_proba,
            num_features=num_features,
            num_samples=num_samples,
            labels=(0, 1),
        )

        word_weights = explanation.as_list(label=pred_label)

        fake_explanation = explanation.as_list(label=1)
        top_fake_words = [(w, round(s, 4)) for w, s in fake_explanation if s > 0]
        top_real_words = [(w, round(abs(s), 4)) for w, s in fake_explanation if s < 0]
        top_fake_words.sort(key=lambda x: x[1], reverse=True)
        top_real_words.sort(key=lambda x: x[1], reverse=True)

        return {
            "prediction": LABEL_MAP[pred_label],
            "confidence": round(confidence, 2),
            "explanation": [(w, round(s, 4)) for w, s in word_weights],
            "top_fake_words": top_fake_words[:num_features],
            "top_real_words": top_real_words[:num_features],
            "probabilities": {
                "real": round(float(probs[0]) * 100, 2),
                "fake": round(float(probs[1]) * 100, 2),
            },
        }
