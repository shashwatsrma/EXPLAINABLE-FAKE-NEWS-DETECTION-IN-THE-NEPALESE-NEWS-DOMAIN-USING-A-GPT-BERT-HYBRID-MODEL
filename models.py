# ============================================================
# MODELS.PY — Unified BERT (DAPT) + GPT-2 Fusion Classifier
# ============================================================

import torch
import torch.nn as nn
from transformers import BertModel, GPT2Model
from config import *


class BertGptFusionClassifier(nn.Module):
    """
    Unified BERT + GPT-2 fusion model for fake news classification.

    Architecture:
        Input text
         ├── BERT-DAPT (domain-adapted, fully fine-tuned) → CLS token [768]
         └── GPT-2 (last 2 layers fine-tuned) → Masked mean pooling [768]
              ↓
        Concatenate [1536]
              ↓
        Dropout(0.3) → Linear(1536, 256) → ReLU → Linear(256, 2)
              ↓
        Output: [Real, Fake] logits
    """

    def __init__(self, bert_path=DAPT_BERT_PATH):
        super().__init__()

        # BERT: load domain-adapted weights with gradient checkpointing
        self.bert = BertModel.from_pretrained(bert_path)
        self.bert.gradient_checkpointing_enable()

        # GPT-2: freeze all layers except last N
        self.gpt2 = GPT2Model.from_pretrained(GPT2_MODEL_NAME)
        for param in self.gpt2.parameters():
            param.requires_grad = False
        for param in self.gpt2.h[-GPT2_UNFREEZE_LAST_N_LAYERS:].parameters():
            param.requires_grad = True

        # Fusion classification head
        self.dropout = nn.Dropout(FUSION_DROPOUT)
        self.fc1 = nn.Linear(768 + 768, FUSION_HIDDEN_DIM)
        self.relu = nn.ReLU()
        self.out = nn.Linear(FUSION_HIDDEN_DIM, NUM_LABELS)

    def forward(self, bert_ids, bert_mask, gpt_ids, gpt_mask):
        # BERT: CLS token
        bert_out = self.bert(input_ids=bert_ids, attention_mask=bert_mask)
        bert_feat = bert_out.last_hidden_state[:, 0, :]

        # GPT-2: masked mean pooling
        gpt_out = self.gpt2(input_ids=gpt_ids, attention_mask=gpt_mask)
        mask_exp = gpt_mask.unsqueeze(-1).float()
        gpt_feat = (gpt_out.last_hidden_state * mask_exp).sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1e-9)

        # Fusion + classification
        fused = torch.cat((bert_feat, gpt_feat), dim=1)
        x = self.dropout(fused)
        x = self.relu(self.fc1(x))
        logits = self.out(x)
        return logits
