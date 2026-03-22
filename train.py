"""# ============================================================
# TRAIN.PY — Training Pipeline
# ============================================================
# Unified BertGptFusionClassifier with DAPT BERT, warmup
# scheduler, gradient clipping, early stopping, mixed precision.
# ============================================================

import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

from config import *
from data_loader import load_dataset, create_dataloaders
from models import BertGptFusionClassifier
from evaluate import evaluate_model

# Reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)


def train_one_epoch(model, dataloader, optimizer, scheduler, criterion, epoch, total_epochs):
    #Train for one epoch with mixed precision and gradient clipping.
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    progress = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{total_epochs}")
    for batch in progress:
        bert_ids = batch["bert_ids"].to(DEVICE)
        bert_mask = batch["bert_mask"].to(DEVICE)
        gpt_ids = batch["gpt_ids"].to(DEVICE)
        gpt_mask = batch["gpt_mask"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        optimizer.zero_grad()

        if AMP_DTYPE:
            with torch.autocast(device_type=DEVICE.type, dtype=AMP_DTYPE):
                logits = model(bert_ids, bert_mask, gpt_ids, gpt_mask)
                loss = criterion(logits, labels)
        else:
            logits = model(bert_ids, bert_mask, gpt_ids, gpt_mask)
            loss = criterion(logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        progress.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct / total * 100:.1f}%")

    return total_loss / len(dataloader), correct / total * 100


def validate(model, dataloader, criterion):
    #Run validation and return loss + accuracy.
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            bert_ids = batch["bert_ids"].to(DEVICE)
            bert_mask = batch["bert_mask"].to(DEVICE)
            gpt_ids = batch["gpt_ids"].to(DEVICE)
            gpt_mask = batch["gpt_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            logits = model(bert_ids, bert_mask, gpt_ids, gpt_mask)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(dataloader), correct / total * 100


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("=" * 60)
    print("GBERT FAKE NEWS DETECTION — TRAINING PIPELINE")
    print(f"Device: {DEVICE} | Mixed Precision: {AMP_DTYPE if AMP_DTYPE else 'disabled'}")
    print(f"BERT: DAPT (Domain-Adapted)")
    print("=" * 60)

    # -------------------- Load Data --------------------
    print("\n[1/4] Loading dataset...")
    train_df, val_df, test_df = load_dataset()

    print("\n[2/4] Tokenizing and creating DataLoaders...")
    dataloaders, bert_tokenizer, gpt2_tokenizer = create_dataloaders(
        train_df, val_df, test_df
    )

    # -------------------- Initialize Model --------------------
    print("\n[3/4] Initializing model...")
    model = BertGptFusionClassifier().to(DEVICE)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total params: {total_params:,} | Trainable: {trainable:,}")

    # -------------------- Training Setup --------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    total_steps = len(dataloaders["train"]) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * WARMUP_RATIO),
        num_training_steps=total_steps,
    )

    # -------------------- Train with Early Stopping --------------------
    print(f"\n[4/4] Training for up to {EPOCHS} epochs (patience={PATIENCE})...\n")

    best_val_loss = float("inf")
    patience_count = 0

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(
            model, dataloaders["train"], optimizer, scheduler, criterion, epoch, EPOCHS
        )

        val_loss, val_acc = validate(model, dataloaders["val"], criterion)

        print(
            f"  Epoch {epoch + 1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_count = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "max_length": MAX_SEQ_LENGTH,
                    "id2label": LABEL_MAP,
                    "label2id": LABEL_MAP_REVERSE,
                },
                FUSION_MODEL_PATH,
            )
            print(f"  ✅ Best model saved (val_loss={val_loss:.4f})")
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"  Early stopping triggered at epoch {epoch + 1}")
                break

    # Load best model for evaluation
    checkpoint = torch.load(FUSION_MODEL_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    # -------------------- Save Tokenizers --------------------
    bert_tokenizer.save_pretrained(BERT_TOKENIZER_PATH)
    gpt2_tokenizer.save_pretrained(GPT_TOKENIZER_PATH)

    # -------------------- Evaluate --------------------
    print("\nEvaluating on TEST set...")

    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in tqdm(dataloaders["test"], desc="Testing"):
            bert_ids = batch["bert_ids"].to(DEVICE)
            bert_mask = batch["bert_mask"].to(DEVICE)
            gpt_ids = batch["gpt_ids"].to(DEVICE)
            gpt_mask = batch["gpt_mask"].to(DEVICE)

            logits = model(bert_ids, bert_mask, gpt_ids, gpt_mask)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            y_true.extend(batch["label"].numpy())
            y_pred.extend(preds)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    evaluate_model(y_true, y_pred, model_name="GBERT Fusion (DAPT + Neural Network)")

    # -------------------- Done --------------------
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Saved to: {MODEL_DIR}/")
    print(f"  fusion_model.pt      — Model checkpoint")
    print(f"  bert_tokenizer/      — DAPT BERT tokenizer")
    print(f"  gpt_tokenizer/       — GPT-2 tokenizer")


if __name__ == "__main__":
    main()
"""

# ============================================================
# TRAIN.PY — Training Pipeline
# ============================================================
# Unified BertGptFusionClassifier with DAPT BERT, warmup
# scheduler, gradient clipping, early stopping, mixed precision.
# ============================================================

import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

from config import *
from data_loader import load_dataset, create_dataloaders
from models import BertGptFusionClassifier
from evaluate import evaluate_model

# Reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)


def train_one_epoch(model, dataloader, optimizer, scheduler, criterion, epoch, total_epochs):
    """Train for one epoch with mixed precision and gradient clipping."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    progress = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{total_epochs}")
    for batch in progress:
        bert_ids  = batch["bert_ids"].to(DEVICE)
        bert_mask = batch["bert_mask"].to(DEVICE)
        gpt_ids   = batch["gpt_ids"].to(DEVICE)
        gpt_mask  = batch["gpt_mask"].to(DEVICE)
        labels    = batch["label"].to(DEVICE)

        optimizer.zero_grad()

        if AMP_DTYPE:
            with torch.autocast(device_type=DEVICE.type, dtype=AMP_DTYPE):
                logits = model(bert_ids, bert_mask, gpt_ids, gpt_mask)
                loss   = criterion(logits, labels)
        else:
            logits = model(bert_ids, bert_mask, gpt_ids, gpt_mask)
            loss   = criterion(logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds       = torch.argmax(logits, dim=1)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)

        progress.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct / total * 100:.1f}%")

    return total_loss / len(dataloader), correct / total


def validate(model, dataloader, criterion):
    """Run validation; return (avg_loss, accuracy 0-1)."""
    model.eval()
    total_loss = 0
    correct    = 0
    total      = 0

    with torch.no_grad():
        for batch in dataloader:
            bert_ids  = batch["bert_ids"].to(DEVICE)
            bert_mask = batch["bert_mask"].to(DEVICE)
            gpt_ids   = batch["gpt_ids"].to(DEVICE)
            gpt_mask  = batch["gpt_mask"].to(DEVICE)
            labels    = batch["label"].to(DEVICE)

            logits = model(bert_ids, bert_mask, gpt_ids, gpt_mask)
            loss   = criterion(logits, labels)

            total_loss += loss.item()
            preds       = torch.argmax(logits, dim=1)
            correct    += (preds == labels).sum().item()
            total      += labels.size(0)

    return total_loss / len(dataloader), correct / total


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("=" * 60)
    print("GBERT FAKE NEWS DETECTION — TRAINING PIPELINE")
    print(f"Device: {DEVICE} | Mixed Precision: {AMP_DTYPE if AMP_DTYPE else 'disabled'}")
    print(f"BERT: DAPT (Domain-Adapted)")
    print("=" * 60)

    # -------------------- Load Data --------------------
    print("\n[1/4] Loading dataset...")
    train_df, val_df, test_df = load_dataset()

    print("\n[2/4] Tokenizing and creating DataLoaders...")
    dataloaders, bert_tokenizer, gpt2_tokenizer = create_dataloaders(
        train_df, val_df, test_df
    )

    # -------------------- Initialize Model --------------------
    print("\n[3/4] Initializing model...")
    model = BertGptFusionClassifier().to(DEVICE)

    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total params: {total_params:,} | Trainable: {trainable:,}")

    # -------------------- Training Setup --------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    total_steps = len(dataloaders["train"]) * EPOCHS
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * WARMUP_RATIO),
        num_training_steps=total_steps,
    )

    # -------------------- Train with Early Stopping --------------------
    print(f"\n[4/4] Training for up to {EPOCHS} epochs (patience={PATIENCE})...\n")

    best_val_loss  = float("inf")
    patience_count = 0

    # ── History collectors (for curve plots) ──────────────────
    train_losses, train_accs = [], []
    val_losses,   val_accs   = [], []

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(
            model, dataloaders["train"], optimizer, scheduler, criterion, epoch, EPOCHS
        )
        val_loss, val_acc = validate(model, dataloaders["val"], criterion)

        # Store per-epoch history
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(
            f"  Epoch {epoch + 1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc * 100:.2f}% | "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc * 100:.2f}%"
        )

        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            patience_count = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "max_length":       MAX_SEQ_LENGTH,
                    "id2label":         LABEL_MAP,
                    "label2id":         LABEL_MAP_REVERSE,
                },
                FUSION_MODEL_PATH,
            )
            print(f"  ✅ Best model saved (val_loss={val_loss:.4f})")
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"  Early stopping triggered at epoch {epoch + 1}")
                break

    # Load best model for evaluation
    checkpoint = torch.load(FUSION_MODEL_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    # -------------------- Save Tokenizers --------------------
    bert_tokenizer.save_pretrained(BERT_TOKENIZER_PATH)
    gpt2_tokenizer.save_pretrained(GPT_TOKENIZER_PATH)

    # -------------------- Evaluate on Test Set --------------------
    print("\nEvaluating on TEST set...")

    model.eval()
    y_true, y_pred, y_prob = [], [], []

    with torch.no_grad():
        for batch in tqdm(dataloaders["test"], desc="Testing"):
            bert_ids  = batch["bert_ids"].to(DEVICE)
            bert_mask = batch["bert_mask"].to(DEVICE)
            gpt_ids   = batch["gpt_ids"].to(DEVICE)
            gpt_mask  = batch["gpt_mask"].to(DEVICE)

            logits = model(bert_ids, bert_mask, gpt_ids, gpt_mask)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
            preds  = np.argmax(probs, axis=1)

            y_true.extend(batch["label"].numpy())
            y_pred.extend(preds)
            y_prob.extend(probs[:, 1])   # probability of "Fake" (class 1)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    # ── Single call generates ALL 6 curve images ──────────────
    evaluate_model(
        y_true      = y_true,
        y_pred      = y_pred,
        model_name  = "GBERT Fusion (DAPT + Neural Network)",
        y_prob      = y_prob,          # → ROC, PR, dashboard PNGs
        train_losses= train_losses,    # → Loss curve PNGs
        train_accs  = train_accs,      # → Accuracy curve PNGs
        val_losses  = val_losses,
        val_accs    = val_accs,
    )

    # -------------------- Done --------------------
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Saved to: {MODEL_DIR}/")
    print(f"  fusion_model.pt      — Model checkpoint")
    print(f"  bert_tokenizer/      — DAPT BERT tokenizer")
    print(f"  gpt_tokenizer/       — GPT-2 tokenizer")
    print(f"\nCurve images saved to: eval_curves/")
    print(f"  *_training_curves.png  — Loss + Accuracy (combined)")
    print(f"  *_loss_curve.png       — Loss only")
    print(f"  *_accuracy_curve.png   — Accuracy only")
    print(f"  *_curves_dashboard.png — ROC + PR (combined)")
    print(f"  *_roc_curve.png        — ROC only")
    print(f"  *_pr_curve.png         — PR only")


if __name__ == "__main__":
    main()