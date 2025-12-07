"""
training/train_loop.py

Training- und Evaluierungslogik für den Mini-Transformer.

Hauptfunktionen:
- train_one_epoch: eine Epoche über den Trainings-Dataloader
- evaluate: Auswertung auf dem Validierungs-Dataloader

Wir nehmen an, dass der Dataloader Batches in einer der Formen liefert:

    (input_ids, target_ids)
    oder
    {"input_ids": ..., "target_ids": ...}

Dabei:
    input_ids:  (B, T), dtype=torch.long
    target_ids: (B, T), dtype=torch.long
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.training.utils import save_checkpoint


def _unpack_batch(batch) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Hilfsfunktion, die einen Batch in (input_ids, target_ids) aufteilt,
    egal ob der Batch als Tuple oder dict vorliegt.
    """
    if isinstance(batch, dict):
        return batch["input_ids"], batch["target_ids"]
    elif isinstance(batch, (list, tuple)) and len(batch) == 2:
        return batch[0], batch[1]
    else:
        raise ValueError(
            "Batch-Format wird nicht unterstützt. Erwartet (input_ids, target_ids) "
            "oder {'input_ids': ..., 'target_ids': ...}."
        )


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_fn: nn.Module,
    *,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    max_grad_norm: Optional[float] = None,
    epoch: int = 0,
    log_interval: int = 100,
) -> float:
    """
    Führt eine Trainings-Epoche durch.

    Parameter:
        model:       MiniTransformer
        dataloader:  DataLoader für Trainingsdaten
        optimizer:   z. B. AdamW
        device:      torch.device('cuda' oder 'cpu')
        loss_fn:     z. B. nn.CrossEntropyLoss()
        scheduler:   optionaler LR-Scheduler (z. B. linear warmup + cosine decay)
        max_grad_norm: optionales Gradient Clipping (L2-Norm)
        epoch:       aktuelle Epochenzahl (nur für Logging)
        log_interval: alle wie viele Schritte geloggt werden soll

    Rückgabe:
        durchschnittlicher Loss über die Epoche
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for step, batch in enumerate(dataloader):
        input_ids, target_ids = _unpack_batch(batch)
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        # Forward
        logits, _ = model(input_ids, return_attentions=False)  # (B, T, V)

        # CrossEntropyLoss erwartet (B*T, V) und (B*T,)
        B, T, V = logits.shape
        loss = loss_fn(logits.view(B * T, V), target_ids.view(B * T))

        optimizer.zero_grad()
        loss.backward()

        # Optional: Gradient Clipping
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        num_batches += 1

        if (step + 1) % log_interval == 0:
            avg_loss = total_loss / num_batches
            print(
                f"[train] epoch={epoch} step={step+1}/{len(dataloader)} "
                f"loss={loss.item():.4f} avg_loss={avg_loss:.4f}"
            )

    avg_loss = total_loss / max(1, num_batches)
    return avg_loss


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
) -> float:
    """
    Auswertung des Modells auf einem Validierungs- oder Test-Dataloader.

    Wir berechnen hier nur den durchschnittlichen Loss.
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids, target_ids = _unpack_batch(batch)
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            logits, _ = model(input_ids, return_attentions=False)
            B, T, V = logits.shape
            loss = loss_fn(logits.view(B * T, V), target_ids.view(B * T))

            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / max(1, num_batches)
    return avg_loss


def maybe_save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    epoch: int,
    global_step: int,
    best_val_loss: float,
    checkpoint_dir,
) -> None:
    """
    Beispiel-Funktion zum Speichern eines Checkpoints.
    (Aktuell nur als Template, kann in main.py genutzt werden.)

    Du kannst diese Funktion nach Bedarf aus main.py heraus aufrufen,
    z. B. wenn sich der Validierungs-Loss verbessert hat.
    """
    state = {
        "epoch": epoch,
        "step": global_step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "best_val_loss": best_val_loss,
    }

    checkpoint_path = checkpoint_dir / f"checkpoint_epoch{epoch}_step{global_step}.pt"
    save_checkpoint(state, checkpoint_path)
