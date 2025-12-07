"""
training/utils.py

Hilfsfunktionen für das Training:

- set_seed: Reproduzierbarkeit
- save_checkpoint / load_checkpoint: Modellzustand sichern/laden
- create_scheduler: einfacher LR-Scheduler (linear warmup + Cosine Decay)
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def set_seed(seed: int) -> None:
    """
    Setzt alle relevanten Zufallssamen für reproduzierbare Ergebnisse.

    Achtung:
    - Perfekte Reproduzierbarkeit auf GPU ist schwierig, aber das hier
      deckt die wichtigsten Quellen ab.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Optional striktere Settings (kann etwas Performance kosten)
    torch.backends.cudnn.deterministic = False  # True = deterministischer, aber langsamer
    torch.backends.cudnn.benchmark = True       # True = evtl. schneller bei fixen Shapes


def save_checkpoint(
    state: Dict[str, Any],
    path: Path,
) -> None:
    """
    Speichert einen Trainings-Checkpoint mit torch.save.

    Üblicher Inhalt von `state`:
        {
            "epoch": int,
            "step": int,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() (optional)
        }
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    print(f"[checkpoint] Gespeichert unter: {path}")


def load_checkpoint(
    path: Path,
    map_location: str | torch.device = "cpu",
) -> Dict[str, Any]:
    """
    Lädt einen Checkpoint von `path` und gibt das gespeicherte Dictionary zurück.

    Hinweis:
    - In PyTorch 2.6 ist der Default `weights_only=True`, was mit unseren
      Checkpoints (die auch Config/Path-Objekte enthalten) nicht kompatibel ist.
    - Da die Checkpoints lokal und vertrauenswürdig sind, setzen wir
      `weights_only=False`, was dem Verhalten vor 2.6 entspricht.
    """
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint nicht gefunden: {path}")

    state = torch.load(path, map_location=map_location, weights_only=False)
    print(f"[checkpoint] Geladen von: {path}")
    return state


@dataclass
class SchedulerConfig:
    """
    Konfiguration für den Learning-Rate-Scheduler.

    Wir implementieren:
        - linearen Warmup
        - danach Cosine Decay bis zum Ende des Trainings
    """
    num_warmup_steps: int
    num_training_steps: int


def create_scheduler(
    optimizer: Optimizer,
    cfg: SchedulerConfig,
) -> LambdaLR:
    """
    Erzeugt einen LambdaLR-Scheduler mit linear warmup + Cosine Decay.

    Lernratenverlauf (t = aktueller Schritt):
        - t < warmup  -> linear von 0 -> 1
        - t >= warmup -> Cosine von 1 -> 0

    Hinweis:
    - Scheduler steuert nur einen Multiplikator auf die "base_lr" des Optimizers.
    """

    def lr_lambda(current_step: int) -> float:
        if current_step < cfg.num_warmup_steps:
            # Linearer Warmup
            return float(current_step) / float(max(1, cfg.num_warmup_steps))

        # Nach Warmup: Cosine Decay
        progress = float(current_step - cfg.num_warmup_steps) / float(
            max(1, cfg.num_training_steps - cfg.num_warmup_steps)
        )
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    return scheduler
