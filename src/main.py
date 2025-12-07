"""
main.py

Kommandozeilen-Einstiegspunkt zum Trainieren und Nutzen des Mini-Transformers.

Funktionen:

- Training:
    python -m src.main --train --epochs 3 --batch-size 64 --max-vocab-size 256

- Textgenerierung (Inference) aus einem Checkpoint:
    python -m src.main --generate \
        --checkpoint checkpoints/best_epoch3_val2.5905.pt \
        --prompt "alice was beginning to get very tired " \
        --max-new-tokens 200 \
        --max-vocab-size 256
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from src.config import Config
from src.dataset import load_cleaned_text, create_dataloader
from src.tokenizer import build_tokenizer_from_text, CharTokenizer
from src.model.transformer import MiniTransformer, TransformerConfig
from src.training.utils import (
    set_seed,
    SchedulerConfig,
    create_scheduler,
    load_checkpoint,
)
from src.training.train_loop import train_one_epoch, evaluate


# ---------------------------------------------------------------------------
#  Daten + Dataloader auf Basis von TextDataset/create_dataloader
# ---------------------------------------------------------------------------


def create_dataloaders_from_text(
    cfg: Config,
    max_vocab_size: int | None = None,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, CharTokenizer]:
    """
    Lädt bereinigten Text, trainiert einen CharTokenizer, wandelt den Text
    in Token-IDs um und baut daraus Train- und Val-Dataloader.

    Schritte:
        1. bereinigten Text laden
        2. Tokenizer fitten
        3. Text zu Token-IDs (Liste von ints)
        4. Train/Val-Split auf Tokenebene
        5. Dataloader aus token_ids + seq_len + batch_size bauen
    """
    data_cfg = cfg.data
    train_cfg = cfg.training

    # 1) Bereinigten Text laden
    text = load_cleaned_text(data_cfg)
    print(f"[data] Länge bereinigter Text: {len(text)} Zeichen")

    # 2) Tokenizer bauen
    tokenizer = build_tokenizer_from_text(
        text,
        max_vocab_size=max_vocab_size,
        min_freq=1,
    )
    print(f"[tokenizer] Vokabulargröße (inkl. Sondertoken): {tokenizer.vocab_size}")

    # 3) Gesamten Text in Token-IDs encoden (ohne BOS/EOS)
    ids = tokenizer.encode(text, add_special_tokens=False)  # List[int]
    print(f"[data] Anzahl Tokens im Korpus: {len(ids)}")

    # 4) Train/Val-Split auf Tokenebene
    val_fraction = data_cfg.val_split
    n_total = len(ids)
    n_val = int(n_total * val_fraction)
    n_train = n_total - n_val

    train_ids = ids[:n_train]
    val_ids = ids[n_train:]

    print(
        f"[data] Token-Split: train_tokens={len(train_ids)}, "
        f"val_tokens={len(val_ids)} (val_fraction={val_fraction})"
    )

    # 5) Dataloader mit deinem TextDataset/create_dataloader
    seq_len = data_cfg.max_seq_len
    batch_size = train_cfg.batch_size

    train_loader = create_dataloader(
        token_ids=train_ids,
        seq_len=seq_len,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )

    val_loader = create_dataloader(
        token_ids=val_ids,
        seq_len=seq_len,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    print(
        f"[data] Train-Batches: {len(train_loader)}, "
        f"Val-Batches: {len(val_loader)}, "
        f"seq_len={seq_len}, batch_size={batch_size}"
    )

    return train_loader, val_loader, tokenizer


# ---------------------------------------------------------------------------
#  Modell-Setup
# ---------------------------------------------------------------------------


def build_model_from_config(cfg: Config, vocab_size: int, pad_token_id: int | None):
    """
    Erzeugt einen MiniTransformer aus der globalen Config.Config und
    der ermittelten Vokabulargröße.
    """
    model_cfg = cfg.model

    # model_cfg.vocab_size setzen (damit alles konsistent ist)
    model_cfg.vocab_size = vocab_size

    t_cfg = TransformerConfig(
        vocab_size=vocab_size,
        d_model=model_cfg.d_model,
        n_heads=model_cfg.n_heads,
        n_layers=model_cfg.n_layers,
        d_ff=model_cfg.d_ff,
        max_seq_len=model_cfg.max_seq_len,
        dropout=model_cfg.dropout,
        pad_token_id=pad_token_id,
        use_causal_mask=True,
        tie_embeddings=True,
    )

    model = MiniTransformer(t_cfg)
    return model


# ---------------------------------------------------------------------------
#  Inferenz / Textgenerierung
# ---------------------------------------------------------------------------


def run_generation(
    cfg: Config,
    checkpoint_path: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float = 1.0,
    max_vocab_size: int | None = None,
) -> None:
    """
    Lädt Korpus + Tokenizer + Modell, zieht einen Checkpoint rein
    und generiert Text aus einem gegebenen Prompt.
    """
    model, tokenizer, device = _load_model_and_tokenizer_for_inference(
        cfg=cfg,
        checkpoint_path=checkpoint_path,
        max_vocab_size=max_vocab_size,
    )

    # Prompt -> IDs
    input_ids = tokenizer.encode(prompt, add_special_tokens=True)
    input_ids_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    print(f"[gen] Prompt: {repr(prompt)}")
    print(f"[gen] Input-IDs-Länge: {input_ids_tensor.shape[1]}")

    # Generieren
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids_tensor,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

    # IDs -> Text
    generated_ids_list = generated_ids[0].tolist()
    generated_text = tokenizer.decode(generated_ids_list, skip_special_tokens=False)

    print("=" * 27 + "[ GENERIERTER TEXT ]" + "=" * 27)
    print(generated_text)
    print("=" * 80)


# ---------------------------------------------------------------------------
#  CLI-Argumente
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mini-Transformer Training / Inference"
    )

    # Modi
    parser.add_argument(
        "--train",
        action="store_true",
        help="Training starten",
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Textgenerierung (Inference) starten",
    )

    # Training-Parameter
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Anzahl Epochen (überschreibt TrainingConfig.num_epochs)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batchgröße (überschreibt TrainingConfig.batch_size)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Lernrate (überschreibt TrainingConfig.learning_rate)",
    )
    parser.add_argument(
        "--max-vocab-size",
        type=int,
        default=None,
        help="Optional: Maximale Vokabulargröße (Char-Tokenizer).",
    )
    parser.add_argument(
        "--logdir",
        type=str,
        default="runs",
        help="TensorBoard-Logverzeichnis (Default: 'runs')",
    )

    # Inferenz-Parameter
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Pfad zu einem Checkpoint (.pt) für Inferenz (für --generate notwendig)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="hello world",
        help="Start-Text (Prompt) für die Generierung",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=100,
        help="Anzahl neuer Tokens, die generiert werden sollen",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help=(
            "Sampling-Temperatur (>0, 1.0 = neutral, "
            ">1.0 = kreativer, <1.0 = konservativer)"
        ),
    )

    return parser.parse_args()







def _load_model_and_tokenizer_for_inference(
    cfg: Config,
    checkpoint_path: str,
    max_vocab_size: int | None = None,
):
    """
    Gemeinsamer Helper für Inferenz/Visualisierung:

    - lädt bereinigten Korpus
    - trainiert CharTokenizer darauf
    - baut den MiniTransformer mit korrekter vocab_size und pad_token_id
    - lädt den Checkpoint
    """
    device = cfg.device.get_device()
    print(f"[device] Verwende Device für Inferenz: {device}")
    print(f"[gen] Lade Checkpoint: {checkpoint_path}")

    # 1) Korpus laden & Tokenizer wie beim Training bauen
    data_cfg = cfg.data
    text = load_cleaned_text(data_cfg)

    tokenizer = build_tokenizer_from_text(
        text,
        max_vocab_size=max_vocab_size,
        min_freq=1,
    )
    print(f"[gen] Vokabulargröße: {tokenizer.vocab_size}")

    # 2) Modell erstellen (gleiche Logik wie in run_generation)
    pad_token_id = tokenizer.pad_id if tokenizer.pad_id is not None else 0
    model = build_model_from_config(
        cfg,
        vocab_size=tokenizer.vocab_size,
        pad_token_id=pad_token_id,
    )
    model.to(device)

    # 3) Checkpoint laden
    state = load_checkpoint(Path(checkpoint_path), map_location=device)
    model.load_state_dict(state["model_state"])
    model.eval()
    print(
        f"[gen] Checkpoint-Epoche: {state.get('epoch', '?')}, "
        f"best_val_loss: {state.get('best_val_loss', '?')}"
    )

    return model, tokenizer, device

def get_attentions_for_prompt(
    cfg: Config,
    checkpoint_path: str,
    prompt: str,
    max_vocab_size: int | None = None,
):
    """
    Führt einen einzelnen Forward-Pass durch und gibt die Attention-Matrizen zurück.

    Rückgabe:
        tokens: Liste der dekodierten Token (z.B. Zeichen als Strings)
        attentions: Liste von Tensors, Länge = n_layers
            jedes Element: (B, H, T, T)
    """
    # 1) Modell + Tokenizer laden
    model, tokenizer, device = _load_model_and_tokenizer_for_inference(
        cfg=cfg,
        checkpoint_path=checkpoint_path,
        max_vocab_size=max_vocab_size,
    )

    # 2) Prompt encoden
    input_ids = tokenizer.encode(prompt, add_special_tokens=True)
    input_ids_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    print(f"[attn] Prompt: {repr(prompt)}")
    print(f"[attn] Input-IDs-Länge: {input_ids_tensor.shape[1]}")

    # 3) Forward-Pass mit Attention-Rückgabe
    with torch.no_grad():
        logits, attentions = model(
            input_ids_tensor,
            return_attentions=True,
        )

    if attentions is None:
        raise RuntimeError(
            "Model hat keine Attention-Matrizen zurückgegeben. "
            "Ist return_attentions in den Blöcken korrekt implementiert?"
        )

    # 4) Token-Strings aus den IDs machen (für spätere Visualisierung)
    # Wir dekodieren jeden einzelnen ID wieder zu einem "Token-String".
    token_strs = [tokenizer.decode([tid]) for tid in input_ids]


    print(f"[attn] Anzahl Layer: {len(attentions)}")
    for layer_idx, attn in enumerate(attentions):
        B, H, T1, T2 = attn.shape
        print(f"[attn] Layer {layer_idx}: Shape = {attn.shape}")
        assert T1 == T2 == len(input_ids), "T und Prompt-Länge sollten übereinstimmen"

    return token_strs, attentions


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    cfg = Config()  # globale Konfiguration aus src.config
    training_cfg = cfg.training

    # CLI-Overrides auf Config anwenden (nur relevant für Training)
    if args.epochs is not None:
        training_cfg.num_epochs = args.epochs
    if args.batch_size is not None:
        training_cfg.batch_size = args.batch_size
    if args.lr is not None:
        training_cfg.learning_rate = args.lr

    # Seed setzen
    set_seed(42)

    # Sicherstellen, dass nicht beides gleichzeitig gesetzt ist
    if args.train and args.generate:
        raise ValueError("Bitte entweder --train ODER --generate verwenden, nicht beides.")

    # ----------------- Inferenzmodus ----------------- #
    if args.generate:
        if args.checkpoint is None:
            raise ValueError(
                "Für --generate muss ein --checkpoint angegeben werden."
            )

        run_generation(
            cfg=cfg,
            checkpoint_path=args.checkpoint,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            max_vocab_size=args.max_vocab_size,
        )
        return

    # ----------------- Trainingsmodus ----------------- #
    device = cfg.device.get_device()
    print(f"[device] Verwende Device: {device}")

    if not args.train:
        print("Nichts zu tun. Starte mit `--train` oder `--generate`.")
        return

    # TensorBoard SummaryWriter
    writer = SummaryWriter(log_dir=args.logdir)
    print(f"[tensorboard] Logging nach: {args.logdir}")

    # Daten + Tokenizer + Dataloader
    train_loader, val_loader, tokenizer = create_dataloaders_from_text(
        cfg,
        max_vocab_size=args.max_vocab_size,
    )

    # Modell + Optimizer + Scheduler
    pad_token_id = tokenizer.pad_id if tokenizer.pad_id is not None else 0

    model = build_model_from_config(
        cfg,
        vocab_size=tokenizer.vocab_size,
        pad_token_id=pad_token_id,
    )
    model.to(device)

    print("[model] MiniTransformer-Konfiguration:")
    print(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_cfg.learning_rate,
        weight_decay=training_cfg.weight_decay,
    )

    loss_fn = nn.CrossEntropyLoss()

    # Gesamtanzahl Trainingsteps (ungefähr): Epochen * Schritte pro Epoche
    num_training_steps = training_cfg.num_epochs * len(train_loader)
    scheduler_cfg = SchedulerConfig(
        num_warmup_steps=training_cfg.warmup_steps,
        num_training_steps=max(1, num_training_steps),
    )
    scheduler = create_scheduler(optimizer, scheduler_cfg)

    # ----------------- Training Loop ----------------- #
    best_val_loss = float("inf")

    print("[train] Starte Training mit Parametern:")
    print("  ", asdict(training_cfg))

    global_step = 0

    for epoch in range(1, training_cfg.num_epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            loss_fn=loss_fn,
            scheduler=scheduler,
            max_grad_norm=training_cfg.max_grad_norm,
            epoch=epoch,
            log_interval=training_cfg.log_interval,
        )

        val_loss = evaluate(
            model=model,
            dataloader=val_loader,
            device=device,
            loss_fn=loss_fn,
        )

        global_step += len(train_loader)

        print(
            f"[epoch {epoch}] train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f}"
        )

        # TensorBoard-Logging (pro Epoche)
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        current_lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar("lr", current_lr, epoch)
        writer.flush()

        # Checkpoint bei verbessertem Val-Loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            cfg.training.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = (
                cfg.training.checkpoint_dir
                / f"best_epoch{epoch}_val{val_loss:.4f}.pt"
            )
            torch.save(
                {
                    "epoch": epoch,
                    "step": global_step,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "best_val_loss": best_val_loss,
                    "config": asdict(training_cfg),
                },
                ckpt_path,
            )
            print(f"[checkpoint] Neuer Best-Checkpoint: {ckpt_path}")

    writer.close()


if __name__ == "__main__":
    main()
