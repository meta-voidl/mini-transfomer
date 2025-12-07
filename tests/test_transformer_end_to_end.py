"""
End-to-End-Tests für den MiniTransformer in src/model/transformer.py

Wir prüfen:
- Forward-Shape (Batch, Seq, Vocab)
- Keine NaNs im Output
- Backward-Pass (loss.backward()) läuft durch
- Optional: CUDA-Forward
- Padding-Maske maskiert PAD-Token wirklich
"""

import torch

from src.model.transformer import MiniTransformer, TransformerConfig


def test_forward_shapes_and_no_nan_cpu():
    B, T = 3, 7
    vocab_size = 50

    cfg = TransformerConfig(
        vocab_size=vocab_size,
        d_model=32,
        n_heads=4,
        n_layers=2,
        d_ff=64,
        max_seq_len=16,
        dropout=0.1,
        pad_token_id=0,
    )

    model = MiniTransformer(cfg)

    # Input-IDs im Bereich [0, vocab_size)
    input_ids = torch.randint(0, vocab_size, (B, T), dtype=torch.long)

    logits, _ = model(input_ids, return_attentions=False)

    assert logits.shape == (B, T, vocab_size)
    assert not torch.isnan(logits).any()


def test_backward_pass_end_to_end():
    B, T = 2, 5
    vocab_size = 30

    cfg = TransformerConfig(
        vocab_size=vocab_size,
        d_model=32,
        n_heads=4,
        n_layers=2,
        d_ff=64,
        max_seq_len=16,
        dropout=0.1,
        pad_token_id=0,
    )

    model = MiniTransformer(cfg)

    # Dummy-Input und Targets für Next-Token-Prediction
    input_ids = torch.randint(0, vocab_size, (B, T), dtype=torch.long)

    # Targets: einfach die gleichen IDs um 1 nach rechts verschoben,
    # für den Test reicht die Form; in der echten Training-Loop machst du das sauber.
    targets = torch.randint(0, vocab_size, (B, T), dtype=torch.long)

    logits, _ = model(input_ids, return_attentions=False)  # (B, T, V)

    # CrossEntropyLoss erwartet (B*T, V) und (B*T,)
    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(logits.view(-1, vocab_size), targets.view(-1))

    loss.backward()

    # Prüfen, ob zumindest ein Parameter Gradienten erhalten hat
    grad_exists = any(
        (p.grad is not None) for p in model.parameters() if p.requires_grad
    )
    assert grad_exists, "Es wurden keine Gradienten berechnet."


def test_padding_mask_blocks_attention_to_pad():
    """
    Wir konstruieren eine Sequenz mit PAD-Token am Ende und prüfen,
    dass Attention-Gewichte, die auf PAD-Positionen zeigen, klein/nahe 0 sind.

    Hinweis:
    - Das ist kein mathematisch strenger Beweis, aber ein sinnvoller
      Sanity-Check, dass die Padding-Maske überhaupt wirkt.
    """
    B, T = 1, 4
    vocab_size = 20
    pad_id = 0

    cfg = TransformerConfig(
        vocab_size=vocab_size,
        d_model=16,
        n_heads=2,
        n_layers=1,
        d_ff=32,
        max_seq_len=8,
        dropout=0.0,
        pad_token_id=pad_id,
    )

    model = MiniTransformer(cfg)

    # Sequenz: [token, token, PAD, PAD]
    input_ids = torch.tensor([[5, 6, pad_id, pad_id]], dtype=torch.long)

    logits, attentions = model(input_ids, return_attentions=True)
    attn = attentions[0]  # (B=1, H=2, T=4, T=4)

    # An allen Query-Positionen (i) sollen Keys an PAD-Positionen (j=2,3)
    # möglichst wenig Gewicht bekommen.
    pad_positions = torch.tensor([2, 3])

    # Mittelwert der Attention-Gewichte auf PAD-Positionen
    pad_weights = attn[0, :, :, pad_positions]  # (H, T, 2)
    pad_mean = pad_weights.mean()

    # Mittelwert der Attention-Gewichte auf Nicht-PAD-Positionen
    non_pad_positions = torch.tensor([0, 1])
    non_pad_weights = attn[0, :, :, non_pad_positions]  # (H, T, 2)
    non_pad_mean = non_pad_weights.mean()

    # Wir erwarten, dass im Schnitt auf Nicht-PAD mehr "geachtet" wird als auf PAD
    assert non_pad_mean > pad_mean


def test_optional_cuda_forward():
    if not torch.cuda.is_available():
        return  # CUDA nicht vorhanden -> Test überspringen

    device = torch.device("cuda")

    B, T = 2, 6
    vocab_size = 40

    cfg = TransformerConfig(
        vocab_size=vocab_size,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        max_seq_len=16,
        dropout=0.1,
        pad_token_id=0,
    )

    model = MiniTransformer(cfg).to(device)

    input_ids = torch.randint(0, vocab_size, (B, T), dtype=torch.long, device=device)

    logits, _ = model(input_ids, return_attentions=False)

    assert logits.shape == (B, T, vocab_size)
    assert logits.is_cuda
