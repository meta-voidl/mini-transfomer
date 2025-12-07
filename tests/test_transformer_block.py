"""
Tests für src/model/transformer_block.py

Wir prüfen:
- korrekte Input/Output-Shapes
- dass ein Backward-Pass (Gradienten) ohne Fehler funktioniert
- optionaler CUDA-Test, falls verfügbar
"""

import torch

from src.model.transformer_block import TransformerBlock, TransformerBlockConfig


def test_transformer_block_shapes_cpu():
    B, T, D = 3, 7, 32
    n_heads = 4
    d_ff = 64

    cfg = TransformerBlockConfig(
        d_model=D,
        n_heads=n_heads,
        d_ff=d_ff,
        dropout=0.1,
        use_causal_mask=True,
    )

    block = TransformerBlock(cfg)

    x = torch.randn(B, T, D)
    out, attn = block(x, return_attention=True)

    # Shape-Checks
    assert out.shape == (B, T, D)
    assert attn is not None
    assert attn.shape == (B, n_heads, T, T)


def test_transformer_block_backward_pass():
    """
    Ein kleiner Smoke-Test, dass:
    - forward funktioniert
    - loss.backward() ohne RuntimeError durchläuft
    - Gradienten auf den Parametern berechnet werden
    """
    B, T, D = 2, 5, 16
    n_heads = 4
    d_ff = 64

    cfg = TransformerBlockConfig(
        d_model=D,
        n_heads=n_heads,
        d_ff=d_ff,
        dropout=0.1,
        use_causal_mask=True,
    )

    block = TransformerBlock(cfg)

    # Input mit Gradienten
    x = torch.randn(B, T, D, requires_grad=True)

    out, _ = block(x, return_attention=False)

    # Dummy-Loss: Summe aller Ausgabewerte
    loss = out.sum()
    loss.backward()

    # Sicherstellen, dass Gradienten auf einigen Parametern vorhanden sind
    grad_exists = False
    for name, param in block.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad_exists = True
            break

    assert grad_exists, "Keine Gradienten auf den Block-Parametern berechnet"


def test_transformer_block_cuda_optional():
    """
    Optionaler Test: läuft der Block auf GPU (falls verfügbar)?
    """
    if not torch.cuda.is_available():
        return  # Test überspringen, wenn keine CUDA-GPU vorhanden ist

    device = torch.device("cuda")

    B, T, D = 2, 6, 32
    n_heads = 4
    d_ff = 128

    cfg = TransformerBlockConfig(
        d_model=D,
        n_heads=n_heads,
        d_ff=d_ff,
        dropout=0.1,
        use_causal_mask=True,
    )

    block = TransformerBlock(cfg).to(device)

    x = torch.randn(B, T, D, device=device, requires_grad=True)

    out, attn = block(x, return_attention=True)

    assert out.shape == (B, T, D)
    assert out.is_cuda
    assert attn is not None
    assert attn.shape == (B, n_heads, T, T)
    assert attn.is_cuda

    # Backward kurz testen
    loss = out.mean()
    loss.backward()
