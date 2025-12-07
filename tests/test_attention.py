"""
Tests für src/model/attention.py

Wir prüfen:
- korrekte Shapes von scaled_dot_product_attention
- dass die causale Maske Zukunftstokens ausblendet (Gewicht = 0)
- dass keine NaNs entstehen
- dass MultiHeadSelfAttention auf CPU (und optional GPU) läuft
"""

import torch

from src.model.attention import (
    AttentionConfig,
    scaled_dot_product_attention,
    MultiHeadSelfAttention,
)


def test_scaled_dot_product_attention_shapes_and_no_nan():
    B, H, T, Dh = 2, 3, 4, 5

    q = torch.randn(B, H, T, Dh)
    k = torch.randn(B, H, T, Dh)
    v = torch.randn(B, H, T, Dh)

    out, attn = scaled_dot_product_attention(q, k, v, mask=None)

    assert out.shape == (B, H, T, Dh)
    assert attn.shape == (B, H, T, T)

    # Keine NaNs in Ausgabe oder Attention-Matrix
    assert not torch.isnan(out).any()
    assert not torch.isnan(attn).any()


def test_causal_mask_blocks_future_tokens():
    """
    Testet, dass bei Multi-Head Self-Attention mit use_causal_mask=True
    kein Token auf zukünftige Positionen schaut (j > i).

    Wir holen uns dazu explizit die Attention-Gewichte und prüfen,
    dass die oberen Dreieckseinträge (j > i) == 0 sind.
    """
    B, T, D, H = 1, 5, 16, 2

    # Einfacher Input – hier ist der genaue Wert egal,
    # wir schauen nur auf die Maske.
    x = torch.randn(B, T, D)

    cfg = AttentionConfig(d_model=D, n_heads=H, dropout=0.0, use_causal_mask=True)
    layer = MultiHeadSelfAttention(cfg)

    _, attn = layer(x, return_attention=True)  # attn: (B, H, T, T)

    # Obere Dreiecksmatrix (j > i) -> sollte == 0 sein
    # Wir erzeugen Indizes für das obere Dreieck.
    indices = torch.triu_indices(T, T, offset=1)
    i_idx, j_idx = indices[0], indices[1]

    # Auswahl: (B=0, alle Heads, i_idx, j_idx)
    masked_values = attn[0, :, i_idx, j_idx]

    # Alle Werte sollten (numerisch) 0 sein
    assert torch.allclose(masked_values, torch.zeros_like(masked_values))


def test_multihead_self_attention_shapes_cpu_and_optional_cuda():
    B, T, D, H = 3, 7, 32, 4

    x = torch.randn(B, T, D)

    cfg = AttentionConfig(d_model=D, n_heads=H, dropout=0.1, use_causal_mask=True)
    layer = MultiHeadSelfAttention(cfg)

    # CPU-Forward
    out_cpu, attn_cpu = layer(x, return_attention=True)
    assert out_cpu.shape == (B, T, D)
    assert attn_cpu.shape == (B, H, T, T)

    # Optional: CUDA-Forward, wenn verfügbar
    if torch.cuda.is_available():
        device = torch.device("cuda")
        layer_cuda = layer.to(device)
        x_cuda = x.to(device)
        out_cuda, attn_cuda = layer_cuda(x_cuda, return_attention=True)

        assert out_cuda.shape == (B, T, D)
        assert attn_cuda.shape == (B, H, T, T)
        assert out_cuda.is_cuda
        assert attn_cuda.is_cuda
