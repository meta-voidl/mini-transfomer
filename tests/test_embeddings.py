"""
Tests für src/model/embeddings.py

Wir prüfen hier hauptsächlich:
- korrekte Shapes
- sinnvolle Fehlerbehandlung bei zu langen Sequenzen
- dass Positions-Embeddings sich pro Position unterscheiden
"""

import torch
import pytest

from src.model.embeddings import (
    EmbeddingConfig,
    TokenEmbedding,
    PositionalEmbedding,
    TransformerEmbedding,
)


def test_token_embedding_shape():
    vocab_size = 100
    d_model = 32
    batch_size = 4
    seq_len = 10

    layer = TokenEmbedding(vocab_size, d_model)

    token_ids = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)
    out = layer(token_ids)

    assert out.shape == (batch_size, seq_len, d_model)
    assert out.dtype == torch.float32


def test_positional_embedding_shape_and_error():
    d_model = 32
    max_seq_len = 8
    batch_size = 2
    seq_len_ok = 5
    seq_len_too_long = 10

    layer = PositionalEmbedding(max_seq_len=max_seq_len, d_model=d_model)

    # Fall 1: seq_len <= max_seq_len -> sollte funktionieren
    token_ids_ok = torch.zeros((batch_size, seq_len_ok), dtype=torch.long)
    out_ok = layer(token_ids_ok)
    assert out_ok.shape == (batch_size, seq_len_ok, d_model)

    # Fall 2: seq_len > max_seq_len -> sollte ValueError werfen
    token_ids_too_long = torch.zeros((batch_size, seq_len_too_long), dtype=torch.long)
    with pytest.raises(ValueError):
        _ = layer(token_ids_too_long)


def test_positional_embeddings_differ_per_position():
    """
    Positions-Embeddings für unterschiedliche Positionen sollten sich
    (typischerweise) unterscheiden – ansonsten wäre die Position
    nicht codiert.

    Wir testen nur sehr grob, dass Position 0 und 1 nicht exakt
    denselben Vektor haben.
    """
    d_model = 16
    max_seq_len = 10
    batch_size = 1
    seq_len = 4

    layer = PositionalEmbedding(max_seq_len=max_seq_len, d_model=d_model)

    token_ids = torch.zeros((batch_size, seq_len), dtype=torch.long)
    pos_emb = layer(token_ids)  # (1, seq_len, d_model)

    vec_pos0 = pos_emb[0, 0, :]
    vec_pos1 = pos_emb[0, 1, :]

    # Es ist extrem unwahrscheinlich, dass zwei zufällige Initialisierungen
    # EXACT gleich sind. Wir nutzen daher einen einfachen Vergleich.
    assert not torch.allclose(vec_pos0, vec_pos1)


def test_transformer_embedding_shape_cpu_and_optional_cuda():
    """
    Testet, ob TransformerEmbedding auf CPU (und optional CUDA) die
    erwarteten Shapes liefert.
    """
    vocab_size = 100
    d_model = 64
    max_seq_len = 20
    batch_size = 3
    seq_len = 7

    cfg = EmbeddingConfig(
        vocab_size=vocab_size,
        d_model=d_model,
        max_seq_len=max_seq_len,
        dropout=0.0,
    )

    layer = TransformerEmbedding(cfg)

    # CPU-Test
    token_ids_cpu = torch.randint(
        0, vocab_size, (batch_size, seq_len), dtype=torch.long
    )
    out_cpu = layer(token_ids_cpu)
    assert out_cpu.shape == (batch_size, seq_len, d_model)

    # Optionaler CUDA-Test, nur wenn CUDA verfügbar ist
    if torch.cuda.is_available():
        device = torch.device("cuda")
        layer = layer.to(device)
        token_ids_gpu = token_ids_cpu.to(device)
        out_gpu = layer(token_ids_gpu)
        assert out_gpu.shape == (batch_size, seq_len, d_model)
        assert out_gpu.is_cuda
