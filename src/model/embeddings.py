"""
embeddings.py

Hier definieren wir die Embedding-Schicht(en) für unseren Mini-Transformer:

- TokenEmbedding:
    wandelt Token-IDs (int) in dichte Vektoren der Dimension d_model um.
- PositionalEmbedding:
    liefert für jede Position in der Sequenz einen eigenen Positionsvektor.
- TransformerEmbedding:
    kombiniert Token- und Positions-Embeddings und wendet optional Dropout an.

Formate (Shapes):
    Eingabe:  token_ids: (batch_size, seq_len)
    Ausgabe:  embeddings: (batch_size, seq_len, d_model)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


@dataclass
class EmbeddingConfig:
    """
    Kleine Konfigurationsklasse nur für Embeddings.

    In der Praxis kannst du später auch direkt auf ModelConfig aus src.config
    zurückgreifen. Diese Klasse hilft beim Testen und Kapseln.
    """
    vocab_size: int
    d_model: int
    max_seq_len: int
    dropout: float = 0.1


class TokenEmbedding(nn.Module):
    """
    Einfache Token-Embedding-Schicht.

    Idee:
    - Jedes Token im Vokabular bekommt einen d_model-dimensionalen Vektor.
    - Intern nutzt PyTorch nn.Embedding, was letztlich einer Lookup-Tabelle
      entspricht.

    Eingabe:
        token_ids: LongTensor der Form (batch_size, seq_len)

    Ausgabe:
        embeddings: FloatTensor der Form (batch_size, seq_len, d_model)
    """

    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Führt den Lookup für die Token-IDs durch.

        token_ids: (batch_size, seq_len) vom Typ torch.long
        """
        # nn.Embedding gibt direkt Tensor der Form (batch_size, seq_len, d_model) zurück
        return self.embedding(token_ids)


class PositionalEmbedding(nn.Module):
    """
    Learned Positional Embeddings.

    Idee:
    - Für jede mögliche Position 0..max_seq_len-1 wird ein eigener
      d_model-dimensionaler Vektor gelernt.
    - Bei einer Eingabesequenz der Länge L holen wir uns einfach
      die ersten L Positionsvektoren und "broadcasten" sie über die Batch-Dimension.

    Eingabe:
        token_ids: (batch_size, seq_len) -> wir nutzen nur die seq_len-Info

    Ausgabe:
        pos_embeddings: (batch_size, seq_len, d_model)
    """

    def __init__(self, max_seq_len: int, d_model: int) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.embedding = nn.Embedding(max_seq_len, d_model)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        token_ids: (batch_size, seq_len) – wir verwenden die seq_len,
        um passende Positionsindizes zu erzeugen.

        Beispiel:
            seq_len = 4
            positions = [0, 1, 2, 3]
        """
        batch_size, seq_len = token_ids.shape

        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequenzlänge {seq_len} > max_seq_len {self.max_seq_len}. "
                "Bitte max_seq_len erhöhen oder kürzere Sequenzen verwenden."
            )

        # positions: (seq_len,) = [0, 1, 2, ..., seq_len-1]
        positions = torch.arange(seq_len, device=token_ids.device, dtype=torch.long)

        # pos_embeddings: (seq_len, d_model)
        pos_embeddings = self.embedding(positions)

        # Wir wollen Shape (batch_size, seq_len, d_model).
        # Dazu erweitern wir die Batch-Dimension (1, seq_len, d_model)
        # und wiederholen sie für jede Batch.
        pos_embeddings = pos_embeddings.unsqueeze(0).expand(batch_size, seq_len, -1)

        return pos_embeddings


class TransformerEmbedding(nn.Module):
    """
    Kombination aus Token-Embedding und Positional-Embedding.

    Typischer Ablauf in einem Transformer:
        embeddings = TokenEmbedding(token_ids) + PositionalEmbedding(token_ids)
        embeddings = Dropout(embeddings)

    Eingabe:
        token_ids: (batch_size, seq_len)

    Ausgabe:
        embeddings: (batch_size, seq_len, d_model)
    """

    def __init__(self, cfg: EmbeddingConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.token_embedding = TokenEmbedding(cfg.vocab_size, cfg.d_model)
        self.pos_embedding = PositionalEmbedding(cfg.max_seq_len, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Führt Token- und Positional-Embedding aus und summiert beide.

        token_ids: (batch_size, seq_len)
        """
        # Token-Embeddings: (batch_size, seq_len, d_model)
        tok_emb = self.token_embedding(token_ids)

        # Positions-Embeddings: (batch_size, seq_len, d_model)
        pos_emb = self.pos_embedding(token_ids)

        # Elementweise Addition:
        # Jedes Token bekommt seine Positionsinformation addiert.
        x = tok_emb + pos_emb

        # Dropout zur Regularisierung
        x = self.dropout(x)

        return x


def quick_demo() -> None:
    """
    Kleine Demo-Funktion, um Embeddings visuell/konzeptionell zu testen.

    Aufruf:
        python -m src.model.embeddings
    """
    batch_size = 2
    seq_len = 4
    vocab_size = 50
    d_model = 8
    max_seq_len = 16

    # Fake-Token-IDs im Bereich [0, vocab_size)
    token_ids = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, seq_len),
        dtype=torch.long,
    )

    cfg = EmbeddingConfig(
        vocab_size=vocab_size,
        d_model=d_model,
        max_seq_len=max_seq_len,
        dropout=0.1,
    )

    emb_layer = TransformerEmbedding(cfg)
    out = emb_layer(token_ids)

    print("[embeddings demo] token_ids shape:", token_ids.shape)
    print("[embeddings demo] output shape:", out.shape)


if __name__ == "__main__":
    quick_demo()
