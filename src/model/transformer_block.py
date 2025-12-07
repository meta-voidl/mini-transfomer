"""
transformer_block.py

Ein einzelner Decoder-Block unseres Mini-Transformers (Pre-LN-Variante):

    x --> LayerNorm --> Multi-Head Self-Attention --> Dropout --> + x (Residual)
      --> LayerNorm --> Feed-Forward (MLP)       --> Dropout --> + x (Residual)

Shapes:
    Eingabe:  x: (B, T, D)
    Ausgabe:  x': (B, T, D)

B = batch_size
T = seq_len
D = d_model
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn

from src.model.attention import AttentionConfig, MultiHeadSelfAttention


@dataclass
class TransformerBlockConfig:
    """
    Konfigurationsparameter für einen Transformer-Decoder-Block.

    Typische Werte:
        d_model = 256 oder 384
        n_heads = 4 oder 8
        d_ff    = 4 * d_model
        dropout = 0.1
    """
    d_model: int
    n_heads: int
    d_ff: int
    dropout: float = 0.1
    use_causal_mask: bool = True  # Decoder-Block -> in die Zukunft schauen verboten


class FeedForward(nn.Module):
    """
    Einfaches 2-Layer-Feed-Forward-Netzwerk (position-wise MLP).

    Formel:
        FFN(x) = max(0, x W1 + b1) W2 + b2

    Dabei wird dieselbe MLP für jedes Token in der Sequenz unabhängig
    angewendet (daher "position-wise").
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),             # Aktivierung (könnte auch ReLU sein)
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    Ein einzelner Transformer-Decoder-Block (Pre-LayerNorm).

    Aufbau:
        1. x_norm1 = LayerNorm(x)
           attn_out = MultiHeadSelfAttention(x_norm1)
           x = x + Dropout(attn_out)                # Residual 1

        2. x_norm2 = LayerNorm(x)
           ff_out  = FeedForward(x_norm2)
           x = x + Dropout(ff_out)                  # Residual 2

    Vorteile von Pre-LN:
        - Stabilisierter Gradientfluss, trainiert in der Praxis oft
          robuster als die ursprüngliche Post-LN-Variante.

    Eingabe:
        x: (B, T, D)

    Ausgabe:
        x: (B, T, D)

    Optional können die Attention-Gewichte für Visualisierung/Debugging
    zurückgegeben werden.
    """

    def __init__(self, cfg: TransformerBlockConfig) -> None:
        super().__init__()

        self.cfg = cfg
        d_model = cfg.d_model
        d_ff = cfg.d_ff
        dropout = cfg.dropout

        # LayerNorms (arbeiten über die letzte Dimension D)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        # Multi-Head Self-Attention
        attn_cfg = AttentionConfig(
            d_model=d_model,
            n_heads=cfg.n_heads,
            dropout=dropout,
            use_causal_mask=cfg.use_causal_mask,
        )
        self.self_attn = MultiHeadSelfAttention(attn_cfg)

        # Feed-Forward-Netzwerk
        self.ff = FeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)

        # Dropout auf den Residual-Zweigen
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        *,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward-Pass durch den Block.

        Parameter:
            x: (B, T, D)
            attention_mask:
                optionale boolsche Maske, broadcastbar auf (B, H, T, T)
                True = wird maskiert (z. B. Padding-Token).
            return_attention:
                Wenn True, wird zusätzlich die Attention-Matrix zurückgegeben.

        Rückgabe:
            x_out: (B, T, D)
            attn_weights (optional): (B, H, T, T)
        """
        # --- 1) Self-Attention-Zweig ---
        # Pre-LN: erst normalisieren, dann Attention
        x_norm1 = self.ln1(x)

        attn_out, attn_weights = self.self_attn(
            x_norm1,
            attention_mask=attention_mask,
            return_attention=return_attention,
        )
        # Residual-Verbindung + Dropout
        x = x + self.dropout(attn_out)

        # --- 2) Feed-Forward-Zweig ---
        x_norm2 = self.ln2(x)
        ff_out = self.ff(x_norm2)
        x = x + self.dropout(ff_out)

        return x, attn_weights if return_attention else None


def quick_demo() -> None:
    """
    Kleine Demo: ein Forward-Pass durch einen TransformerBlock.

    Aufruf:
        python -m src.model.transformer_block
    """
    B, T, D = 2, 5, 16
    n_heads = 4
    d_ff = 64

    x = torch.randn(B, T, D)

    cfg = TransformerBlockConfig(
        d_model=D,
        n_heads=n_heads,
        d_ff=d_ff,
        dropout=0.1,
        use_causal_mask=True,
    )

    block = TransformerBlock(cfg)
    out, attn = block(x, return_attention=True)

    print("[block demo] input shape:", x.shape)
    print("[block demo] output shape:", out.shape)
    print("[block demo] attention shape:", attn.shape)


if __name__ == "__main__":
    quick_demo()
