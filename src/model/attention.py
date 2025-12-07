"""
attention.py

Implementierung von:
- scaled dot-product attention
- multi-head self-attention (mit optionaler causaler Maske)

Notation / Shapes (sehr wichtig beim Verstehen):

    B  = batch_size
    T  = seq_len (sequence length)
    D  = d_model (Modeldimension)
    H  = n_heads (Anzahl der Attention-Heads)
    Dh = D // H (Dimension pro Head)

    x:          (B, T, D)
    q, k, v:    (B, H, T, Dh)
    scores:     (B, H, T, T)    # "Wie stark Token i auf Token j schaut"
    attn:       (B, H, T, T)    # Softmax-normalisierte Scores
    out_heads:  (B, H, T, Dh)
    out:        (B, T, D)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

import torch
from torch import nn


@dataclass
class AttentionConfig:
    """
    Konfiguration für Multi-Head-Attention.

    In der Praxis kannst du diese Werte aus ModelConfig in src.config
    übernehmen; diese Klasse ist vor allem für Tests/Lesbarkeit.
    """
    d_model: int
    n_heads: int
    dropout: float = 0.0
    # Wenn True: causale Maske (kein Blick in die Zukunft) wird genutzt.
    use_causal_mask: bool = True

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) muss durch n_heads ({self.n_heads}) "
                f"teilbar sein, damit alle Heads gleich groß sind."
            )


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Implementiert die "scaled dot-product attention".

    Formeln (vereinfacht):

        scores = (q @ k^T) / sqrt(Dh)
        scores = scores + mask   (optional)
        attn   = softmax(scores, dim=-1)
        out    = attn @ v

    Shapes:
        q, k, v: (B, H, T, Dh)
        mask:    (B, H, T, T) oder broadcastbar auf diese Form
        scores:  (B, H, T, T)
        attn:    (B, H, T, T)
        out:     (B, H, T, Dh)

    Masken-Konvention:
        - Wir verwenden eine boolsche Maske, bei der
            mask == True   -> Eintrag wird MASKIERT (nicht sichtbar)
        - Diese True-Einträge werden vor dem Softmax auf -inf gesetzt.
    """
    B, H, T, Dh = q.shape
    # k: (B, H, T, Dh) -> (B, H, Dh, T) für Matrizenmultiplikation
    k_t = k.transpose(-2, -1)

    # Rohscores: (B, H, T, T)
    scores = torch.matmul(q, k_t) / (Dh**0.5)

    if mask is not None:
        # mask: True an Stellen, die "unsichtbar" sein sollen
        # -> wir setzen diese scores auf -inf, damit Softmax -> 0 daraus macht.
        scores = scores.masked_fill(mask, float("-inf"))

    # Softmax über die "Key/Value"-Dimension (letzte Achse)
    attn = torch.softmax(scores, dim=-1)

    # Sicherheit: NaNs vermeiden, falls irgendwas komplett auf -inf stand
    attn = torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)

    # Attention-Gewichtete Summe der Values: (B, H, T, Dh)
    out = torch.matmul(attn, v)

    return out, attn


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention für eine Eingabesequenz x.

    Schritte im Überblick:
        1. Linear-Projektion von x auf q, k, v.
        2. Aufteilen der D-Dimension in H Heads (D -> H x Dh).
        3. scaled_dot_product_attention auf jedem Head.
        4. Zusammenfügen der Heads zurück zu D.
        5. Abschließende lineare Projektion + Dropout.

    Eingabe:
        x: (B, T, D)

    Rückgabe:
        out: (B, T, D)
        (optional) attn_weights: (B, H, T, T) – für Visualisierung/Debugging
    """

    def __init__(self, cfg: AttentionConfig) -> None:
        super().__init__()

        self.cfg = cfg
        self.d_model = cfg.d_model
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads

        # W_q, W_k, W_v: jeweils (D -> D)
        self.q_proj = nn.Linear(self.d_model, self.d_model)
        self.k_proj = nn.Linear(self.d_model, self.d_model)
        self.v_proj = nn.Linear(self.d_model, self.d_model)

        # Ausgabeprojektion (nach dem Zusammenfügen der Heads)
        self.out_proj = nn.Linear(self.d_model, self.d_model)

        # Dropout auf den Attention-Gewichten (üblich in vielen Implementierungen)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        # Dropout auf der Projektion
        self.out_dropout = nn.Dropout(cfg.dropout)

    def _shape_to_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Hilfsfunktion: wandelt Tensor von (B, T, D) in (B, H, T, Dh) um.

        Wir "splitten" die letzte Dimension D in (H, Dh) und permutieren
        die Achsen so, dass H direkt nach B kommt.
        """
        B, T, D = x.shape
        H = self.n_heads
        Dh = self.head_dim

        # Schritt 1: view(B, T, H, Dh)
        x = x.view(B, T, H, Dh)
        # Schritt 2: permute -> (B, H, T, Dh)
        x = x.permute(0, 2, 1, 3)
        return x

    def _shape_from_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Hilfsfunktion: (B, H, T, Dh) -> (B, T, D)

        Wir "mergen" die Head-Dimension H und die Dh-Dimension wieder
        zurück in eine einzelne D-Dimension.
        """
        B, H, T, Dh = x.shape
        D = H * Dh

        # permute -> (B, T, H, Dh)
        x = x.permute(0, 2, 1, 3).contiguous()
        # view -> (B, T, D)
        x = x.view(B, T, D)
        return x

    @staticmethod
    def _build_causal_mask(T: int, device: torch.device) -> torch.Tensor:
        """
        Erzeugt eine causale Maske der Form (1, 1, T, T).

        Konvention:
            mask[b, h, i, j] == True  -> Eintrag (i,j) wird MASKIERT,
                                         d. h. Token i darf NICHT auf Token j schauen.

        Für causale Self-Attention wollen wir verhindern, dass Token i
        auf zukünftige Tokens j > i schaut:

            j > i -> mask = True
            j <= i -> mask = False
        """
        # Oberes Dreieck (j > i) = 1/True, Diagonale und darunter = 0/False
        mask = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)
        # Shape (T, T) -> (1, 1, T, T) für Broadcast auf (B, H, T, T)
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(
        self,
        x: torch.Tensor,
        *,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Führt Multi-Head Self-Attention auf Eingabe x aus.

        Parameter:
            x: (B, T, D)
            attention_mask:
                Optional zusätzliche Maske, broadcastbar auf (B, H, T, T).
                True = wird MASKIERT (unsichtbar).
                Kann z. B. für Padding genutzt werden.
            return_attention:
                Wenn True, werden außerdem die Attention-Gewichte zurückgegeben.

        Rückgabe:
            out: (B, T, D)
            attn_weights (optional): (B, H, T, T)
        """
        B, T, D = x.shape
        device = x.device

        # 1) lineare Projektionen auf q, k, v
        #    jeweils (B, T, D)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 2) in Heads aufteilen -> (B, H, T, Dh)
        q = self._shape_to_heads(q)
        k = self._shape_to_heads(k)
        v = self._shape_to_heads(v)

        # 3) Masken vorbereiten
        #    - causale Maske (j > i) falls gewünscht
        #    - zusätzliche attention_mask optional (z. B. für Padding)
        mask: Optional[torch.Tensor] = None

        if self.cfg.use_causal_mask:
            causal = self._build_causal_mask(T, device=device)  # (1, 1, T, T)
            mask = causal

        if attention_mask is not None:
            # Wir erwarten boolsche Maske, die broadcastbar ist.
            # Kombination: "oder" – wenn eine Maske True sagt, wird maskiert.
            if mask is None:
                mask = attention_mask
            else:
                mask = mask | attention_mask

        # 4) scaled dot-product attention
        attn_out, attn_weights = scaled_dot_product_attention(q, k, v, mask=mask)

        # optional Dropout auf den Attention-Gewichten
        attn_weights = self.attn_dropout(attn_weights)

        # 5) Heads wieder zusammenfügen -> (B, T, D)
        out = self._shape_from_heads(attn_out)

        # 6) Ausgabeprojektion + Dropout
        out = self.out_proj(out)
        out = self.out_dropout(out)

        if return_attention:
            return out, attn_weights
        else:
            return out, None


def quick_demo() -> None:
    """
    Kleine Demo für einen Forward-Pass durch Multi-Head Self-Attention.

    Aufruf:
        python -m src.model.attention
    """
    B, T, D = 2, 4, 8
    H = 2

    x = torch.randn(B, T, D)

    cfg = AttentionConfig(d_model=D, n_heads=H, dropout=0.0, use_causal_mask=True)
    layer = MultiHeadSelfAttention(cfg)

    out, attn = layer(x, return_attention=True)

    print("[attention demo] input shape:", x.shape)
    print("[attention demo] output shape:", out.shape)
    print("[attention demo] attn shape:", attn.shape)


if __name__ == "__main__":
    quick_demo()
