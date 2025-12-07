"""
transformer.py

Voller Mini-Transformer (Decoder-only, GPT-ähnlich) auf Basis der bereits
implementierten Bausteine:

- Embeddings (Token + Position)
- Mehrere Transformer-Decoder-Blöcke
- Finale lineare Projektion auf Vokabulargröße

Aufgabe:
    Next-Token-Prediction (autoregressiv):

        Input:  Sequenz von Token-IDs (z. B. "hello worl")
        Output: Logits für jedes Token, inkl. Vorhersage für das nächste Token

Wichtige Shapes:
    B = batch_size
    T = seq_len
    D = d_model
    V = vocab_size

    input_ids: (B, T)        -- int64 Token-IDs
    embeddings: (B, T, D)
    logits:     (B, T, V)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch import nn

from src.model.embeddings import EmbeddingConfig, TransformerEmbedding
from src.model.transformer_block import TransformerBlock, TransformerBlockConfig


@dataclass
class TransformerConfig:
    """
    Konfiguration für den vollständigen Mini-Transformer.

    Typische Beispielwerte:
        d_model    = 256
        n_heads    = 4
        n_layers   = 4
        d_ff       = 4 * d_model
        max_seq_len = 128
        dropout    = 0.1
    """
    vocab_size: int
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 1024
    max_seq_len: int = 128
    dropout: float = 0.1
    # Optional: ID des PAD-Tokens (wichtig für Maskierung)
    pad_token_id: Optional[int] = None
    # Ob causale Maske (kein Blick in Zukunft) verwendet werden soll
    use_causal_mask: bool = True
    # Ob lm_head und Token-Embedding-Gewichte geteilt werden sollen
    tie_embeddings: bool = True


class MiniTransformer(nn.Module):
    """
    Kleiner Decoder-only Transformer für Next-Token-Prediction.

    Aufbau:
        1) Embeddings:
            - Token-Embedding
            - Positional-Embedding
        2) N x TransformerBlock
        3) LayerNorm (Final-Norm)
        4) Lineare Projektion: (D -> vocab_size)

    Forward:
        input_ids: (B, T)   -> logits: (B, T, V)

    Optional:
        - Rückgabe der Attention-Gewichte aller Blöcke (für Visualisierung)
    """

    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()

        self.cfg = cfg
        self.vocab_size = cfg.vocab_size
        self.d_model = cfg.d_model

        # --- 1) Embeddings ---
        emb_cfg = EmbeddingConfig(
            vocab_size=cfg.vocab_size,
            d_model=cfg.d_model,
            max_seq_len=cfg.max_seq_len,
            dropout=cfg.dropout,
        )
        self.embeddings = TransformerEmbedding(emb_cfg)

        # --- 2) Stapel von Transformer-Blöcken ---
        block_cfg = TransformerBlockConfig(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            d_ff=cfg.d_ff,
            dropout=cfg.dropout,
            use_causal_mask=cfg.use_causal_mask,
        )
        self.blocks = nn.ModuleList(
            [TransformerBlock(block_cfg) for _ in range(cfg.n_layers)]
        )

        # --- 3) Finale LayerNorm ---
        self.ln_f = nn.LayerNorm(cfg.d_model)

        # --- 4) LM-Head: Projektion auf Vokabulargröße ---
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Optional: Embedding-Gewichte und LM-Head-Gewichte teilen
        # (typischer Trick in GPT/Transformern -> weniger Parameter, etwas bessere Performance)
        if cfg.tie_embeddings:
            # Zugriff auf die Token-Embedding-Matrix
            token_embedding_weight = self.embeddings.token_embedding.embedding.weight
            self.lm_head.weight = token_embedding_weight

        # Initialisierung (einfach gehalten)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """
        Einfache Gewichtsinitialisierung für Linear- und Embedding-Schichten.

        Hinweis:
        - Dies ist nicht super "fancy", aber für ein kleines Lernprojekt ausreichend.
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _build_padding_mask(
        self, input_ids: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """
        Erzeugt eine Padding-Maske aus input_ids, falls pad_token_id gesetzt ist.

        input_ids: (B, T)

        Rückgabe:
            mask: (B, 1, 1, T) bool, broadcastbar auf (B, H, T, T)

        Masken-Konvention:
            True  = wird MASKIERT (z. B. PAD-Token).
        """
        if self.cfg.pad_token_id is None:
            return None

        # pad_positions: True an Stellen, wo PAD-Token steht
        pad_positions = input_ids == self.cfg.pad_token_id  # (B, T)

        # Wir wollen eine Maske, die für alle Query-Positionen T dieselben
        # Keys maskiert -> Shape (B, 1, 1, T)
        mask = pad_positions.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, T)
        return mask

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        return_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Forward-Pass durch den kompletten Transformer.

        Parameter:
            input_ids: (B, T), dtype=torch.long
            return_attentions:
                Wenn True, wird eine Liste der Attention-Matrizen aller Blöcke
                zurückgegeben. Jede hat Shape (B, H, T, T).

        Rückgabe:
            logits: (B, T, V)
            attentions (optional): List[Tensor] der Länge n_layers
        """
        B, T = input_ids.shape

        if T > self.cfg.max_seq_len:
            raise ValueError(
                f"Sequenzlänge {T} > max_seq_len {self.cfg.max_seq_len}. "
                "Bitte max_seq_len erhöhen oder kürzere Sequenzen verwenden."
            )

        # 1) Padding-Maske aus input_ids erzeugen (falls konfiguriert)
        padding_mask = self._build_padding_mask(input_ids)

        # 2) Token + Positions-Embeddings
        x = self.embeddings(input_ids)  # (B, T, D)

        # 3) Transformer-Blöcke
        all_attentions: List[torch.Tensor] = []

        for block in self.blocks:
            x, attn = block(
                x,
                attention_mask=padding_mask,
                return_attention=return_attentions,
            )
            if return_attentions and attn is not None:
                all_attentions.append(attn)

        # 4) Finale LayerNorm
        x = self.ln_f(x)  # (B, T, D)

        # 5) LM-Head: Logits über Vokabular
        logits = self.lm_head(x)  # (B, T, V)

        if return_attentions:
            return logits, all_attentions
        else:
            return logits, None

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Sehr einfache Greedy-Generation (ohne fancy Sampling/Top-k etc.).

        Ablauf:
            - Wiederholt:
                - Forward-Pass mit aktuellem Kontext
                - Nimmt Logits des letzten Tokens
                - Wählt argmax (oder softes Sampling via Temperatur)
                - Hängt Token an Sequenz an

        Parameter:
            input_ids: (B, T) Startsequenz
            max_new_tokens: wie viele neue Tokens generiert werden sollen
            temperature: >0, skaliert Logits vor Softmax

        Rückgabe:
            generated_ids: (B, T + max_new_tokens)
        """
        self.eval()

        B, T = input_ids.shape
        device = next(self.parameters()).device
        generated = input_ids.to(device)

        for _ in range(max_new_tokens):
            # Falls Sequenz zu lang wird, auf max_seq_len beschneiden
            if generated.shape[1] > self.cfg.max_seq_len:
                generated = generated[:, -self.cfg.max_seq_len :]

            logits, _ = self(generated, return_attentions=False)  # (B, T_cur, V)
            last_logits = logits[:, -1, :]  # (B, V)

            if temperature != 1.0:
                last_logits = last_logits / temperature

            probs = torch.softmax(last_logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # An Sequenz anhängen
            generated = torch.cat([generated, next_token], dim=1)  # (B, T_cur+1)

        return generated


def quick_demo() -> None:
    """
    Kleine Demo für einen Forward-Pass durch den MiniTransformer.

    Aufruf:
        python -m src.model.transformer
    """
    B, T = 2, 5
    vocab_size = 100
    cfg = TransformerConfig(
        vocab_size=vocab_size,
        d_model=32,
        n_heads=4,
        n_layers=2,
        d_ff=64,
        max_seq_len=16,
        dropout=0.1,
        pad_token_id=0,
        tie_embeddings=True,
    )

    model = MiniTransformer(cfg)

    # Fake-Input-IDs im Bereich [0, vocab_size)
    input_ids = torch.randint(0, vocab_size, (B, T), dtype=torch.long)

    logits, attentions = model(input_ids, return_attentions=True)

    print("[transformer demo] input_ids shape:", input_ids.shape)
    print("[transformer demo] logits shape:", logits.shape)
    print("[transformer demo] num attention layers:", len(attentions))
    print("[transformer demo] attention[0] shape:", attentions[0].shape)


if __name__ == "__main__":
    quick_demo()
