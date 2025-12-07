"""
tokenizer.py

Sehr einfacher **Zeichen-basierter Tokenizer** (Char-Level) als Startpunkt.

Warum Char-Level?
- Der Code bleibt extrem einfach und transparent.
- Wir können Encode/Decode-Roundtrips leicht testen.
- Kein komplexer Algorithmus (wie BPE) nötig, um das Transformer-Grundprinzip
  zu verstehen.

Später kannst du diesen Tokenizer leicht gegen einen komplexeren
(Wort- oder BPE-Tokenizer) austauschen, ohne das Modell grundlegend zu ändern.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import Counter
from typing import Dict, List


# Wir definieren ein paar Sondertoken als normale Strings.
# Sie kommen IMMER zuerst im Vokabular.
PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"  # "begin of sequence"
EOS_TOKEN = "<eos>"  # "end of sequence"
UNK_TOKEN = "<unk>"  # "unknown"


@dataclass
class CharTokenizer:
    """
    Einfacher Tokenizer, der auf Ebene einzelner Zeichen arbeitet.

    Verwendung:
        tok = CharTokenizer()
        tok.fit(text)               # Vokabular aus Text bauen
        ids = tok.encode("hello")   # [<bos>, 'h', 'e', 'l', 'l', 'o', <eos>]
        text = tok.decode(ids)      # "hello"

    Attribute:
        stoi: Mapping von Zeichen -> Token-ID (string-to-index)
        itos: Mapping von Token-ID -> Zeichen (index-to-string)
    """

    # Mapping: Zeichen -> Index
    stoi: Dict[str, int] = field(default_factory=dict)
    # Mapping: Index -> Zeichen (Liste mit Länge = Vokabulargröße)
    itos: List[str] = field(default_factory=list)

    # IDs der Sondertoken (werden nach fit(...) gesetzt)
    pad_id: int | None = None
    bos_id: int | None = None
    eos_id: int | None = None
    unk_id: int | None = None

    def fit(
        self,
        text: str,
        max_vocab_size: int | None = None,
        min_freq: int = 1,
    ) -> None:
        """
        Baut das Vokabular aus einem gegebenen Text.

        Schritte:
        1. Zählt alle Zeichen (Counter)
        2. Filtert seltene Zeichen (freq < min_freq)
        3. Sortiert nach Häufigkeit (absteigend) und dann alphabetisch
        4. Fügt Sondertoken an den Anfang
        5. Erstellt stoi/itos

        Hinweis:
        - Char-Level bedeutet: jedes einzelne Zeichen (inkl. Leerzeichen, Punkt, etc.)
          wird ein Token, sofern häufig genug.
        """
        # Zeichenhäufigkeit zählen
        counter = Counter(text)

        # Seltene Zeichen entfernen
        # (z. B. exotische Unicode-Zeichen, die nur einmal vorkommen)
        filtered_chars = [
            (ch, freq) for ch, freq in counter.items() if freq >= min_freq
        ]

        # Nach Häufigkeit (absteigend), dann lexikographisch sortieren
        filtered_chars.sort(key=lambda x: (-x[1], x[0]))

        # Max. Vokabulargröße berücksichtigen (ohne Sondertoken)
        if max_vocab_size is not None:
            filtered_chars = filtered_chars[:max_vocab_size]

        # Basis-Vokabular: nur die Zeichen
        vocab_chars = [ch for ch, _ in filtered_chars]

        # Sondertoken an den Anfang setzen (Reihenfolge ist fix)
        all_tokens = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN] + vocab_chars

        # itos: Liste aller Token-Strings nach Index
        self.itos = list(all_tokens)

        # stoi: Mapping String -> Index
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

        # IDs der Sondertoken setzen
        self.pad_id = self.stoi[PAD_TOKEN]
        self.bos_id = self.stoi[BOS_TOKEN]
        self.eos_id = self.stoi[EOS_TOKEN]
        self.unk_id = self.stoi[UNK_TOKEN]

    @property
    def vocab_size(self) -> int:
        """Gibt die aktuelle Vokabulargröße zurück."""
        return len(self.itos)

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
    ) -> List[int]:
        """
        Wandelt einen String in eine Liste von Token-IDs um.

        - Unbekannte Zeichen erhalten die ID `unk_id`.
        - Wenn `add_special_tokens=True`, werden BOS/EOS-Token um den Text gelegt.

        Beispiel:
            text = "hi"
            ids  = [bos_id, id('h'), id('i'), eos_id]
        """
        if not self.stoi:
            raise RuntimeError(
                "Tokenizer ist noch nicht trainiert. "
                "Bitte zuerst `fit(text)` aufrufen."
            )

        # Normale Zeichen in IDs umwandeln
        ids = []
        for ch in text:
            if ch in self.stoi:
                ids.append(self.stoi[ch])
            else:
                ids.append(self.unk_id)

        # Sondertoken am Anfang/Ende hinzufügen (optional)
        if add_special_tokens:
            ids = [self.bos_id] + ids + [self.eos_id]

        return ids

    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = True,
    ) -> str:
        """
        Wandelt eine Liste von Token-IDs zurück in einen String.

        - Token-IDs werden über `itos` in Strings zurückgemappt.
        - Falls `skip_special_tokens=True`, werden PAD/BOS/EOS/UNK entfernt.
        - Unbekannte IDs (Out-of-Range) werden als '?' dargestellt.

        Achtung:
        - Bei Char-Level-Tokenisierung werden Zeichen einfach konkateniert.
        """
        if not self.itos:
            raise RuntimeError(
                "Tokenizer hat noch kein Vokabular. "
                "Bitte zuerst `fit(text)` aufrufen."
            )

        chars: List[str] = []
        for idx in token_ids:
            if 0 <= idx < len(self.itos):
                tok = self.itos[idx]
            else:
                tok = "?"  # Fallback für völlig ungültige IDs

            if skip_special_tokens and tok in {
                PAD_TOKEN,
                BOS_TOKEN,
                EOS_TOKEN,
                UNK_TOKEN,
            }:
                # Sondertoken überspringen
                continue

            # Normale Zeichen anhängen
            chars.append(tok)

        return "".join(chars)


def build_tokenizer_from_text(
    text: str,
    max_vocab_size: int | None = None,
    min_freq: int = 1,
) -> CharTokenizer:
    """
    Komfortfunktion: erzeugt einen CharTokenizer und trainiert ihn
    direkt auf dem gegebenen Text.

    Beispiel:
        tok = build_tokenizer_from_text(corpus_text)
    """
    tok = CharTokenizer()
    tok.fit(text, max_vocab_size=max_vocab_size, min_freq=min_freq)
    return tok


def quick_demo() -> None:
    """
    Kleine Demo, die nur ausgeführt wird, wenn `python -m src.tokenizer`
    gestartet wird.

    Sie zeigt:
    - Vokabulargröße
    - Encode/Decode eines Beispiels
    """
    example_text = "hello world!"

    tok = CharTokenizer()
    tok.fit(example_text)

    print("[tokenizer demo] Vokabulargröße:", tok.vocab_size)
    ids = tok.encode(example_text)
    print("[tokenizer demo] Token-IDs:", ids)
    decoded = tok.decode(ids)
    print("[tokenizer demo] Decoded:", repr(decoded))


if __name__ == "__main__":
    quick_demo()
