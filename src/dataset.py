
""""

## 2️⃣ `src/dataset.py`

Lege unter `mini_transformer/src/` die Datei `dataset.py` an:

```python
"
dataset.py

Mini-Modul für:
- Download eines kleinen Beispieltextes (gemeinfreier Text / Project Gutenberg)
- erweitert um mehere Beispieltexte
- einfache Bereinigung (Preprocessing)
- Laden des Rohltexts als Python-String

WICHTIG:
Dies ist noch NICHT das fertige PyTorch-Dataset für das Training.
Das eigentliche `TextDataset` bauen wir später in einem eigenen Schritt.
"""

from __future__ import annotations

import re
import urllib.request
from pathlib import Path

from src.config import DataConfig, PROJECT_ROOT


# Konstante URL eines Beispieltexts.
# Hier verwenden wir einen Project-Gutenberg-Text (Alice im Wunderland).
# Du kannst diese URL später leicht durch eine andere ersetzen.
GUTENBERG_URL = "https://www.gutenberg.org/cache/epub/11/pg11.txt"

# Dateiname für die bereinigte Version des Textes
CLEANED_FILENAME = "corpus_clean.txt"


def ensure_data_directories(data_cfg: DataConfig) -> None:
    """
    Stellt sicher, dass die Verzeichnisse data/, data/raw/ und data/processed/
    existieren. Wenn nicht, werden sie erstellt.

    Das ist hilfreich, damit andere Funktionen sich darauf verlassen können,
    dass die Pfade existieren.
    """
    data_cfg.data_dir.mkdir(parents=True, exist_ok=True)
    data_cfg.raw_dir.mkdir(parents=True, exist_ok=True)
    data_cfg.processed_dir.mkdir(parents=True, exist_ok=True)


def download_corpus_if_needed(data_cfg: DataConfig) -> list[Path]:
    """
    Lädt einen oder mehrere Beispieltexte herunter, falls sie noch nicht existieren.

    Rückgabe:
        Liste von Pfaden zu Rohtextdateien unter data/raw/.

    Fälle:
    - data_cfg.raw_text_sources is None:
        -> altes Verhalten: ein einzelner Text von GUTENBERG_URL nach raw_text_filename
    - data_cfg.raw_text_sources ist eine Liste:
        -> für jede (url, filename) wird data/raw/filename angelegt (falls fehlt)
    """
    ensure_data_directories(data_cfg)

    raw_paths: list[Path] = []

    # Fall 1: keine expliziten Quellen angegeben -> Fallback auf Einzel-URL
    if not data_cfg.raw_text_sources:
        raw_path = data_cfg.raw_dir / data_cfg.raw_text_filename
        if not raw_path.exists():
            print(f"[dataset] Lade Text von {GUTENBERG_URL} ...")
            try:
                with urllib.request.urlopen(GUTENBERG_URL) as response:
                    raw_bytes = response.read()
            except Exception as e:
                raise RuntimeError(f"Fehler beim Download des Korpus: {e}") from e

            text = raw_bytes.decode("utf-8", errors="replace")
            raw_path.write_text(text, encoding="utf-8")
            print(f"[dataset] Rohtext gespeichert unter: {raw_path}")

        raw_paths.append(raw_path)
        return raw_paths

    # Fall 2: mehrere Quellen
    for url, filename in data_cfg.raw_text_sources:
        raw_path = data_cfg.raw_dir / filename
        if raw_path.exists():
            raw_paths.append(raw_path)
            continue

        print(f"[dataset] Lade Text von {url} ...")
        try:
            with urllib.request.urlopen(url) as response:
                raw_bytes = response.read()
        except Exception as e:
            raise RuntimeError(f"Fehler beim Download des Korpus von {url}: {e}") from e

        text = raw_bytes.decode("utf-8", errors="replace")
        raw_path.write_text(text, encoding="utf-8")
        print(f"[dataset] Rohtext gespeichert unter: {raw_path}")
        raw_paths.append(raw_path)

    return raw_paths

def load_raw_text(data_cfg: DataConfig) -> str:
    """
    Lädt einen oder mehrere Rohtexte als einen großen String.

    - Falls mehrere Dateien existieren (raw_text_sources gesetzt),
      werden sie in der Reihenfolge der Liste mit zwei Zeilenumbrüchen
      dazwischen konkateniert.

    Rückgabe:
        text (str): kompletter zusammengesetzter Rohltext.
    """
    raw_paths = download_corpus_if_needed(data_cfg)

    texts: list[str] = []
    for p in raw_paths:
        t = p.read_text(encoding="utf-8")
        if t.strip():
            texts.append(t)

    if not texts:
        raise ValueError("Die geladenen Rohtexte sind alle leer. Bitte Quellen prüfen.")

    # Mit Abstand zusammenkleben, damit der Übergang nicht verschmiert
    full_text = "\n\n" + ("\n\n".join(texts)) + "\n\n"

    return full_text



def basic_clean_text(text: str) -> str:
    """
    Sehr einfache Textbereinigung (Preprocessing).

    Ziele:
    - alles in Kleinbuchstaben (vereinfachte Vokabulargröße)
    - Zeilenumbrüche normalisieren
    - exotische Zeichen entfernen oder durch Leerzeichen ersetzen

    Diese Funktion ist bewusst simpel gehalten, damit man sie leicht versteht
    und bei Bedarf anpassen/erweitern kann.

    Schritte:
    1. Lowercasing
    2. \r\n -> \n Normalisierung
    3. Entfernen nicht gewünschter Zeichen per Regex
    4. Mehrfache Leerzeichen reduzieren
    """
    # 1) Kleinbuchstaben
    text = text.lower()

    # 2) Zeilenumbrüche normalisieren
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 3) Nur eine einfache Auswahl an Zeichen behalten:
    #    - Kleinbuchstaben a-z
    #    - Ziffern 0-9
    #    - grundlegende Satzzeichen und Leerzeichen
    #    Alles andere wird durch Leerzeichen ersetzt.
    text = re.sub(r"[^a-z0-9 .,;:!?'\n-]", " ", text)

    # 4) Mehrfache Leerzeichen zusammenfassen
    text = re.sub(r"[ ]+", " ", text)

    # Optional: führende/trailing Leerzeichen pro Zeile entfernen
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)

    return text


def save_cleaned_text(raw_text: str, data_cfg: DataConfig) -> Path:
    """
    Wendet `basic_clean_text` auf den Rohltext an und speichert das Ergebnis
    in `data/processed/CLEANED_FILENAME`.

    Gibt den Pfad zur bereinigten Datei zurück.
    """
    ensure_data_directories(data_cfg)

    cleaned = basic_clean_text(raw_text)
    processed_path = data_cfg.processed_dir / CLEANED_FILENAME
    processed_path.write_text(cleaned, encoding="utf-8")

    print(f"[dataset] Bereinigter Text gespeichert unter: {processed_path}")
    return processed_path


def load_cleaned_text(data_cfg: DataConfig) -> str:
    """
    Lädt den bereits bereinigten Text aus `data/processed/CLEANED_FILENAME`,
    falls vorhanden. Wenn nicht, wird der Rohltext geladen, bereinigt und
    zuerst gespeichert.

    Rückgabe:
        text (str): bereinigter Text.
    """
    ensure_data_directories(data_cfg)
    processed_path = data_cfg.processed_dir / CLEANED_FILENAME

    if not processed_path.exists():
        # Kein bereinigter Text vorhanden -> neu erzeugen
        print("[dataset] Kein bereinigter Text gefunden. Erzeuge neuen ...")
        raw_text = load_raw_text(data_cfg)
        processed_path = save_cleaned_text(raw_text, data_cfg)

    text = processed_path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(
            "Der bereinigte Text ist leer. "
            "Bitte Preprocessing-Funktion bzw. Quelle überprüfen."
        )

    return text


def quick_sanity_check() -> None:
    """
    Kleine Hilfsfunktion, um dataset.py schnell zu testen.

    Wird im __main__-Block aufgerufen.
    """
    from src.config import Config  # Import lokal, um Zirkularitäten zu vermeiden

    cfg = Config()
    data_cfg = cfg.data

    print("[sanity_check] Projektwurzel:", PROJECT_ROOT)
    print("[sanity_check] Datenverzeichnis:", data_cfg.data_dir)

    # 1) Rohltext laden (inkl. Download beim ersten Mal)
    raw_text = load_raw_text(data_cfg)
    print(f"[sanity_check] Länge Rohltext: {len(raw_text)} Zeichen")

    # 2) Bereinigten Text erzeugen/lesen
    cleaned_text = load_cleaned_text(data_cfg)
    print(f"[sanity_check] Länge bereinigter Text: {len(cleaned_text)} Zeichen")

    # 3) Einen kleinen Ausschnitt anzeigen
    print("\n[sanity_check] Ausschnitt bereinigter Text:")
    print(cleaned_text[:500])


if __name__ == "__main__":
    # Wenn du `python -m src.dataset` im Projekt-Root aufrufst,
    # wird diese kleine Selbstprüfung ausgeführt.
    quick_sanity_check()


# ---------------------------------------------------------
# Teil 2: PyTorch-Dataset für Next-Token-Prediction
# ---------------------------------------------------------

import torch
from torch.utils.data import Dataset, DataLoader


class TextDataset(Dataset):
    """
    Ein einfaches PyTorch-Dataset für Next-Token-Prediction.

    Es erwartet:
    - einen bereits TOKENISIERTEN Text (Liste von Token-IDs)
    - eine feste Sequenzlänge (seq_len)

    Aus den Token-IDs werden Sliding Windows erzeugt.

    Beispiel:
        ids = [10, 11, 12, 13, 14]
        seq_len = 3

        Samples:
            input : [10,11,12]   target: [11,12,13]
            input : [11,12,13]   target: [12,13,14]
    """

    def __init__(self, token_ids: list[int], seq_len: int):
        assert seq_len >= 2, "seq_len muss >= 2 sein"
        self.token_ids = token_ids
        self.seq_len = seq_len

        # Anzahl möglicher Fenster
        self.num_samples = len(token_ids) - seq_len
        if self.num_samples <= 0:
            raise ValueError(
                f"Zu wenig Token ({len(token_ids)}) für seq_len={seq_len}"
            )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        Liefert:
        - input_ids  : Tokens von idx bis idx+seq_len-1
        - target_ids : Tokens von idx+1 bis idx+seq_len
        """
        chunk = self.token_ids[idx : idx + self.seq_len + 1]

        input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
        target_ids = torch.tensor(chunk[1:], dtype=torch.long)

        return input_ids, target_ids


def create_dataloader(
    token_ids: list[int],
    seq_len: int,
    batch_size: int,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
):
    """
    Bequeme Helper-Funktion zum Erzeugen eines Dataloaders.

    Wird später in main/train_loop verwendet.
    """
    dataset = TextDataset(token_ids, seq_len)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )


# ---------------------------------------------------------
# Kleiner Selbsttest
# ---------------------------------------------------------
def _quick_dataset_demo():
    """
    Kurzer Selbsttest für TextDataset.
    """

    example_ids = [1, 2, 3, 4, 5, 6]
    seq_len = 4

    ds = TextDataset(example_ids, seq_len)

    print(f"[demo] Dataset length = {len(ds)} Samples")
    for i in range(len(ds)):
        inp, tgt = ds[i]
        print(f"[demo] {i}: input={inp.tolist()} -> target={tgt.tolist()}")


if __name__ == "__main__":
    # Dieser Block wird zusätzlich zur vorherigen Cleaning-Demo ausgeführt.
    print("\n=== Quick Dataset Demo ===")
    _quick_dataset_demo()
