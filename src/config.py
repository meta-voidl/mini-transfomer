"""
config.py

Zentrale Konfigurationswerte für das Mini-Transformer-Projekt.

Die Idee:
- Alle wichtigen Hyperparameter (Modellgröße, Trainingseinstellungen, Pfade)
  sind an einer Stelle definiert.
- So kannst du leicht experimentieren, ohne an vielen Stellen im Code
  Änderungen machen zu müssen.
"""

from dataclasses import dataclass
from pathlib import Path
import torch


# Basisverzeichnis des Projekts (ausgehend von dieser Datei)
# Path(__file__).resolve() -> Pfad zu dieser Datei
# .parent.parent        -> geht zwei Ebenen hoch: src/ -> mini_transformer/
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class DataConfig:
    """Konfiguration für Datenpfade und grundlegende Datenparameter."""
    data_dir: Path = PROJECT_ROOT / "data"
    raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    processed_dir: Path = PROJECT_ROOT / "data" / "processed"

    raw_text_filename: str = "corpus.txt"

    max_seq_len: int = 128
    val_split: float = 0.1

    # NEU: kein mutable Default, nur None
    raw_text_sources: list[tuple[str, str]] | None = None


    max_seq_len: int = 128
    val_split: float = 0.1


@dataclass
class ModelConfig:
    """Hyperparameter für das Transformer-Modell."""

    # Vektor-Dimension der Embeddings und der internen Repräsentation
    d_model: int = 256

    # Anzahl der Self-Attention-Heads
    n_heads: int = 4

    # Anzahl der Decoder-Blöcke
    n_layers: int = 4

    # Dimension des Feed-Forward-Netzwerks
    # (typisch 4 * d_model bei vielen Transformern)
    d_ff: int = 4 * 256

    # Dropout-Rate für reguläre Dropout-Schichten
    dropout: float = 0.1

    # Maximale Sequenzlänge, muss mit DataConfig.max_seq_len harmonieren
    max_seq_len: int = 128

    # Platzhalter für Vokabulargröße – wird später gesetzt,
    # sobald der Tokenizer trainiert wurde
    vocab_size: int = 0


@dataclass
class TrainingConfig:
    """Hyperparameter für das Training."""
    # Anzahl Epochen
    num_epochs: int = 10

    # Batchgröße pro Schritt
    batch_size: int = 32

    # Lernrate
    learning_rate: float = 3e-4

    # Gewichtungszerfall für AdamW (leichte Regularisierung)
    weight_decay: float = 1e-2

    # Optional: Clipping der Gradienten-Norm zur Stabilisierung
    max_grad_norm: float = 1.0

    # Wie oft (in Schritten) soll logging erfolgen?
    log_interval: int = 100

    # Optional: Anzahl der warmup-Schritte für LR-Scheduler (0 = kein Warmup)
    warmup_steps: int = 0

    # Checkpoint-Verwaltung
    checkpoint_dir: Path = PROJECT_ROOT / "checkpoints"
    checkpoint_every_n_steps: int = 1000


@dataclass
class DeviceConfig:
    """Geräteeinstellungen (CPU oder GPU)."""

    # Automatische Wahl: 'cuda' falls verfügbar, sonst 'cpu'.
    # Die eigentliche Device-Instanz wird über get_device() erzeugt.
    preferred_device: str = "cuda"

    def get_device(self) -> torch.device:
        """
        Gibt ein torch.device-Objekt zurück, das im restlichen Code
        verwendet werden kann.

        Beispiel:
            device = device_config.get_device()
            model.to(device)
        """
        if self.preferred_device == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        # Falls CUDA nicht verfügbar ist, auf CPU zurückfallen
        return torch.device("cpu")


@dataclass
class Config:
    data: DataConfig = DataConfig(
        raw_text_sources=[
            ("https://www.gutenberg.org/ebooks/11.txt.utf-8", "alice.txt"),
            ("https://www.gutenberg.org/ebooks/1661.txt.utf-8", "sherlock.txt"),
            ("https://www.gutenberg.org/ebooks/120.txt.utf-8", "treasure_island.txt"),
            ("https://www.gutenberg.org/ebooks/1342.txt.utf-8", "pride_prejudice.txt"),
            ("https://www.gutenberg.org/ebooks/2701.txt.utf-8", "moby_dick.txt"),
            ("https://www.gutenberg.org/ebooks/84.txt.utf-8", "frankenstein.txt"),
        ]
    )
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    device: DeviceConfig = DeviceConfig()

def print_config(cfg: Config) -> None:
    """
    Hilfsfunktion, um die aktuelle Konfiguration lesbar auszugeben.

    Das ist nützlich, um schnell zu prüfen, ob die gewünschten Werte
    tatsächlich gesetzt sind (z. B. vor einem Training).
    """
    print("=== Mini-Transformer Konfiguration ===")
    print("\n[DataConfig]")
    print(cfg.data)

    print("\n[ModelConfig]")
    print(cfg.model)

    print("\n[TrainingConfig]")
    print(cfg.training)

    print("\n[DeviceConfig]")
    device = cfg.device.get_device()
    print(cfg.device)
    print(f"-> Effektiv verwendetes Device: {device}")


# Mini-Selbsttest, der nur ausgeführt wird, wenn diese Datei direkt gestartet wird.
# So kannst du mit `python -m src.config` oder `python src/config.py` schnell prüfen,
# ob Pfade und Device-Erkennung funktionieren.
if __name__ == "__main__":
    cfg = Config()
    print_config(cfg)
