"""
Einfache Tests für dataset.py

Hinweis:
- Beim allerersten Lauf kann dieser Test einen kurzen HTTP-Download auslösen,
  um den Beispieltext herunterzuladen.
"""

from src.config import Config
from src import dataset


def test_raw_text_is_non_empty():
    """
    Prüft, dass der Rohltext erfolgreich geladen werden kann und
    nicht leer ist.
    """
    cfg = Config()
    data_cfg = cfg.data

    raw_text = dataset.load_raw_text(data_cfg)

    assert isinstance(raw_text, str)
    # Wir erwarten, dass der Text eine gewisse Mindestlänge hat.
    assert len(raw_text.strip()) > 1000


def test_cleaned_text_is_non_empty():
    """
    Prüft, dass der bereinigte Text erzeugt und geladen werden kann
    und ebenfalls nicht leer ist.
    """
    cfg = Config()
    data_cfg = cfg.data

    cleaned_text = dataset.load_cleaned_text(data_cfg)

    assert isinstance(cleaned_text, str)
    assert len(cleaned_text.strip()) > 1000
