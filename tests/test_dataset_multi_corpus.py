# tests/test_dataset_multi_corpus.py

from pathlib import Path

from src.config import DataConfig
from src.dataset import (
    load_raw_text,
    save_cleaned_text,
    load_cleaned_text,
    basic_clean_text,
)


def _make_data_cfg(tmp_path) -> DataConfig:
    """
    Hilfsfunktion: Erzeugt eine DataConfig, die komplett in tmp_path lebt.
    """
    data_dir = tmp_path / "data"
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"

    cfg = DataConfig(
        data_dir=data_dir,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        raw_text_filename="ignored.txt",
        raw_text_sources=[
            ("http://example.com/book1.txt", "book1.txt"),
            ("http://example.com/book2.txt", "book2.txt"),
        ],
    )
    return cfg


def test_load_raw_text_concatenates_multiple_files(tmp_path):
    data_cfg = _make_data_cfg(tmp_path)

    # Rohverzeichnisse anlegen und Dummy-Dateien schreiben
    data_cfg.raw_dir.mkdir(parents=True, exist_ok=True)

    (data_cfg.raw_dir / "book1.txt").write_text("Hello from book1.", encoding="utf-8")
    (data_cfg.raw_dir / "book2.txt").write_text("And greetings from book2!", encoding="utf-8")

    # Sollte beide Dateien laden und zusammenfügen
    text = load_raw_text(data_cfg)

    assert "Hello from book1." in text
    assert "And greetings from book2!" in text
    # Reihenfolge muss stimmen
    assert text.index("Hello from book1.") < text.index("And greetings from book2!")


def test_save_and_load_cleaned_text_multi_corpus(tmp_path):
    data_cfg = _make_data_cfg(tmp_path)

    data_cfg.raw_dir.mkdir(parents=True, exist_ok=True)
    data_cfg.processed_dir.mkdir(parents=True, exist_ok=True)

    # Inhalt so wählen, dass basic_clean_text etwas sichtbar macht
    (data_cfg.raw_dir / "book1.txt").write_text("HELLO, WORLD!!!", encoding="utf-8")
    (data_cfg.raw_dir / "book2.txt").write_text("2nd BOOK?", encoding="utf-8")

    # load_cleaned_text -> triggert intern load_raw_text + save_cleaned_text
    cleaned = load_cleaned_text(data_cfg)

    # basic_clean_text macht alles lower case und lässt Zahlen und Satzzeichen zu
    assert "hello, world!" in cleaned
    assert "2nd book?" in cleaned

    # Sicherstellen, dass die bereinigte Datei existiert
    processed_file = data_cfg.processed_dir / "corpus_clean.txt"
    assert processed_file.exists()
    assert processed_file.read_text(encoding="utf-8") == cleaned
