"""
Tests für den CharTokenizer in src/tokenizer.py
"""

import pytest

from src.tokenizer import (
    CharTokenizer,
    build_tokenizer_from_text,
    PAD_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
    UNK_TOKEN,
)


def test_fit_builds_non_empty_vocab():
    """
    Ein Aufruf von fit(...) sollte ein nicht-leeres Vokabular erzeugen,
    das mindestens die Sondertoken enthält.
    """
    text = "abcabc"
    tok = CharTokenizer()
    tok.fit(text)

    # Vokabular sollte mindestens 4 (Sondertoken) + 3 (a,b,c) Einträge haben
    assert tok.vocab_size >= 7
    assert PAD_TOKEN in tok.stoi
    assert BOS_TOKEN in tok.stoi
    assert EOS_TOKEN in tok.stoi
    assert UNK_TOKEN in tok.stoi


def test_encode_decode_roundtrip_simple():
    """
    Für einen Text, der nur bekannte Zeichen enthält, sollte
    decode(encode(text)) ungefähr den Originaltext ergeben.

    Hinweis:
    - Da wir bei decode() standardmäßig Sondertoken überspringen,
      sollten sie im Ergebnis nicht auftauchen.
    """
    text = "hello world"

    tok = build_tokenizer_from_text(text)
    ids = tok.encode(text)  # enthält BOS/EOS
    decoded = tok.decode(ids)  # skip_special_tokens=True per Default

    assert decoded == text


def test_unknown_character_maps_to_unk():
    """
    Zeichen, die beim Training (fit) nicht im Text vorkamen, sollen
    bei encode(...) auf UNK gemappt werden.
    """
    # Fit nur auf a,b,c
    base_text = "abcabc"
    tok = build_tokenizer_from_text(base_text)

    # 'x' kam nie vor -> sollte auf unk_id gemappt werden
    encoded = tok.encode("x", add_special_tokens=False)
    assert len(encoded) == 1
    assert encoded[0] == tok.unk_id


def test_decode_skips_special_tokens_by_default():
    """
    decode() sollte standardmäßig die Sondertoken nicht im Text auftauchen lassen.
    """
    text = "hi"

    tok = build_tokenizer_from_text(text)
    ids = tok.encode(text, add_special_tokens=True)

    # ids beginnt mit bos_id und endet mit eos_id
    assert ids[0] == tok.bos_id
    assert ids[-1] == tok.eos_id

    decoded = tok.decode(ids)

    # In der decodierten Version sollten BOS/EOS verschwunden sein
    assert decoded == text


def test_error_when_encoding_without_fit():
    """
    Wenn encode() aufgerufen wird, bevor fit() erfolgt ist, sollte
    ein sinnvoller Fehler geworfen werden.
    """
    tok = CharTokenizer()
    with pytest.raises(RuntimeError):
        tok.encode("test")
