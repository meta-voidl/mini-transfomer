# Daten für den Mini-Transformer

Dieses Projekt verwendet als Beispielkorpus einen **gemeinfreien Text**
von [Project Gutenberg](https://www.gutenberg.org/).

Standardmäßig wird ein einzelnes englisches Buch verwendet, das klein genug ist,
um schnell damit zu experimentieren (z. B. *Alice's Adventures in Wonderland*).

## Verzeichnisstruktur

```text
data/
├─ raw/          # Rohdaten (Original-Download, z. B. "corpus.txt")
├─ processed/    # Vorverarbeitete Daten (bereinigter Text, Tensor-Dateien, ...)
└─ README.md
