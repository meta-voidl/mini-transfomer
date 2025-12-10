# Mini Transformer – von Grund auf in PyTorch

Dieses Projekt implementiert einen **kleinen, didaktischen Decoder-only Transformer** (GPT-ähnlich) in **PyTorch**.

Ziel ist **Verständnis**, nicht State-of-the-Art-Performance:

- Wie funktionieren **Token-Embeddings** und **Positionsembeddings**?
- Wie berechnet man **Multi-Head Self-Attention** (inkl. Maskierung)?
- Wie sehen **Residual-Verbindungen** und **Layer Normalization** im Code aus?
- Wie läuft eine einfache **Training-Loop** (Loss, Optimizer, Backprop, Gradient Clipping)?

Das komplette Projekt ist so klein gehalten, dass man es in wenigen Stunden lesen und nachvollziehen kann – und auf einer einzelnen **RTX 4080** in **wenigen Stunden trainieren** kann (auf einem kleinen Textkorpus).

---

## Projektstruktur (geplant)

```text
mini_transformer/
├─ data/
│  ├─ raw/                # Rohdaten (z. B. heruntergeladene Textdateien)
│  ├─ processed/          # Vorgefertigte/serialisierte Datensätze (.pt, .pkl)
│  └─ README.md           # Kurze Notizen zum Datensatz
├─ notebooks/
│  ├─ 01_explore_data.ipynb          # Erkundung des Textkorpus
│  ├─ 02_visualize_attention.ipynb   # Visualisierung Self-Attention
│  └─ 03_training_curves.ipynb       # Lernkurven, Metriken
├─ src/
│  ├─ __init__.py
│  ├─ config.py            # Zentrale Hyperparameter & Pfade
│  ├─ dataset.py           # Dataset-/Dataloader-Logik
│  ├─ tokenizer.py         # Einfache Tokenisierung & Vokabularbau
│  ├─ model/
│  │  ├─ __init__.py
│  │  ├─ embeddings.py     # Token- & Positionsembeddings
│  │  ├─ attention.py      # Scaled Dot-Product & Multi-Head Attention
│  │  ├─ transformer_block.py  # Ein Decoder-Block
│  │  └─ transformer.py    # Voller Mini-Transformer (GPT-ähnlich)
│  ├─ training/
│  │  ├─ __init__.py
│  │  ├─ train_loop.py     # Training-, Eval- und Logging-Logik
│  │  └─ utils.py          # Hilfsfunktionen (Seed, Checkpoints, etc.)
│  ├─ visualization.py     # Plots und Attention-Heatmaps
│  └─ main.py              # Kommandozeilen-Einstiegspunkt
├─ tests/
│  ├─ test_tokenizer.py
│  ├─ test_attention.py
│  ├─ test_transformer_block.py
│  ├─ test_transformer_end_to_end.py
│  └─ conftest.py
├─ requirements.txt
├─ .gitignore
└─ README.md
```
## Hardware Konfiguraton & IDE

Ubuntu 22.04.5 LTS with a high-performance hardware setup including an MSI motherboard, an Intel i7-14700F CPU, 32GB of RAM, and an NVIDIA RTX 4080 GPU. 
PyCharm is the IDE.


### some useful scripts 
start training
python -m src.main   --train   --epochs 5   --batch-size 32   --max-vocab-size 256   --logdir runs/12books_bigmodel_6heads_e5
tensorboard --logdir runs
#### dataset generation
python -m src.dataset 





