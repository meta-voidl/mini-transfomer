"""
conftest.py

Pytest-Konfiguration für das Mini-Transformer-Projekt.

Ziel:
- Sicherstellen, dass das Projekt-Root-Verzeichnis auf sys.path liegt,
  damit Importe wie `from src.config import Config` funktionieren.

Pytest lädt conftest.py automatisch, du musst es nicht explizit importieren.
"""

import sys
from pathlib import Path

# Pfad zum Projekt-Root:
# tests/conftest.py -> tests/ -> Projekt-Root (eine Ebene höher)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Optional: Pfad zur Kontrolle ausgeben (kann bei Debug helfen)
# print(f"[conftest] PROJECT_ROOT = {PROJECT_ROOT}")

# Falls das Projekt-Root noch nicht in sys.path ist, fügen wir es vorne ein.
# Dadurch kann Python das Paket "src" finden (src ist ein Unterordner von PROJECT_ROOT).
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
