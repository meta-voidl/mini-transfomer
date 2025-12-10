# src/notebook_setup.py

from __future__ import annotations

import sys
from pathlib import Path


def setup_notebook():
    """
    Stellt sicher, dass das Projekt-Root im sys.path ist
    und gibt (cfg, PROJECT_ROOT, project_root) zurück.
    """
    # Projektwurzel: eine Ebene über src/
    project_root = Path(__file__).resolve().parent.parent

    # Falls noch nicht im sys.path, hinzufügen
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # Jetzt können wir ganz normal aus src importieren
    from src.config import Config, PROJECT_ROOT

    cfg = Config()
    return cfg, PROJECT_ROOT, project_root
