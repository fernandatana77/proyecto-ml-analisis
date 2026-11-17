"""Carga de CSV crudos y normalización de nombres de columnas."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import DATA_RAW

# Alias aceptados -> nombre interno
ALIAS = {
    "stars": "estrellas", "review_body": "cuerpo", "reviewtext": "cuerpo",
    "review_title": "titulo", "language": "idioma", "product_category": "categoria",
    "sentiment": "sentimiento_original",
}


def cargar_crudo(carpeta: Path = DATA_RAW) -> pd.DataFrame:
    """Lee y concatena todos los CSV de `carpeta`. Falla si faltan estrellas o texto."""
    archivos = sorted(Path(carpeta).glob("*.csv"))
    if not archivos:
        raise FileNotFoundError(f"No hay archivos CSV en {carpeta}")
    partes = []
    for a in archivos:
        df = pd.read_csv(a)
        df.columns = [ALIAS.get(c.strip().lower(), c.strip().lower()) for c in df.columns]
        faltan = [c for c in ("estrellas", "cuerpo") if c not in df.columns]
        if faltan:
            raise ValueError(f"{a.name}: faltan columnas obligatorias {faltan} "
                             f"(se esperaba 'stars' y 'review_body'/'reviewText')")
        # split oficial si el nombre del archivo lo indica
        nombre = a.stem.lower()
        df["split_origen"] = next((s for s in ("train", "validation", "test") if s in nombre), "unico")
        partes.append(df)
    return pd.concat(partes, ignore_index=True)
