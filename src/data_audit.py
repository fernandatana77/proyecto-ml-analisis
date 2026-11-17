"""Auditoría de calidad del dataset crudo (sección 5.1)."""
from __future__ import annotations

import json
import re

import pandas as pd

from .config import REPORTS, asegurar_carpetas
from .data_loader import cargar_crudo

RE_HTML = re.compile(r"<[^>]+>")
RE_URL = re.compile(r"https?://\S+|www\.\S+")
RE_MOJI = re.compile(r"Ã.|Â.|�")
RE_EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿]")


def _percentiles(s: pd.Series) -> dict:
    q = s.quantile([0.01, 0.25, 0.5, 0.75, 0.99])
    return {"min": float(s.min()), "p1": float(q[0.01]), "p25": float(q[0.25]),
            "p50": float(q[0.5]), "p75": float(q[0.75]), "p99": float(q[0.99]),
            "max": float(s.max())}


def auditar(df: pd.DataFrame, muestra_idioma: int = 2000) -> dict:
    texto = df["cuerpo"].fillna("").astype(str)
    norm = texto.str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    palabras = texto.str.split().str.len()
    rep = {
        "total_filas": len(df),
        "nulos_por_columna": {c: int(v) for c, v in df.isna().sum().items()},
        "textos_vacios": int((norm == "").sum()),
        "duplicados_exactos": int(texto.duplicated().sum()),
        "duplicados_normalizados": int(norm.duplicated().sum()),
        "distribucion_estrellas": {str(k): int(v) for k, v in df["estrellas"].value_counts().sort_index().items()},
        "longitud_caracteres": _percentiles(texto.str.len()),
        "longitud_palabras": _percentiles(palabras),
        "con_mojibake": int(texto.str.contains(RE_MOJI).sum()),
        "con_html": int(texto.str.contains(RE_HTML).sum()),
        "con_url": int(texto.str.contains(RE_URL).sum()),
        "con_emoji": int(texto.str.contains(RE_EMOJI).sum()),
        "muy_cortas_lt3_palabras": int((palabras < 3).sum()),
        "muy_largas_gt_p99": int((palabras > palabras.quantile(0.99)).sum()),
        "neutrales_3_estrellas": int((df["estrellas"] == 3).sum()),
    }
    if "categoria" in df:
        rep["distribucion_categorias"] = {k: int(v) for k, v in df["categoria"].value_counts().items()}
    if "idioma" in df:
        rep["idioma_declarado"] = {k: int(v) for k, v in df["idioma"].value_counts().items()}
    else:
        rep["idioma_declarado"] = "columna no disponible en este dataset"
    try:
        from langdetect import DetectorFactory, detect
        DetectorFactory.seed = 42
        m = texto[texto.str.len() > 20].sample(min(muestra_idioma, len(texto)), random_state=42)
        det = []
        for t in m:
            try:
                det.append(detect(t))
            except Exception:
                det.append("??")
        rep["idioma_detectado_muestra"] = {k: int(v) for k, v in pd.Series(det).value_counts().items()}
        rep["muestra_idioma_n"] = len(m)
    except ImportError:
        rep["idioma_detectado_muestra"] = "langdetect no instalado"
    return rep


def main() -> dict:
    asegurar_carpetas()
    df = cargar_crudo()
    rep = auditar(df)
    destino = REPORTS / "auditoria_datos.json"
    # conserva el embudo si ya existe (lo escribe preprocessing)
    if destino.exists():
        previo = json.loads(destino.read_text(encoding="utf-8"))
        if "embudo_limpieza" in previo:
            rep["embudo_limpieza"] = previo["embudo_limpieza"]
    destino.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in rep.items() if k != "distribucion_categorias"}, indent=2, ensure_ascii=False))
    print(f"\nGuardado en {destino}")
    return rep


if __name__ == "__main__":
    main()
