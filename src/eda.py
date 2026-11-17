"""Análisis exploratorio: figuras PNG a 300 DPI en reports/figures (sección 6)."""
from __future__ import annotations

import json
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import COLOR_NEG, COLOR_POS, DATA_PROC, FIGURES, REPORTS, asegurar_carpetas

BG, FG = "#0E1117", "#F0EFEC"
COLS_CORR = ["longitud_caracteres", "num_palabras", "num_exclamaciones", "num_interrogaciones",
             "proporcion_mayusculas", "num_negaciones", "estrellas", "sentimiento"]


def _estilo() -> None:
    plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
                         "text.color": FG, "axes.labelcolor": FG, "xtick.color": FG, "ytick.color": FG,
                         "axes.edgecolor": "#555", "font.size": 10})


def _guardar(fig, nombre: str, carpeta=FIGURES) -> str:
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta = carpeta / nombre
    fig.savefig(ruta, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return nombre


def cargar_procesado() -> dict[str, pd.DataFrame]:
    return {n: pd.read_csv(DATA_PROC / f"{n}.csv") for n in ("train", "val", "test")}


def fig_embudo() -> str:
    emb = json.loads((REPORTS / "auditoria_datos.json").read_text(encoding="utf-8"))["embudo_limpieza"]
    emb = [e for e in emb if e["paso"] != "train_tras_balanceo"]
    fig, ax = plt.subplots(figsize=(8, 4))
    pasos = [e["paso"].replace("_", " ") for e in emb]
    vals = [e["filas"] for e in emb]
    ax.barh(pasos[::-1], vals[::-1], color=COLOR_POS)
    for i, v in enumerate(vals[::-1]):
        ax.text(v, i, f" {v:,}", va="center")
    ax.set_xlim(0, max(vals) * 1.15)
    ax.set_title("Embudo de limpieza (registros)")
    return _guardar(fig, "eda_1_embudo_limpieza.png")


def fig_balance(d: dict) -> str:
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(3)
    neg = [(d[n]["sentimiento"] == 0).sum() for n in d]
    pos = [(d[n]["sentimiento"] == 1).sum() for n in d]
    ax.bar(x - 0.2, neg, 0.4, color=COLOR_NEG, label="Negativo")
    ax.bar(x + 0.2, pos, 0.4, color=COLOR_POS, label="Positivo")
    ax.set_xticks(x, list(d))
    ax.set_title("Balance de clases por partición")
    ax.set_ylabel("Reseñas")
    ax.legend()
    return _guardar(fig, "eda_2_balance_clases.png")


def fig_estrellas(df: pd.DataFrame) -> str:
    c = df["estrellas"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(c.index.astype(str), c.values, color=[COLOR_NEG if e <= 2 else COLOR_POS for e in c.index])
    ax.set_title("Distribución por estrellas")
    ax.set_xlabel("Estrellas")
    ax.set_ylabel("Reseñas")
    return _guardar(fig, "eda_3_estrellas.png")


def fig_categorias(df: pd.DataFrame) -> str:
    top = df["categoria"].value_counts().head(15).index
    t = pd.crosstab(df["categoria"], df["sentimiento"]).loc[top].rename(columns={0: "Negativo", 1: "Positivo"})
    t = t.loc[t.sum(axis=1).sort_values().index]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(t.index, t["Negativo"], color=COLOR_NEG, label="Negativo")
    ax.barh(t.index, t["Positivo"], left=t["Negativo"], color=COLOR_POS, label="Positivo")
    ax.set_title("Top 15 categorías de producto (Positivo/Negativo)")
    ax.legend()
    return _guardar(fig, "eda_4_categorias.png")


def fig_longitud(df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(7, 4))
    lim = df["num_palabras"].quantile(0.99)
    bins = np.linspace(0, lim, 40)
    ax.hist(df.loc[df.sentimiento == 0, "num_palabras"].clip(upper=lim), bins=bins, alpha=0.7, color=COLOR_NEG, label="Negativo")
    ax.hist(df.loc[df.sentimiento == 1, "num_palabras"].clip(upper=lim), bins=bins, alpha=0.7, color=COLOR_POS, label="Positivo")
    ax.set_title("Longitud de reseñas (palabras, hasta p99)")
    ax.set_xlabel("Palabras")
    ax.legend()
    return _guardar(fig, "eda_5_longitud.png")


def matriz_correlacion(df: pd.DataFrame, extra: pd.Series | None = None) -> pd.DataFrame:
    x = df[COLS_CORR].copy()
    if extra is not None:
        x["p_positivo_modelo"] = extra.values
    return x.corr(method="pearson")


def fig_correlacion(df: pd.DataFrame, extra: pd.Series | None = None, nombre: str = "eda_6_correlacion.png", carpeta=FIGURES) -> str:
    c = matriz_correlacion(df, extra)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(c.values, cmap="RdBu", vmin=-1, vmax=1)
    ax.set_xticks(range(len(c)), c.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(c)), c.index)
    for i in range(len(c)):
        for j in range(len(c)):
            v = c.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="black" if abs(v) < 0.6 else "white")
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title("Matriz de correlación de Pearson")
    return _guardar(fig, nombre, carpeta)


def terminos_frecuentes(df: pd.DataFrame, clase: int, n: int = 15, ngram: int = 1) -> list[tuple[str, int]]:
    cnt: Counter = Counter()
    for t in df.loc[df.sentimiento == clase, "texto_limpio"].dropna():
        w = t.split()
        cnt.update(" ".join(w[i:i + ngram]) for i in range(len(w) - ngram + 1))
    return cnt.most_common(n)


def fig_terminos(df: pd.DataFrame) -> str:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for r, ng in enumerate((1, 2)):
        for c, (clase, col, nom) in enumerate(((0, COLOR_NEG, "Negativo"), (1, COLOR_POS, "Positivo"))):
            tf = terminos_frecuentes(df, clase, 15, ng)[::-1]
            axes[r, c].barh([t for t, _ in tf], [v for _, v in tf], color=col)
            axes[r, c].set_title(f"{nom} — {'unigramas' if ng == 1 else 'bigramas'}")
    fig.tight_layout()
    return _guardar(fig, "eda_7_terminos_frecuentes.png")


def main() -> list[str]:
    asegurar_carpetas()
    _estilo()
    d = cargar_procesado()
    todo = pd.concat(d.values(), ignore_index=True)
    figs = [fig_embudo(), fig_balance(d), fig_estrellas(todo), fig_categorias(todo),
            fig_longitud(todo), fig_correlacion(todo), fig_terminos(d["train"])]
    corr = matriz_correlacion(todo)["sentimiento"].drop("sentimiento").sort_values()
    print("Figuras generadas en", FIGURES)
    for f in figs:
        print("  -", f)
    print("\nCorrelación de cada variable con 'sentimiento':")
    print(corr.round(3).to_string())
    print("\nTop unigramas Negativo:", terminos_frecuentes(d["train"], 0, 8))
    print("Top unigramas Positivo:", terminos_frecuentes(d["train"], 1, 8))
    return figs


if __name__ == "__main__":
    main()
