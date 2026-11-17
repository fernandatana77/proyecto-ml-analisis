"""Gráficas del modelo (PNG a 300 DPI). Positivo = azul, Negativo = rojo."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

from .config import COLOR_NEG, COLOR_POS, FIGURES

BG, FG = "#0E1117", "#F0EFEC"


def _estilo() -> None:
    plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
                         "text.color": FG, "axes.labelcolor": FG, "xtick.color": FG, "ytick.color": FG,
                         "axes.edgecolor": "#555", "font.size": 10})


def _guardar(fig, carpeta: Path, nombre: str) -> str:
    carpeta.mkdir(parents=True, exist_ok=True)
    fig.savefig(carpeta / nombre, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return nombre


def fig_epocas(hist: pd.DataFrame, mejor: int, carpeta: Path = FIGURES) -> str:
    _estilo()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (m, tit) in zip(axes, (("logloss", "Pérdida (log-loss)"), ("accuracy", "Accuracy"), ("f1", "F1"))):
        ax.plot(hist["epoca"], hist[f"{m}_train"], color=COLOR_NEG, label="Entrenamiento")
        ax.plot(hist["epoca"], hist[f"{m}_val"], color=COLOR_POS, label="Validación")
        ax.axvline(mejor, color="#C3C2B7", ls=":", label=f"Mejor época ({mejor})")
        ax.set_title(tit)
        ax.set_xlabel("Época")
        ax.legend()
    fig.tight_layout()
    return _guardar(fig, carpeta, "1_evolucion_epocas.png")


def fig_confusion(y, pred, carpeta: Path = FIGURES) -> str:
    _estilo()
    cm = confusion_matrix(y, pred, labels=[0, 1])
    pct = cm / cm.sum(axis=1, keepdims=True) * 100
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
    nombres = ["Negativo", "Positivo"]
    ax.set_xticks([0, 1], nombres)
    ax.set_yticks([0, 1], nombres)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}\n({pct[i, j]:.1f}%)", ha="center", va="center",
                    color="white" if pct[i, j] > 50 else "black", fontsize=12)
    ax.set_title("Matriz de confusión (prueba)")
    return _guardar(fig, carpeta, "2_matriz_confusion.png")


def fig_roc(y, p, auc: float, carpeta: Path = FIGURES) -> str:
    _estilo()
    fpr, tpr, _ = roc_curve(y, p)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(fpr, tpr, color=COLOR_POS, label=f"Modelo (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], ":", color="#C3C2B7", label="Azar")
    ax.set_xlabel("Tasa de falsos positivos")
    ax.set_ylabel("Tasa de verdaderos positivos")
    ax.set_title("Curva ROC (prueba)")
    ax.legend()
    return _guardar(fig, carpeta, "3_curva_roc.png")


def fig_palabras(terminos: np.ndarray, pesos: np.ndarray, n: int = 20, carpeta: Path = FIGURES) -> str:
    _estilo()
    idx = np.argsort(pesos)
    neg, pos = idx[:n], idx[-n:]
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.5))
    axes[0].barh(terminos[neg][::-1], pesos[neg][::-1], color=COLOR_NEG)
    axes[0].set_title("Empujan a Negativo")
    axes[1].barh(terminos[pos], pesos[pos], color=COLOR_POS)
    axes[1].set_title("Empujan a Positivo")
    fig.suptitle("Términos más influyentes (coeficientes aprendidos)")
    fig.tight_layout()
    return _guardar(fig, carpeta, "4_palabras_influyentes.png")


def fig_comparacion(tabla: pd.DataFrame, carpeta: Path = FIGURES) -> str:
    _estilo()
    t = tabla.sort_values("f1_prueba")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(t["modelo"], t["f1_prueba"], color=COLOR_POS)
    for i, v in enumerate(t["f1_prueba"]):
        ax.text(v, i, f" {v:.3f}", va="center")
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("F1 en prueba")
    ax.set_title("Comparación de modelos")
    return _guardar(fig, carpeta, "6_comparacion_modelos.png")


def fig_pr(y, p, carpeta: Path = FIGURES) -> str:
    _estilo()
    pr, rc, th = precision_recall_curve(y, p)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(th, pr[:-1], color=COLOR_POS, label="Precisión")
    ax.plot(th, rc[:-1], color=COLOR_NEG, label="Recall")
    ax.set_xlabel("Umbral de decisión")
    ax.set_title("Precisión y Recall vs. umbral (clase Positivo, prueba)")
    ax.legend()
    return _guardar(fig, carpeta, "7_precision_recall_umbral.png")


def fig_calibracion(y, p, carpeta: Path = FIGURES) -> str:
    _estilo()
    frac, med = calibration_curve(y, p, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(med, frac, "o-", color=COLOR_POS, label="Modelo")
    ax.plot([0, 1], [0, 1], ":", color="#C3C2B7", label="Calibración perfecta")
    ax.set_xlabel("Probabilidad predicha")
    ax.set_ylabel("Fracción real de positivos")
    ax.set_title("Curva de calibración (prueba)")
    ax.legend()
    return _guardar(fig, carpeta, "curva_calibracion.png")
