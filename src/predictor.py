"""Inferencia y explicación por término."""
from __future__ import annotations

import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import sklearn

from .config import MODELS
from .preprocessing import limpiar_texto
from .train import pesos_modelo, probabilidad_positivo


class ModeloNoDisponible(FileNotFoundError):
    """No hay un modelo entrenado en disco."""


class Predictor:
    def __init__(self, carpeta: Path = MODELS) -> None:
        f_mod, f_vec = Path(carpeta) / "modelo.joblib", Path(carpeta) / "vectorizador.joblib"
        if not f_mod.exists() or not f_vec.exists():
            raise ModeloNoDisponible("No hay modelo entrenado. Entrena uno primero (python -m src.train o POST /train).")
        paquete = joblib.load(f_mod)
        self.modelo = paquete["modelo"]
        self.umbral = float(paquete["umbral"])
        self.config = paquete["config"]
        self.fecha = paquete["fecha"]
        self.metricas_resumen = paquete["metricas_resumen"]
        self.algoritmo = paquete.get("algoritmo", "sgd")
        self.vec = joblib.load(f_vec)
        self.terminos = np.array(self.vec.get_feature_names_out())
        self.pesos = pesos_modelo(self.modelo)
        if paquete.get("version_sklearn") != sklearn.__version__:
            warnings.warn(f"Modelo entrenado con scikit-learn {paquete.get('version_sklearn')} "
                          f"pero hay {sklearn.__version__} instalado.")

    def info(self) -> dict:
        return {"fecha": self.fecha, "algoritmo": self.algoritmo, "config": self.config,
                "umbral": self.umbral, "metricas_resumen": self.metricas_resumen}

    def predecir(self, texto: str, umbral: float | None = None, top: int = 8) -> dict:
        if not isinstance(texto, str) or not texto.strip():
            raise ValueError("El texto no puede estar vacío.")
        u = self.umbral if umbral is None else float(umbral)
        t0 = time.perf_counter()
        limpio = limpiar_texto(texto)
        if not limpio:
            raise ValueError("El texto no contiene palabras analizables (solo números o símbolos).")
        X = self.vec.transform([limpio])
        p = float(probabilidad_positivo(self.modelo, X)[0])
        contrib = []
        if self.pesos is not None:
            idx = X.indices
            vals = X.data * self.pesos[idx]
            orden = np.argsort(-np.abs(vals))[:top]
            contrib = [{"termino": str(self.terminos[idx[i]]), "contribucion": float(vals[i])} for i in orden]
        positivo = p >= u
        return {"etiqueta": "Positivo" if positivo else "Negativo", "prob_positivo": p,
                "confianza": p if positivo else 1 - p, "umbral": u, "top_terminos": contrib,
                "texto_limpio": limpio, "ms": round((time.perf_counter() - t0) * 1000, 2)}

    def predecir_lote(self, textos: list[str], umbral: float | None = None) -> list[dict]:
        res = []
        for t in textos:
            try:
                res.append(self.predecir(t, umbral, top=5))
            except ValueError as e:
                res.append({"error": str(e)})
        return res
