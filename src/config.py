"""Configuración global: rutas y parámetros de entrenamiento validados."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
DATA_RAW = RAIZ / "data" / "raw"
DATA_PROC = RAIZ / "data" / "processed"
MODELS = RAIZ / "models"
REPORTS = RAIZ / "reports"
FIGURES = REPORTS / "figures"
SEED = 42

COLOR_POS = "#3987E5"
COLOR_NEG = "#E66767"

ALGORITMOS = ("sgd", "logreg", "svm", "nb")
PENALIZACIONES = ("l2", "l1", "elasticnet")


def asegurar_carpetas() -> None:
    for c in (DATA_RAW, DATA_PROC, MODELS, REPORTS, FIGURES):
        c.mkdir(parents=True, exist_ok=True)


@dataclass
class TrainConfig:
    """Hiperparámetros configurables (desde la API/interfaz)."""

    n_por_clase: int = 2754          # máx. reseñas por clase en entrenamiento
    ngram_max: int = 3
    max_features: int = 10000
    min_df: int = 5
    algoritmo: str = "sgd"
    epocas: int = 50
    alpha: float = 1e-4
    penalizacion: str = "l2"
    paciencia: int = 5
    umbral: float = 0.5
    comparar_modelos: bool = False
    random_state: int = SEED

    def __post_init__(self) -> None:
        errores = []
        if not 100 <= self.n_por_clase <= 80000:
            errores.append("n_por_clase debe estar entre 100 y 80000")
        if self.ngram_max not in (1, 2, 3):
            errores.append("ngram_max debe ser 1, 2 o 3")
        if not 1000 <= self.max_features <= 50000:
            errores.append("max_features debe estar entre 1000 y 50000")
        if not 1 <= self.min_df <= 20:
            errores.append("min_df debe estar entre 1 y 20")
        if self.algoritmo not in ALGORITMOS:
            errores.append(f"algoritmo debe ser uno de {ALGORITMOS}")
        if not 1 <= self.epocas <= 100:
            errores.append("epocas debe estar entre 1 y 100")
        if not 1e-8 <= self.alpha <= 10:
            errores.append("alpha debe estar entre 1e-8 y 10")
        if self.penalizacion not in PENALIZACIONES:
            errores.append(f"penalizacion debe ser una de {PENALIZACIONES}")
        if not 1 <= self.paciencia <= 50:
            errores.append("paciencia debe estar entre 1 y 50")
        if not 0.05 <= self.umbral <= 0.95:
            errores.append("umbral debe estar entre 0.05 y 0.95")
        if errores:
            raise ValueError("; ".join(errores))

    def a_dict(self) -> dict:
        return asdict(self)


RANGOS = {
    "n_por_clase": (100, 80000), "ngram_max": (1, 3), "max_features": (1000, 50000),
    "min_df": (1, 20), "epocas": (1, 100), "alpha": (1e-8, 10.0),
    "paciencia": (1, 50), "umbral": (0.05, 0.95),
}
