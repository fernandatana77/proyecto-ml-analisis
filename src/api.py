"""Backend FastAPI: entrenamiento en segundo plano, métricas, dataset y predicción."""
from __future__ import annotations

import json
import threading
import uuid

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from . import train as train_mod
from .config import ALGORITMOS, DATA_PROC, PENALIZACIONES, RANGOS, TrainConfig
from .predictor import ModeloNoDisponible, Predictor

app = FastAPI(title="Análisis de sentimientos en reseñas (ES)", version="1.0",
              description="API para entrenar, evaluar y consultar el modelo de sentimientos.")
app.add_middleware(CORSMiddleware, allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
                   allow_methods=["*"], allow_headers=["*"])


class Estado:
    """Estado compartido del servidor (modelo cargado y trabajo de entrenamiento)."""

    def __init__(self) -> None:
        self.rutas = train_mod.Rutas()
        self.datos: tuple | None = None          # inyección para pruebas
        self.predictor: Predictor | None = None
        self.lock = threading.Lock()
        self.job: dict = {"estado": "inactivo"}

    def cargar_predictor(self) -> None:
        try:
            self.predictor = Predictor(self.rutas.modelos)
        except ModeloNoDisponible:
            self.predictor = None


ESTADO = Estado()
ESTADO.cargar_predictor()


def configurar(rutas: train_mod.Rutas | None = None, datos: tuple | None = None) -> None:
    """Cambia rutas/datos (usado por las pruebas) y recarga el predictor."""
    if rutas is not None:
        ESTADO.rutas = rutas
    ESTADO.datos = datos
    ESTADO.job = {"estado": "inactivo"}
    ESTADO.cargar_predictor()


# ---------- modelos Pydantic ----------
class TrainIn(BaseModel):
    n_por_clase: int = 2754
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


class PredictIn(BaseModel):
    texto: str = Field(..., min_length=1, max_length=5000)
    umbral: float = Field(0.5, ge=0.05, le=0.95)


class BatchIn(BaseModel):
    textos: list[str] = Field(..., min_length=1, max_length=500)
    umbral: float = Field(0.5, ge=0.05, le=0.95)


def _predictor() -> Predictor:
    if ESTADO.predictor is None:
        raise HTTPException(404, "No hay un modelo entrenado. Entrena uno desde la página Entrenamiento o POST /train.")
    return ESTADO.predictor


def _leer_json(nombre: str):
    f = ESTADO.rutas.reportes / nombre
    if not f.exists():
        return None
    return json.loads(f.read_text(encoding="utf-8"))


# ---------- estado ----------
@app.get("/health", tags=["estado"])
def health() -> dict:
    return {"estado": "ok", "modelo_cargado": ESTADO.predictor is not None,
            "entrenando": ESTADO.job.get("estado") == "en_curso"}


@app.get("/config/default", tags=["entrenamiento"])
def config_default() -> dict:
    return {"defaults": TrainConfig().a_dict(), "rangos": {k: list(v) for k, v in RANGOS.items()},
            "algoritmos": list(ALGORITMOS), "penalizaciones": list(PENALIZACIONES)}


# ---------- entrenamiento ----------
def _trabajo(cfg: TrainConfig, job_id: str) -> None:
    def cb(ev: dict) -> None:
        with ESTADO.lock:
            j = ESTADO.job
            if j.get("job_id") != job_id:
                return
            j["etapa"] = ev["etapa"]
            for k in ("epoca", "epocas_total", "mejor_epoca"):
                if k in ev:
                    j[k] = ev[k]
            if "historial" in ev:
                j["historial"] = ev["historial"]
    try:
        m = train_mod.entrenar(cfg, cb=cb, datos=ESTADO.datos, rutas=ESTADO.rutas)
        ESTADO.cargar_predictor()   # recarga sin reiniciar el servidor
        with ESTADO.lock:
            ESTADO.job.update({"estado": "completado", "etapa": "completado",
                               "resumen": {"mejor_epoca": m["mejor_epoca"], "tiempo_total_s": m["tiempo_total_s"],
                                           "prueba": m["prueba"], "validacion": m["validacion"]}})
    except Exception as e:  # noqa: BLE001
        with ESTADO.lock:
            ESTADO.job.update({"estado": "error", "mensaje_error": f"{type(e).__name__}: {e}"})


@app.post("/train", status_code=202, tags=["entrenamiento"])
def entrenar(body: TrainIn) -> dict:
    try:
        cfg = TrainConfig(**body.model_dump())
    except ValueError as e:
        raise HTTPException(400, f"Parámetros inválidos: {e}")
    with ESTADO.lock:
        if ESTADO.job.get("estado") == "en_curso":
            raise HTTPException(409, "Ya hay un entrenamiento en curso. Espera a que termine.")
        job_id = uuid.uuid4().hex[:8]
        ESTADO.job = {"job_id": job_id, "estado": "en_curso", "etapa": "iniciando", "epoca": 0,
                      "epocas_total": cfg.epocas, "historial": [], "config": cfg.a_dict()}
    threading.Thread(target=_trabajo, args=(cfg, job_id), daemon=True).start()
    return {"job_id": job_id, "estado": "en_curso"}


@app.get("/train/status", tags=["entrenamiento"])
def train_status() -> dict:
    with ESTADO.lock:
        return json.loads(json.dumps(ESTADO.job, default=float))


# ---------- métricas ----------
@app.get("/metrics", tags=["métricas"])
def metrics() -> dict:
    m = _leer_json("metricas.json")
    if m is None:
        raise HTTPException(404, "Aún no hay métricas: entrena un modelo primero.")
    return m


@app.get("/metrics/detalle", tags=["métricas"])
def metrics_detalle() -> dict:
    """Material para las pestañas de Métricas: épocas, predicciones de prueba, palabras, errores…"""
    p = _predictor()
    rep = ESTADO.rutas.reportes
    out: dict = {"historial": None, "pred_test": None, "errores": None, "palabras": None}
    if (rep / "historial_epocas.csv").exists():
        out["historial"] = pd.read_csv(rep / "historial_epocas.csv").to_dict("records")
    if (rep / "pred_test.csv").exists():
        d = pd.read_csv(rep / "pred_test.csv")
        out["pred_test"] = {"y": d["y"].tolist(), "p_positivo": d["p_positivo"].round(5).tolist()}
    if (rep / "errores.csv").exists():
        out["errores"] = pd.read_csv(rep / "errores.csv").head(50).fillna("").to_dict("records")
    if p.pesos is not None:
        idx = np.argsort(p.pesos)
        mk = lambda ii: [{"termino": str(p.terminos[i]), "peso": float(p.pesos[i])} for i in ii]
        out["palabras"] = {"negativas": mk(idx[:20]), "positivas": mk(idx[::-1][:20])}
    out["seleccion_modelo"] = _leer_json("seleccion_modelo.json")
    out["analisis_errores"] = _leer_json("analisis_errores.json")
    return out


@app.get("/model/info", tags=["métricas"])
def model_info() -> dict:
    return _predictor().info()


# ---------- dataset ----------
def _dataset() -> pd.DataFrame:
    if ESTADO.datos is not None:
        return pd.concat(ESTADO.datos, ignore_index=True)
    fs = [DATA_PROC / f"{n}.csv" for n in ("train", "val", "test")]
    if not all(f.exists() for f in fs):
        raise HTTPException(404, "No hay datos procesados. Ejecuta: python -m src.preprocessing")
    return pd.concat([pd.read_csv(f).assign(particion=f.stem) for f in fs], ignore_index=True)


def _filtrar(df: pd.DataFrame, estrellas: int | None, categoria: str | None, sentimiento: int | None) -> pd.DataFrame:
    if estrellas is not None:
        df = df[df["estrellas"] == estrellas]
    if categoria and "categoria" in df:
        df = df[df["categoria"] == categoria]
    if sentimiento is not None:
        df = df[df["sentimiento"] == sentimiento]
    return df


@app.get("/dataset/summary", tags=["dataset"])
def dataset_summary(estrellas: int | None = None, categoria: str | None = None,
                    sentimiento: int | None = None) -> dict:
    df_all = _dataset()
    df = _filtrar(df_all, estrellas, categoria, sentimiento)
    out: dict = {"total": len(df),
                 "positivas": int((df["sentimiento"] == 1).sum()), "negativas": int((df["sentimiento"] == 0).sum()),
                 "estrellas": {str(k): int(v) for k, v in df["estrellas"].value_counts().sort_index().items()},
                 "palabras_media": float(df["num_palabras"].mean()) if "num_palabras" in df and len(df) else None,
                 "categorias_disponibles": sorted(df_all["categoria"].dropna().unique().tolist()) if "categoria" in df_all else []}
    if "categoria" in df:
        t = pd.crosstab(df["categoria"], df["sentimiento"]).rename(columns={0: "negativo", 1: "positivo"})
        for c in ("negativo", "positivo"):
            if c not in t:
                t[c] = 0
        out["categorias"] = t.assign(total=t.sum(axis=1)).sort_values("total", ascending=False).head(15).reset_index().to_dict("records")
    if "num_palabras" in df and len(df):
        lim = float(df["num_palabras"].quantile(0.99))
        bins = np.linspace(0, lim, 31)
        out["histograma"] = {"bordes": bins.tolist(),
                             "negativo": np.histogram(df.loc[df.sentimiento == 0, "num_palabras"].clip(upper=lim), bins)[0].tolist(),
                             "positivo": np.histogram(df.loc[df.sentimiento == 1, "num_palabras"].clip(upper=lim), bins)[0].tolist()}
    out["auditoria"] = _leer_json("auditoria_datos.json")
    return out


@app.get("/dataset/sample", tags=["dataset"])
def dataset_sample(estrellas: int | None = None, n: int = 50, categoria: str | None = None,
                   sentimiento: int | None = None, pagina: int = 0) -> dict:
    n = max(1, min(n, 500))
    df = _filtrar(_dataset(), estrellas, categoria, sentimiento)
    cols = [c for c in ("texto", "estrellas", "sentimiento", "categoria") if c in df]
    parte = df.sort_index().iloc[pagina * n:(pagina + 1) * n][cols]
    return {"total_filtrado": len(df), "pagina": pagina, "filas": parte.to_dict("records")}


# ---------- predicción ----------
@app.post("/predict", tags=["predicción"])
def predict(body: PredictIn) -> dict:
    try:
        return _predictor().predecir(body.texto, body.umbral)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/predict/batch", tags=["predicción"])
def predict_batch(body: BatchIn) -> dict:
    return {"resultados": _predictor().predecir_lote(body.textos, body.umbral)}


@app.get("/metrics/correlacion", tags=["métricas"])
def metrics_correlacion() -> dict:
    """Correlación de Pearson de las variables derivadas, el sentimiento y P(Positivo) del modelo."""
    from .eda import COLS_CORR
    from .train import probabilidad_positivo
    p = _predictor()
    df = _dataset()
    faltan = [c for c in COLS_CORR if c not in df]
    if faltan:
        raise HTTPException(404, f"Faltan variables derivadas en los datos: {faltan}")
    x = df[COLS_CORR].copy()
    x["p_positivo_modelo"] = probabilidad_positivo(p.modelo, p.vec.transform(df["texto_limpio"].fillna("")))
    c = x.corr(method="pearson")
    return {"variables": c.columns.tolist(), "matriz": c.round(4).values.tolist()}
