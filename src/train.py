"""Entrenamiento por épocas con checkpointing, early stopping, evaluación única en prueba y guardado."""
from __future__ import annotations

import copy
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from . import eda, plots
from .config import DATA_PROC, FIGURES, MODELS, REPORTS, SEED, TrainConfig, asegurar_carpetas

Callback = Callable[[dict], None]


@dataclass
class Rutas:
    modelos: Path = MODELS
    reportes: Path = REPORTS
    figuras: Path = FIGURES


def _sigmoide(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def probabilidad_positivo(modelo, X) -> np.ndarray:
    """P(Positivo). Para SVM (sin predict_proba) se usa la sigmoide de la función de decisión."""
    if hasattr(modelo, "predict_proba"):
        return modelo.predict_proba(X)[:, 1]
    return _sigmoide(modelo.decision_function(X))


def pesos_modelo(modelo) -> np.ndarray | None:
    """Peso por término (positivo = empuja a Positivo)."""
    if hasattr(modelo, "coef_"):
        return np.asarray(modelo.coef_).ravel()
    if hasattr(modelo, "feature_log_prob_"):
        return modelo.feature_log_prob_[1] - modelo.feature_log_prob_[0]
    return None


def _metricas(y, p: np.ndarray, umbral: float = 0.5) -> dict:
    pred = (p >= umbral).astype(int)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "f1_macro": float(f1_score(y, pred, average="macro", zero_division=0)),
        "logloss": float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1])),
        "roc_auc": float(roc_auc_score(y, p)) if len(set(y)) > 1 else float("nan"),
        "pr_auc": float(average_precision_score(y, p)) if len(set(y)) > 1 else float("nan"),
    }


def crear_modelo(cfg: TrainConfig):
    a = cfg.algoritmo
    if a == "sgd":
        pen = cfg.penalizacion
        return SGDClassifier(loss="log_loss", penalty=pen, alpha=cfg.alpha, l1_ratio=0.15,
                             random_state=cfg.random_state)
    if a == "logreg":
        return LogisticRegression(C=1.0 / max(cfg.alpha * 1e4, 1e-3), max_iter=1000, random_state=cfg.random_state)
    if a == "svm":
        return LinearSVC(C=1.0 / max(cfg.alpha * 1e4, 1e-3), random_state=cfg.random_state)
    return MultinomialNB(alpha=min(max(cfg.alpha * 1e4, 0.01), 10.0))


def crear_vectorizador(cfg: TrainConfig) -> TfidfVectorizer:
    return TfidfVectorizer(ngram_range=(1, cfg.ngram_max), max_features=cfg.max_features, min_df=cfg.min_df,
                           max_df=0.95, sublinear_tf=True)


def _subconjunto_balanceado(tr: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    m = min(n, int(tr["sentimiento"].value_counts().min()))
    partes = [g.sample(n=m, random_state=seed) for _, g in tr.groupby("sentimiento")]
    return pd.concat(partes).sample(frac=1, random_state=seed).reset_index(drop=True)


def entrenar_epocas(cfg: TrainConfig, Xtr, ytr, Xva, yva, cb: Callback | None = None):
    """SGD con partial_fit. Devuelve (mejor_modelo, historial, mejor_epoca, parada_temprana)."""
    modelo = crear_modelo(cfg)
    rng = np.random.RandomState(cfg.random_state)
    hist: list[dict] = []
    mejor, mejor_modelo, mejor_ep, sin_mejora, parada = (-1.0, np.inf), None, 0, 0, False
    for ep in range(1, cfg.epocas + 1):
        orden = rng.permutation(Xtr.shape[0])
        modelo.partial_fit(Xtr[orden], ytr[orden], classes=np.array([0, 1]))
        ptr, pva = probabilidad_positivo(modelo, Xtr), probabilidad_positivo(modelo, Xva)
        mt, mv = _metricas(ytr, ptr), _metricas(yva, pva)
        fila = {"epoca": ep, **{f"{k}_train": mt[k] for k in ("logloss", "accuracy", "f1")},
                **{f"{k}_val": mv[k] for k in ("logloss", "accuracy", "f1")}}
        hist.append(fila)
        clave = (mv["f1"], -mv["logloss"])
        if clave > (mejor[0], -mejor[1]):
            mejor, mejor_modelo, mejor_ep, sin_mejora = (mv["f1"], mv["logloss"]), copy.deepcopy(modelo), ep, 0
        else:
            sin_mejora += 1
        if cb:
            cb({"etapa": "entrenando", "epoca": ep, "epocas_total": cfg.epocas, "historial": hist,
                "mejor_epoca": mejor_ep})
        if sin_mejora >= cfg.paciencia:
            parada = True
            break
    return mejor_modelo, pd.DataFrame(hist), mejor_ep, parada


def comparar_modelos(cfg: TrainConfig, Xtr, ytr, Xva, yva, Xte, yte) -> pd.DataFrame:
    """Entrena los 5 candidatos con el mismo TF-IDF. Cada uno se evalúa una vez en prueba (sin elegir con ella)."""
    sel = REPORTS / "seleccion_modelo.json"
    cv = json.loads(sel.read_text(encoding="utf-8"))["modelos_cv"] if sel.exists() else {}
    cand = {
        "Dummy": DummyClassifier(strategy="most_frequent"),
        "Naive Bayes": MultinomialNB(alpha=0.5),
        "Regresión Logística": LogisticRegression(C=4, max_iter=1000, random_state=SEED),
        "SVM Lineal": LinearSVC(C=0.5, random_state=SEED),
        "SGD por épocas": None,
    }
    filas = []
    for nombre, m in cand.items():
        t0 = time.time()
        if m is None:
            m = crear_modelo(TrainConfig(**{**cfg.a_dict(), "algoritmo": "sgd"}))
            m, _, _, _ = entrenar_epocas(TrainConfig(**{**cfg.a_dict(), "algoritmo": "sgd"}), Xtr, ytr, Xva, yva)
        else:
            m.fit(Xtr, ytr)
        t = time.time() - t0
        pv = probabilidad_positivo(m, Xva) if hasattr(m, "predict_proba") or hasattr(m, "decision_function") else None
        pt = probabilidad_positivo(m, Xte)
        key = "SGD" if nombre == "SGD por épocas" else nombre
        filas.append({"modelo": nombre,
                      "f1_cv_media": cv.get(key, {}).get("f1_cv_media"), "f1_cv_de": cv.get(key, {}).get("f1_cv_de"),
                      "f1_val": _metricas(yva, pv)["f1"], "f1_prueba": _metricas(yte, pt)["f1"],
                      "roc_auc_prueba": _metricas(yte, pt)["roc_auc"], "tiempo_s": round(t, 3)})
    return pd.DataFrame(filas)


def entrenar(cfg: TrainConfig | None = None, cb: Callback | None = None, datos: tuple | None = None,
             rutas: Rutas | None = None, guardar_figuras: bool = True) -> dict:
    """Pipeline completo. `datos=(train, val, test)` permite inyectar DataFrames (pruebas)."""
    cfg = cfg or TrainConfig()
    rutas = rutas or Rutas()
    asegurar_carpetas()
    for c in (rutas.modelos, rutas.reportes, rutas.figuras):
        c.mkdir(parents=True, exist_ok=True)
    t_ini = time.time()

    def avisar(etapa: str, **kw) -> None:
        if cb:
            cb({"etapa": etapa, **kw})

    avisar("cargando datos")
    tr, va, te = datos if datos is not None else (pd.read_csv(DATA_PROC / f"{n}.csv") for n in ("train", "val", "test"))
    tr = _subconjunto_balanceado(tr, cfg.n_por_clase, cfg.random_state)

    avisar("vectorizando (TF-IDF solo con entrenamiento)")
    vec = crear_vectorizador(cfg)
    Xtr = vec.fit_transform(tr["texto_limpio"].fillna(""))          # ajuste SOLO con train
    Xva, Xte = vec.transform(va["texto_limpio"].fillna("")), vec.transform(te["texto_limpio"].fillna(""))
    ytr, yva, yte = (d["sentimiento"].values for d in (tr, va, te))

    avisar("entrenando", epoca=0, epocas_total=cfg.epocas, historial=[])
    if cfg.algoritmo == "sgd":
        modelo, hist, mejor_ep, parada = entrenar_epocas(cfg, Xtr, ytr, Xva, yva, cb)
    else:
        modelo = crear_modelo(cfg).fit(Xtr, ytr)
        ptr, pva = probabilidad_positivo(modelo, Xtr), probabilidad_positivo(modelo, Xva)
        mt, mv = _metricas(ytr, ptr), _metricas(yva, pva)
        hist = pd.DataFrame([{"epoca": 1, **{f"{k}_train": mt[k] for k in ("logloss", "accuracy", "f1")},
                              **{f"{k}_val": mv[k] for k in ("logloss", "accuracy", "f1")}}])
        mejor_ep, parada = 1, False
        avisar("entrenando", epoca=1, epocas_total=1, historial=hist.to_dict("records"), mejor_epoca=1)

    avisar("evaluando en prueba (una sola vez)")
    pva = probabilidad_positivo(modelo, Xva)
    pte = probabilidad_positivo(modelo, Xte)
    m_val, m_te = _metricas(yva, pva, cfg.umbral), _metricas(yte, pte, cfg.umbral)
    pred = (pte >= cfg.umbral).astype(int)
    cm = confusion_matrix(yte, pred, labels=[0, 1])
    reporte = classification_report(yte, pred, target_names=["Negativo", "Positivo"], output_dict=True, zero_division=0)

    tabla = None
    if cfg.comparar_modelos:
        avisar("comparando modelos")
        tabla = comparar_modelos(cfg, Xtr, ytr, Xva, yva, Xte, yte)

    avisar("guardando artefactos y figuras")
    fecha = datetime.now().isoformat(timespec="seconds")
    resumen = {k: round(m_te[k], 4) for k in ("accuracy", "f1", "roc_auc")}
    joblib.dump({"modelo": modelo, "umbral": cfg.umbral, "config": cfg.a_dict(), "fecha": fecha,
                 "version_sklearn": sklearn.__version__, "metricas_resumen": resumen,
                 "algoritmo": cfg.algoritmo}, rutas.modelos / "modelo.joblib")
    joblib.dump(vec, rutas.modelos / "vectorizador.joblib")

    metricas = {
        "fecha": fecha, "config": cfg.a_dict(), "mejor_epoca": int(mejor_ep), "epocas_ejecutadas": len(hist),
        "parada_temprana": bool(parada), "tiempo_total_s": round(time.time() - t_ini, 2),
        "n_train": len(tr), "n_val": len(va), "n_test": len(te),
        "vocabulario": len(vec.vocabulary_),
        "validacion": m_val, "prueba": m_te,
        "matriz_confusion": {"tn": int(cm[0, 0]), "fp": int(cm[0, 1]), "fn": int(cm[1, 0]), "tp": int(cm[1, 1])},
        "classification_report": reporte,
        "comparacion_modelos": None if tabla is None else tabla.to_dict("records"),
    }
    # Errores más seguros
    err = te.assign(p_positivo=pte, pred=pred)
    err = err[err["pred"] != err["sentimiento"]].copy()
    err["confianza_error"] = np.where(err["pred"] == 1, err["p_positivo"], 1 - err["p_positivo"])
    err["tipo"] = np.where(err["pred"] == 1, "falso_positivo", "falso_negativo")
    cols = [c for c in ("texto", "estrellas", "sentimiento", "pred", "p_positivo", "confianza_error", "tipo", "categoria") if c in err]
    err.sort_values("confianza_error", ascending=False)[cols].to_csv(rutas.reportes / "errores.csv", index=False)
    pd.DataFrame({"y": yte, "p_positivo": pte}).to_csv(rutas.reportes / "pred_test.csv", index=False)
    hist.to_csv(rutas.reportes / "historial_epocas.csv", index=False)
    (rutas.reportes / "metricas.json").write_text(json.dumps(metricas, indent=2, ensure_ascii=False), encoding="utf-8")

    figuras = []
    if guardar_figuras:
        f = rutas.figuras
        figuras += [plots.fig_epocas(hist, mejor_ep, f), plots.fig_confusion(yte, pred, f),
                    plots.fig_roc(yte, pte, m_te["roc_auc"], f), plots.fig_pr(yte, pte, f),
                    plots.fig_calibracion(yte, pte, f)]
        w = pesos_modelo(modelo)
        if w is not None:
            figuras.append(plots.fig_palabras(np.array(vec.get_feature_names_out()), w, 20, f))
        if tabla is not None:
            figuras.append(plots.fig_comparacion(tabla, f))
        if {"num_negaciones", "longitud_caracteres"} <= set(tr.columns):
            todo = pd.concat([tr, va, te], ignore_index=True)
            p_todo = probabilidad_positivo(modelo, vec.transform(todo["texto_limpio"].fillna("")))
            eda._estilo()
            figuras.append(eda.fig_correlacion(todo, pd.Series(p_todo), "5_matriz_correlacion.png", f))
    metricas["figuras"] = figuras
    avisar("completado", metricas=metricas)
    return metricas


def main() -> None:
    cfg = TrainConfig(comparar_modelos=True)
    m = entrenar(cfg, cb=lambda e: print(f"  época {e['epoca']:>2}/{e['epocas_total']}" if e["etapa"] == "entrenando" and e.get("epoca") and e["epoca"] % 5 == 0 else "", end=""))
    h = pd.read_csv(REPORTS / "historial_epocas.csv")
    print("\n\n== Historial por época (cada 5) ==")
    print(h.iloc[::5].round(4).to_string(index=False))
    print(f"\nMejor época: {m['mejor_epoca']}  (ejecutadas: {m['epocas_ejecutadas']}, parada temprana: {m['parada_temprana']})")
    print(f"Vocabulario: {m['vocabulario']}  | n_train={m['n_train']} n_val={m['n_val']} n_test={m['n_test']}")
    print("\n== Validación ==", {k: round(v, 4) for k, v in m["validacion"].items()})
    print("== PRUEBA     ==", {k: round(v, 4) for k, v in m["prueba"].items()})
    mc = m["matriz_confusion"]
    print(f"\nMatriz de confusión (prueba):\n            pred Neg  pred Pos\n  real Neg  {mc['tn']:>8}  {mc['fp']:>8}\n  real Pos  {mc['fn']:>8}  {mc['tp']:>8}")
    print("\n== Comparación de modelos ==")
    print(pd.DataFrame(m["comparacion_modelos"]).round(4).to_string(index=False))
    print(f"\nTiempo total: {m['tiempo_total_s']} s\nFiguras: {m['figuras']}")


if __name__ == "__main__":
    main()
