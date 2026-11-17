"""Selección de modelo: palabras vacías, CV 5 pliegues, rejillas, curva de aprendizaje (sección 8)."""
from __future__ import annotations

import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binom
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .config import COLOR_POS, DATA_PROC, FIGURES, REPORTS, SEED, asegurar_carpetas
from .preprocessing import balancear, limpiar_texto


def tfidf(ngram_max: int = 2, max_features: int = 20000, min_df: int = 2) -> TfidfVectorizer:
    return TfidfVectorizer(ngram_range=(1, ngram_max), max_features=max_features, min_df=min_df,
                           max_df=0.95, sublinear_tf=True)


def cargar() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return tuple(pd.read_csv(DATA_PROC / f"{n}.csv") for n in ("train", "val", "test"))  # type: ignore


def modelos_base() -> dict:
    return {
        "Dummy": DummyClassifier(strategy="most_frequent"),
        "Naive Bayes": MultinomialNB(),
        "Regresión Logística": LogisticRegression(max_iter=1000, random_state=SEED),
        "SVM Lineal": LinearSVC(random_state=SEED),
        "SGD": SGDClassifier(loss="log_loss", random_state=SEED, max_iter=1000, tol=1e-4),
    }


def experimento_stopwords(tr: pd.DataFrame, va: pd.DataFrame) -> dict:
    """(a) sin quitar, (b) lista estándar, (c) lista propia. F1 en validación (LogReg, TF-IDF base)."""
    res = {}
    for nombre, modo in (("a_sin_quitar", "ninguna"), ("b_estandar", "estandar"), ("c_propia_conserva_negaciones", "propia")):
        xt = tr["texto"].map(lambda t: limpiar_texto(t, modo))
        xv = va["texto"].map(lambda t: limpiar_texto(t, modo))
        pipe = Pipeline([("tfidf", tfidf()), ("clf", LogisticRegression(max_iter=1000, random_state=SEED))])
        pipe.fit(xt, tr["sentimiento"])
        res[nombre] = round(float(f1_score(va["sentimiento"], pipe.predict(xv))), 4)
    return res


def _cv_resumen(gs: GridSearchCV) -> dict:
    i = gs.best_index_
    r = gs.cv_results_
    return {"mejores_params": {k: (v if not isinstance(v, tuple) else list(v)) for k, v in gs.best_params_.items()},
            "f1_cv_media": round(float(r["mean_test_score"][i]), 4),
            "f1_cv_de": round(float(r["std_test_score"][i]), 4),
            "tiempo_fit_s": round(float(r["mean_fit_time"][i]), 3)}


def busqueda_tfidf(tr: pd.DataFrame, cv) -> dict:
    """Etapa 1: elegir TF-IDF con LogReg (el TF-IDF se reajusta en cada pliegue: sin fuga)."""
    pipe = Pipeline([("tfidf", tfidf()), ("clf", LogisticRegression(max_iter=1000, random_state=SEED))])
    grid = {"tfidf__ngram_range": [(1, 1), (1, 2), (1, 3)],
            "tfidf__max_features": [10000, 20000, 50000], "tfidf__min_df": [2, 5]}
    gs = GridSearchCV(pipe, grid, scoring="f1", cv=cv, n_jobs=-1).fit(tr["texto_limpio"], tr["sentimiento"])
    return {"gs": gs, **_cv_resumen(gs)}


def busqueda_modelos(tr: pd.DataFrame, cv, tp: dict) -> dict:
    """Etapa 2: hiperparámetros de cada modelo con el TF-IDF elegido."""
    ng = tp["tfidf__ngram_range"][1]
    grids = {
        "Dummy": {},
        "Naive Bayes": {"clf__alpha": [0.1, 0.5, 1.0]},
        "Regresión Logística": {"clf__C": [0.5, 1, 4, 10]},
        "SVM Lineal": {"clf__C": [0.1, 0.5, 1]},
        "SGD": {"clf__alpha": [1e-6, 1e-5, 1e-4], "clf__penalty": ["l2", "elasticnet"]},
    }
    out = {}
    for nombre, modelo in modelos_base().items():
        pipe = Pipeline([("tfidf", tfidf(ng, tp["tfidf__max_features"], tp["tfidf__min_df"])), ("clf", modelo)])
        t0 = time.time()
        gs = GridSearchCV(pipe, grids[nombre] or {"clf__strategy": ["most_frequent"]}, scoring="f1", cv=cv,
                          n_jobs=-1).fit(tr["texto_limpio"], tr["sentimiento"])
        r = _cv_resumen(gs)
        r["tiempo_busqueda_s"] = round(time.time() - t0, 1)
        r["f1_macro_cv"] = round(float(np.mean(
            [GridSearchCV(gs.best_estimator_, {}, scoring="f1_macro", cv=cv).fit(
                tr["texto_limpio"], tr["sentimiento"]).best_score_])), 4)
        out[nombre] = {"gs": gs, **r}
    return out


def curva_aprendizaje(tr: pd.DataFrame, va: pd.DataFrame, tp: dict, modelo_params: dict) -> list[dict]:
    ng = tp["tfidf__ngram_range"][1]
    maxn = int(tr["sentimiento"].value_counts().min())
    tamanos = sorted({n for n in (250, 500, 1000, 1500, 2000, maxn) if n <= maxn})
    filas = []
    for n in tamanos:
        sub = balancear(tr, n_por_clase=n)
        t0 = time.time()
        pipe = Pipeline([("tfidf", tfidf(ng, tp["tfidf__max_features"], tp["tfidf__min_df"])),
                         ("clf", LogisticRegression(max_iter=1000, random_state=SEED, **modelo_params))])
        pipe.fit(sub["texto_limpio"], sub["sentimiento"])
        t = time.time() - t0
        f_tr = f1_score(sub["sentimiento"], pipe.predict(sub["texto_limpio"]))
        f_va = f1_score(va["sentimiento"], pipe.predict(va["texto_limpio"]))
        filas.append({"n_por_clase": n, "f1_entrenamiento": round(float(f_tr), 4),
                      "f1_validacion": round(float(f_va), 4), "tiempo_s": round(t, 2)})
    return filas


def fig_curva(filas: list[dict]) -> None:
    plt.rcParams.update({"figure.facecolor": "#0E1117", "axes.facecolor": "#0E1117", "savefig.facecolor": "#0E1117",
                         "text.color": "#F0EFEC", "axes.labelcolor": "#F0EFEC", "xtick.color": "#F0EFEC",
                         "ytick.color": "#F0EFEC", "axes.edgecolor": "#555"})
    d = pd.DataFrame(filas)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(d.n_por_clase, d.f1_validacion, "o-", color=COLOR_POS, label="Validación")
    ax.plot(d.n_por_clase, d.f1_entrenamiento, "o--", color="#E66767", label="Entrenamiento")
    ax.set_xlabel("Reseñas por clase")
    ax.set_ylabel("F1")
    ax.set_title("Curva de aprendizaje (cantidad de datos)")
    ax.legend()
    fig.savefig(FIGURES / "8_curva_aprendizaje.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def mcnemar(y, p1, p2) -> dict:
    b = int(((p1 == y) & (p2 != y)).sum())
    c = int(((p1 != y) & (p2 == y)).sum())
    n = b + c
    p = 1.0 if n == 0 else float(min(1.0, 2 * binom.cdf(min(b, c), n, 0.5)))
    return {"solo_modelo1_acierta": b, "solo_modelo2_acierta": c, "p_valor_exacto": round(p, 4)}


def main() -> dict:
    asegurar_carpetas()
    tr, va, _ = cargar()   # la prueba NO se toca aquí
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

    print("== Experimento de palabras vacías (F1 validación, LogReg) ==")
    sw = experimento_stopwords(tr, va)
    for k, v in sw.items():
        print(f"  {k:<32}{v:.4f}")

    print("\n== Etapa 1: TF-IDF (CV 5 pliegues, LogReg) ==")
    e1 = busqueda_tfidf(tr, cv)
    tp = e1["gs"].best_params_
    print("  mejores:", e1["mejores_params"], f"F1={e1['f1_cv_media']}±{e1['f1_cv_de']}")

    print("\n== Etapa 2: modelos (CV 5 pliegues, F1) ==")
    mods = busqueda_modelos(tr, cv, tp)
    print(f"  {'Modelo':<22}{'F1 CV':>16}{'F1 macro':>10}{'t fit (s)':>10}  params")
    for n, r in mods.items():
        print(f"  {n:<22}{r['f1_cv_media']:>9.4f} ± {r['f1_cv_de']:.4f}{r['f1_macro_cv']:>10.4f}{r['tiempo_fit_s']:>10.3f}  {r['mejores_params']}")

    # McNemar entre los 2 mejores (en validación)
    orden = sorted((n for n in mods if n != "Dummy"), key=lambda n: -mods[n]["f1_cv_media"])
    m1, m2 = orden[:2]
    p1 = mods[m1]["gs"].best_estimator_.predict(va["texto_limpio"])
    p2 = mods[m2]["gs"].best_estimator_.predict(va["texto_limpio"])
    mc = {"modelos": [m1, m2], **mcnemar(va["sentimiento"].values, p1, p2)}
    print(f"\n== McNemar ({m1} vs {m2}, validación) ==", mc)

    print("\n== Curva de aprendizaje (LogReg, F1 validación) ==")
    cp = {k.replace("clf__", ""): v for k, v in mods["Regresión Logística"]["mejores_params"].items()}
    curva = curva_aprendizaje(tr, va, tp, cp)
    for f in curva:
        print(" ", f)
    fig_curva(curva)
    f = [c["f1_validacion"] for c in curva]
    ganancia = round((f[-1] - f[-2]) * 100, 2) if len(f) > 1 else None
    concl = (f"Pasar de {curva[-2]['n_por_clase']} a {curva[-1]['n_por_clase']} reseñas por clase cambia el F1 de "
             f"validación en {ganancia:+.2f} puntos; el máximo disponible es {curva[-1]['n_por_clase']} por clase.")
    print("  Conclusión:", concl)

    ser = {n: {k: v for k, v in r.items() if k != "gs"} for n, r in mods.items()}
    salida = {"stopwords_f1_val": sw,
              "tfidf_elegido": {k: (list(v) if isinstance(v, tuple) else v) for k, v in tp.items()},
              "tfidf_cv": {k: v for k, v in e1.items() if k != "gs"},
              "modelos_cv": ser, "mcnemar": mc, "curva_aprendizaje": curva, "conclusion_curva": concl}
    (REPORTS / "seleccion_modelo.json").write_text(json.dumps(salida, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nGuardado reports/seleccion_modelo.json y figura 8_curva_aprendizaje.png")
    return salida


if __name__ == "__main__":
    main()
