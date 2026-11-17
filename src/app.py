"""Frontend Streamlit. Consume la API por HTTP (no importa el modelo directamente)."""
from __future__ import annotations

import io
import os
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

API = os.environ.get("API_URL", "http://127.0.0.1:8000")
POS, NEG = "#3987E5", "#E66767"
BG, PANEL, WINE, TXT, TXT2 = "#0E1117", "#14213D", "#7B2D3B", "#F0EFEC", "#C3C2B7"

st.set_page_config(page_title="Sentimientos en reseñas", page_icon="💬", layout="wide")
st.markdown(f"""<style>
h1,h2,h3 {{ color:{TXT}; }}
div.stButton > button[kind="primary"] {{ background:{WINE}; border:0; color:{TXT}; }}
[data-testid="stMetric"] {{ background:{PANEL}; padding:12px 16px; border-radius:10px; }}
</style>""", unsafe_allow_html=True)


# ---------------- utilidades ----------------
def ancho(fn, *a, **kw):
    """Compatibilidad: width='stretch' (nuevo) con respaldo a use_container_width (antiguo)."""
    try:
        return fn(*a, width="stretch", **kw)
    except TypeError:
        return fn(*a, use_container_width=True, **kw)


class ApiError(Exception):
    pass


def llamar(metodo: str, ruta: str, **kw):
    try:
        r = requests.request(metodo, f"{API}{ruta}", timeout=kw.pop("timeout", 30), **kw)
    except requests.exceptions.RequestException:
        raise ApiError("La API no está encendida: ejecuta `python -m uvicorn src.api:app --port 8000`")
    if r.status_code >= 400:
        try:
            det = r.json().get("detail", r.text)
        except Exception:  # noqa: BLE001
            det = r.text
        raise ApiError(f"{r.status_code}: {det}" if not isinstance(det, list) else f"{r.status_code}: {det[0].get('msg')}")
    return r.json()


@st.cache_data(ttl=5, show_spinner=False)
def get_cache(ruta: str, params: tuple = ()):
    return llamar("GET", ruta, params=dict(params))


def api_ok() -> bool:
    try:
        requests.get(f"{API}/health", timeout=2)
        return True
    except requests.exceptions.RequestException:
        return False


def tema(fig: go.Figure, alto: int = 380) -> go.Figure:
    fig.update_layout(template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG, height=alto,
                      margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.2))
    return fig


def etiqueta(e: str) -> str:
    return "✅ Positivo" if e == "Positivo" else "⚠️ Negativo"


def sin_modelo() -> None:
    st.warning("Todavía no hay un modelo entrenado. Ve a la página **Entrenamiento**.")


# ---------------- barra lateral ----------------
st.sidebar.title("💬 Sentimientos ES")
st.sidebar.caption("Reseñas de Amazon en español")
pagina = st.sidebar.radio("Navegación", ["Inicio", "Entrenamiento", "Métricas", "Simulador", "Dashboard"])
conectada = api_ok()
st.sidebar.markdown("🟢 API conectada" if conectada else "🔴 API sin conexión")
if not conectada:
    st.error("La API no está encendida: ejecuta `python -m uvicorn src.api:app --port 8000`")
    st.stop()


def metricas_o_none():
    try:
        return llamar("GET", "/metrics")
    except ApiError:
        return None


# ---------------- 1. Inicio ----------------
def pag_inicio() -> None:
    st.title("Análisis de sentimientos en reseñas de comercio electrónico")
    st.write("Clasifica reseñas en español como **Positivo** (4–5★) o **Negativo** (1–2★), indica la "
             "certeza y **explica** qué palabras pesaron en la decisión.")
    st.markdown("**Cómo usarla:** 1) *Entrenamiento*: configura y entrena · 2) *Métricas*: evalúa · "
                "3) *Simulador*: prueba reseñas nuevas · 4) *Dashboard*: explora el dataset.")
    m = metricas_o_none()
    if m is None:
        sin_modelo()
        return
    p = m["prueba"]
    c = st.columns(4)
    c[0].metric("Accuracy (prueba)", f"{p['accuracy']:.3f}")
    c[1].metric("F1 (prueba)", f"{p['f1']:.3f}")
    c[2].metric("ROC-AUC (prueba)", f"{p['roc_auc']:.3f}")
    c[3].metric("Entrenado", m["fecha"][:10])
    st.caption(f"Algoritmo: {m['config']['algoritmo']} · mejor época: {m['mejor_epoca']} · "
               f"vocabulario: {m['vocabulario']:,} términos")


# ---------------- 2. Entrenamiento ----------------
def pag_entrenamiento() -> None:
    st.title("Entrenamiento")
    d = get_cache("/config/default")
    df = d["defaults"]
    if "reset" not in st.session_state:
        st.session_state.reset = 0
    k = st.session_state.reset

    with st.expander("📦 Datos", expanded=True):
        n_clase = st.slider("Reseñas por clase", 100, 4000, min(df["n_por_clase"], 4000), 100, key=f"n{k}",
                            help="El dataset procesado ofrece ~2 754 por clase en entrenamiento.")
    with st.expander("🔤 TF-IDF", expanded=True):
        c = st.columns(3)
        ng = c[0].selectbox("N-grama máximo", [1, 2, 3], index=df["ngram_max"] - 1, key=f"ng{k}")
        mf = c[1].slider("max_features", 1000, 50000, df["max_features"], 1000, key=f"mf{k}")
        md = c[2].slider("min_df", 1, 20, df["min_df"], key=f"md{k}")
    with st.expander("🧠 Modelo", expanded=True):
        c = st.columns(3)
        alg = c[0].selectbox("Algoritmo", d["algoritmos"], key=f"al{k}",
                             format_func=lambda a: {"sgd": "SGD (por épocas)", "logreg": "Regresión Logística",
                                                    "svm": "SVM lineal", "nb": "Naive Bayes"}[a])
        exp = c[1].slider("α (log10)", -8.0, 1.0, float(np.log10(df["alpha"])), 0.5, key=f"a{k}")
        pen = c[2].selectbox("Penalización (SGD)", d["penalizaciones"][:1] + ["elasticnet"], key=f"pe{k}")
        c = st.columns(3)
        ep = c[0].slider("Épocas (SGD)", 1, 100, df["epocas"], key=f"ep{k}")
        pa = c[1].slider("Paciencia (early stopping)", 1, 50, df["paciencia"], key=f"pa{k}")
        um = c[2].slider("Umbral", 0.05, 0.95, df["umbral"], 0.05, key=f"um{k}")
        comp = st.checkbox("Comparar con otros modelos (Dummy, NB, LogReg, SVM)", key=f"co{k}")

    cfg = {"n_por_clase": n_clase, "ngram_max": ng, "max_features": mf, "min_df": md, "algoritmo": alg,
           "epocas": ep, "alpha": float(10 ** exp), "penalizacion": pen, "paciencia": pa, "umbral": um,
           "comparar_modelos": comp}
    errores = []
    if pa > ep and alg == "sgd":
        errores.append("La paciencia no puede ser mayor que el número de épocas.")
    if alg == "nb" and pen != "l2":
        st.info("La penalización solo aplica a SGD.")
    for e in errores:
        st.error(e)

    c1, c2 = st.columns([1, 1])
    if c1.button("↺ Restaurar valores por defecto"):
        st.session_state.reset += 1
        st.rerun()
    lanzar = c2.button("▶ Entrenar", type="primary", disabled=bool(errores))

    try:
        estado = llamar("GET", "/train/status")
    except ApiError as e:
        st.error(str(e))
        return
    if lanzar:
        try:
            llamar("POST", "/train", json=cfg)
        except ApiError as e:
            st.error(f"No se pudo iniciar: {e}")
            return
        estado = {"estado": "en_curso"}
    if estado.get("estado") == "en_curso":
        seguir_entrenamiento()
    elif estado.get("estado") == "error":
        st.error(f"El entrenamiento falló: {estado.get('mensaje_error')}")
    elif estado.get("estado") == "completado":
        resumen_entrenamiento(estado)


def curvas_vivas(h: list[dict]) -> go.Figure:
    d = pd.DataFrame(h)
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Pérdida (log-loss)", "F1"))
    if len(d):
        for col, nom, color, dash in (("logloss_train", "Entrenamiento", NEG, "solid"), ("logloss_val", "Validación", POS, "solid")):
            fig.add_trace(go.Scatter(x=d.epoca, y=d[col], name=nom, line=dict(color=color, dash=dash), legendgroup=nom), 1, 1)
        for col, nom, color in (("f1_train", "Entrenamiento", NEG), ("f1_val", "Validación", POS)):
            fig.add_trace(go.Scatter(x=d.epoca, y=d[col], name=nom, line=dict(color=color), legendgroup=nom, showlegend=False), 1, 2)
    fig.update_xaxes(title_text="Época")
    return tema(fig, 340)


def seguir_entrenamiento() -> None:
    barra, etapa, graf = st.progress(0.0), st.empty(), st.empty()
    while True:
        try:
            s = llamar("GET", "/train/status")
        except ApiError as e:
            st.error(str(e))
            return
        tot = max(s.get("epocas_total", 1), 1)
        barra.progress(min(s.get("epoca", 0) / tot, 1.0) if s.get("estado") == "en_curso" else 1.0)
        etapa.info(f"Etapa: **{s.get('etapa', '…')}** · época {s.get('epoca', 0)}/{tot}")
        graf.plotly_chart(curvas_vivas(s.get("historial", [])), key=f"viva{time.time()}")
        if s.get("estado") != "en_curso":
            break
        time.sleep(0.5)
    get_cache.clear()
    if s["estado"] == "completado":
        resumen_entrenamiento(s)
    else:
        st.error(f"El entrenamiento falló: {s.get('mensaje_error')}")


def resumen_entrenamiento(s: dict) -> None:
    r = s.get("resumen")
    if not r:
        return
    st.success("Entrenamiento completado")
    c = st.columns(4)
    c[0].metric("Mejor época", r["mejor_epoca"])
    c[1].metric("F1 (prueba)", f"{r['prueba']['f1']:.3f}")
    c[2].metric("Accuracy (prueba)", f"{r['prueba']['accuracy']:.3f}")
    c[3].metric("Tiempo total", f"{r['tiempo_total_s']:.1f} s")
    st.caption("Abre la página **Métricas** del menú lateral para ver el detalle.")


# ---------------- 3. Métricas ----------------
def pag_metricas() -> None:
    st.title("Métricas del modelo")
    m = metricas_o_none()
    if m is None:
        sin_modelo()
        return
    det = llamar("GET", "/metrics/detalle")
    p, v = m["prueba"], m["validacion"]
    cols = st.columns(6)
    for col, (k, nom) in zip(cols, (("accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Recall"),
                                    ("f1", "F1"), ("roc_auc", "AUC"), ("logloss", "Log-Loss"))):
        col.metric(nom, f"{p[k]:.3f}", f"{p[k] - v[k]:+.3f} vs val", delta_color="inverse" if k == "logloss" else "normal")
    st.download_button("⬇ Descargar metricas.json", io.BytesIO(__import__("json").dumps(m, ensure_ascii=False, indent=2).encode()),
                       "metricas.json", "application/json")
    tabs = st.tabs(["Matriz de confusión", "Curvas por época", "ROC / PR", "Correlación", "Palabras influyentes",
                    "Comparación", "Curva de aprendizaje", "Errores"])
    y = np.array(det["pred_test"]["y"]) if det.get("pred_test") else None
    pp = np.array(det["pred_test"]["p_positivo"]) if det.get("pred_test") else None

    with tabs[0]:
        u = st.slider("Umbral de decisión", 0.05, 0.95, float(m["config"]["umbral"]), 0.05)
        cm = confusion_matrix(y, (pp >= u).astype(int), labels=[0, 1])
        pct = cm / cm.sum(axis=1, keepdims=True) * 100
        txt = [[f"{cm[i, j]}<br>({pct[i, j]:.1f}%)" for j in range(2)] for i in range(2)]
        fig = go.Figure(go.Heatmap(z=pct, x=["Negativo", "Positivo"], y=["Negativo", "Positivo"], text=txt,
                                   texttemplate="%{text}", colorscale=[[0, PANEL], [1, POS]], zmin=0, zmax=100, showscale=False))
        fig.update_layout(xaxis_title="Predicho", yaxis_title="Real", yaxis_autorange="reversed")
        ancho(st.plotly_chart, tema(fig, 420))
    with tabs[1]:
        h = pd.DataFrame(det["historial"])
        fig = curvas_vivas(h.to_dict("records"))
        fig.add_vline(x=m["mejor_epoca"], line_dash="dot", line_color=TXT2, annotation_text="mejor época")
        ancho(st.plotly_chart, fig)
        st.caption("Si la pérdida de entrenamiento baja y la de validación sube → sobreajuste; por eso se conserva la mejor época.")
    with tabs[2]:
        fpr, tpr, _ = roc_curve(y, pp)
        pr, rc, th = precision_recall_curve(y, pp)
        fig = make_subplots(rows=1, cols=2, subplot_titles=(f"ROC (AUC={p['roc_auc']:.3f})", "Precisión/Recall vs umbral"))
        fig.add_trace(go.Scatter(x=fpr, y=tpr, name="Modelo", line=dict(color=POS)), 1, 1)
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Azar", line=dict(color=TXT2, dash="dot")), 1, 1)
        fig.add_trace(go.Scatter(x=th, y=pr[:-1], name="Precisión", line=dict(color=POS)), 1, 2)
        fig.add_trace(go.Scatter(x=th, y=rc[:-1], name="Recall", line=dict(color=NEG)), 1, 2)
        ancho(st.plotly_chart, tema(fig))
    with tabs[3]:
        c = llamar("GET", "/metrics/correlacion")
        z = np.array(c["matriz"])
        fig = go.Figure(go.Heatmap(z=z, x=c["variables"], y=c["variables"], zmin=-1, zmax=1, colorscale="RdBu",
                                   text=np.round(z, 2), texttemplate="%{text}"))
        fig.update_layout(yaxis_autorange="reversed")
        ancho(st.plotly_chart, tema(fig, 560))
    with tabs[4]:
        pal = det.get("palabras")
        if pal:
            fig = make_subplots(rows=1, cols=2, subplot_titles=("Empujan a Negativo", "Empujan a Positivo"))
            n_, p_ = pal["negativas"][::-1], pal["positivas"][::-1]
            fig.add_trace(go.Bar(x=[a["peso"] for a in n_], y=[a["termino"] for a in n_], orientation="h", marker_color=NEG, name="Negativo"), 1, 1)
            fig.add_trace(go.Bar(x=[a["peso"] for a in p_], y=[a["termino"] for a in p_], orientation="h", marker_color=POS, name="Positivo"), 1, 2)
            ancho(st.plotly_chart, tema(fig, 560))
    with tabs[5]:
        comp = m.get("comparacion_modelos")
        if comp:
            t = pd.DataFrame(comp)
            ancho(st.dataframe, t.round(4))
            fig = go.Figure(go.Bar(x=t["modelo"], y=t["f1_prueba"], marker_color=POS, text=t["f1_prueba"].round(3), name="F1 prueba"))
            ancho(st.plotly_chart, tema(fig, 320))
        else:
            st.info("Entrena con la casilla «Comparar con otros modelos» para ver esta tabla.")
        sel = det.get("seleccion_modelo")
        if sel:
            st.markdown("**Validación cruzada (5 pliegues, F1) — resultados de `model_selection`**")
            ancho(st.dataframe, pd.DataFrame(sel["modelos_cv"]).T[["f1_cv_media", "f1_cv_de", "mejores_params"]].astype(str))
            st.markdown("**Experimento de palabras vacías (F1 validación)**")
            st.json(sel["stopwords_f1_val"])
    with tabs[6]:
        sel = det.get("seleccion_modelo")
        if sel:
            c = pd.DataFrame(sel["curva_aprendizaje"])
            fig = go.Figure([go.Scatter(x=c.n_por_clase, y=c.f1_validacion, name="Validación", line=dict(color=POS)),
                             go.Scatter(x=c.n_por_clase, y=c.f1_entrenamiento, name="Entrenamiento", line=dict(color=NEG, dash="dash"))])
            fig.update_layout(xaxis_title="Reseñas por clase", yaxis_title="F1")
            ancho(st.plotly_chart, tema(fig))
            st.caption(sel["conclusion_curva"])
        else:
            st.info("Ejecuta `python -m src.model_selection` para generar la curva.")
    with tabs[7]:
        ae = det.get("analisis_errores")
        if ae:
            st.markdown(f"**Revisión manual de {ae['n_revisados']} errores más seguros**")
            ancho(st.dataframe, pd.DataFrame({k: {"casos": v["n"], "%": v["pct"]} for k, v in ae["categorias"].items()}).T)
            st.caption(ae["nota"])
        if det.get("errores"):
            ancho(st.dataframe, pd.DataFrame(det["errores"]))


# ---------------- 4. Simulador ----------------
EJEMPLOS = {"👍 Positiva": "Excelente producto, llegó rápido y funciona perfecto. Muy recomendable.",
            "👎 Negativa": "Pésima calidad, se rompió a la semana. Una estafa, no lo recomiendo.",
            "🚫 Con negación": "No funciona nada, jamás volvería a comprarlo.",
            "↔ Con contraste": "Bonito diseño pero se rompió al primer uso, una lástima."}


def poner_ejemplo(t: str) -> None:
    st.session_state["texto_sim"] = t


def pag_simulador() -> None:
    st.title("Simulador")
    if metricas_o_none() is None:
        sin_modelo()
        return
    st.session_state.setdefault("historial", [])
    cols = st.columns(len(EJEMPLOS))
    for c, (nom, t) in zip(cols, EJEMPLOS.items()):
        c.button(nom, on_click=poner_ejemplo, args=(t,))
    texto = st.text_area("Escribe una reseña", key="texto_sim", height=120, max_chars=5000)
    umbral = st.slider("Umbral", 0.05, 0.95, 0.5, 0.05)
    if st.button("Analizar", type="primary"):
        try:
            r = llamar("POST", "/predict", json={"texto": texto, "umbral": umbral})
        except ApiError as e:
            st.error(f"No se pudo analizar: {e}")
        else:
            st.session_state.historial = ([{"texto": texto[:80], "etiqueta": r["etiqueta"], "P(pos)": round(r["prob_positivo"], 3)}]
                                          + st.session_state.historial)[:10]
            mostrar_resultado(r)
    if st.session_state.historial:
        st.markdown("**Últimas consultas**")
        ancho(st.dataframe, pd.DataFrame(st.session_state.historial))
    st.divider()
    st.subheader("Análisis por lote")
    arch = st.file_uploader("CSV con una columna `texto`", type="csv")
    if arch is not None:
        d = pd.read_csv(arch)
        if "texto" not in d:
            st.error("El CSV debe tener una columna llamada `texto`.")
        else:
            if len(d) > 500:
                st.warning("Se analizarán solo las primeras 500 filas.")
                d = d.head(500)
            res = llamar("POST", "/predict/batch", json={"textos": d["texto"].astype(str).tolist(), "umbral": umbral}, timeout=120)["resultados"]
            d["etiqueta"] = [r.get("etiqueta", "error") for r in res]
            d["prob_positivo"] = [r.get("prob_positivo") for r in res]
            ancho(st.dataframe, d)
            vc = d["etiqueta"].value_counts()
            fig = go.Figure(go.Bar(x=vc.index, y=vc.values, marker_color=[POS if i == "Positivo" else NEG for i in vc.index]))
            ancho(st.plotly_chart, tema(fig, 280))
            st.download_button("⬇ Descargar predicciones", d.to_csv(index=False).encode("utf-8"), "predicciones.csv", "text/csv")


def mostrar_resultado(r: dict) -> None:
    color = POS if r["etiqueta"] == "Positivo" else NEG
    st.markdown(f"<h2 style='color:{color}'>{etiqueta(r['etiqueta'])}</h2>", unsafe_allow_html=True)
    c = st.columns([1, 1, 1])
    c[1].metric("Confianza", f"{r['confianza'] * 100:.1f}%")
    c[2].metric("Inferencia", f"{r['ms']:.1f} ms")
    g = go.Figure(go.Indicator(mode="gauge+number", value=r["prob_positivo"] * 100,
                               number={"suffix": "%"}, title={"text": "P(Positivo)"},
                               gauge={"axis": {"range": [0, 100]}, "bar": {"color": color},
                                      "threshold": {"line": {"color": TXT, "width": 3}, "value": r["umbral"] * 100}}))
    with c[0]:
        ancho(st.plotly_chart, tema(g, 230))
    t = r["top_terminos"]
    if t:
        t = t[::-1]
        fig = go.Figure(go.Bar(x=[a["contribucion"] for a in t], y=[a["termino"] for a in t], orientation="h",
                               marker_color=[POS if a["contribucion"] > 0 else NEG for a in t]))
        fig.update_layout(title="Contribución por término (azul → Positivo, rojo → Negativo)")
        ancho(st.plotly_chart, tema(fig, 320))
    st.caption(f"Texto limpio que vio el modelo: «{r['texto_limpio']}»")


# ---------------- 5. Dashboard ----------------
def pag_dashboard() -> None:
    st.title("Dashboard del dataset")
    base = get_cache("/dataset/summary")
    f = st.columns(3)
    est = f[0].selectbox("Estrellas", ["Todas", 1, 2, 4, 5])
    cat = f[1].selectbox("Categoría", ["Todas"] + base["categorias_disponibles"])
    cls = f[2].selectbox("Clase", ["Todas", "Negativo", "Positivo"])
    params = {}
    if est != "Todas":
        params["estrellas"] = est
    if cat != "Todas":
        params["categoria"] = cat
    if cls != "Todas":
        params["sentimiento"] = int(cls == "Positivo")
    s = get_cache("/dataset/summary", tuple(params.items()))
    c = st.columns(4)
    c[0].metric("Reseñas", f"{s['total']:,}")
    c[1].metric("Positivas", f"{s['positivas']:,}")
    c[2].metric("Negativas", f"{s['negativas']:,}")
    c[3].metric("Palabras (media)", f"{s['palabras_media']:.1f}" if s["palabras_media"] else "—")
    if not s["total"]:
        st.info("Ningún registro con esos filtros.")
        return
    a, b = st.columns(2)
    emb = (s.get("auditoria") or base.get("auditoria") or {}).get("embudo_limpieza")
    if emb:
        e = [x for x in emb if x["paso"] != "train_tras_balanceo"]
        fig = go.Figure(go.Bar(y=[x["paso"].replace("_", " ") for x in e][::-1], x=[x["filas"] for x in e][::-1],
                               orientation="h", marker_color=POS, text=[x["filas"] for x in e][::-1]))
        fig.update_layout(title="Embudo de limpieza")
        with a:
            ancho(st.plotly_chart, tema(fig, 320))
    fig = go.Figure(go.Bar(x=[f"{k}★" for k in s["estrellas"]], y=list(s["estrellas"].values()),
                           marker_color=[NEG if int(k) <= 2 else POS for k in s["estrellas"]]))
    fig.update_layout(title="Distribución por estrellas")
    with b:
        ancho(st.plotly_chart, tema(fig, 320))
    a, b = st.columns(2)
    if s.get("categorias"):
        t = pd.DataFrame(s["categorias"]).iloc[::-1]
        fig = go.Figure([go.Bar(y=t["categoria"], x=t["negativo"], orientation="h", name="Negativo", marker_color=NEG),
                         go.Bar(y=t["categoria"], x=t["positivo"], orientation="h", name="Positivo", marker_color=POS)])
        fig.update_layout(barmode="stack", title="Top categorías")
        with a:
            ancho(st.plotly_chart, tema(fig, 420))
    if s.get("histograma"):
        h = s["histograma"]
        x = h["bordes"][:-1]
        fig = go.Figure([go.Bar(x=x, y=h["negativo"], name="Negativo", marker_color=NEG),
                         go.Bar(x=x, y=h["positivo"], name="Positivo", marker_color=POS)])
        fig.update_layout(barmode="overlay", title="Longitud (palabras)", xaxis_title="Palabras")
        fig.update_traces(opacity=0.7)
        with b:
            ancho(st.plotly_chart, tema(fig, 420))
    st.subheader("Reseñas reales")
    pag = st.number_input("Página", 0, max(s["total"] // 25, 0), 0)
    m = llamar("GET", "/dataset/sample", params={**params, "n": 25, "pagina": int(pag)})
    ancho(st.dataframe, pd.DataFrame(m["filas"]))
    st.caption(f"{m['total_filtrado']:,} reseñas con estos filtros")


try:
    {"Inicio": pag_inicio, "Entrenamiento": pag_entrenamiento, "Métricas": pag_metricas,
     "Simulador": pag_simulador, "Dashboard": pag_dashboard}[pagina]()
except ApiError as e:
    st.error(str(e))
