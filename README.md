# Análisis de sentimientos en reseñas de Amazon (español)

Proyecto de Machine Learning 2026-2 — Fernanda Dayana Tana · Docente: Ing. Sergio Granizo MSc.
Clasifica reseñas como **Positivo** (4–5★) o **Negativo** (1–2★), con API (FastAPI) e interfaz (Streamlit).
Especificación completa: `ESPECIFICACION_PROYECTO_SENTIMIENTOS.md`.

## Dataset
Multilingual Amazon Reviews Corpus (Keung, Lu, Szarvas & Smith, 2020, EMNLP, 4563–4568), subconjunto en español,
copia de Kaggle. **Enlace de Kaggle y fecha de descarga: [COMPLETAR].** Uso académico/no comercial.
El archivo usado (`data/raw/amazon_reviews_es.csv`) tiene 8 000 reseñas balanceadas (columnas `reviewText, stars, sentiment, product_category`);
no incluye título, idioma ni reseñas de 3★, por lo que se usó la partición 70/15/15 estratificada (Opción B de la especificación).

## Instalación y ejecución
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m src.data_audit            # auditoría -> reports/auditoria_datos.json
python -m src.preprocessing         # limpieza y particiones -> data/processed/
python -m src.eda                   # figuras del EDA
python -m src.model_selection       # CV, rejillas, curva de aprendizaje (~1 min)
python -m src.train                 # entrenamiento por épocas, métricas y figuras
python -m src.validar_vocabulario   # 20 frases nuevas
python -m pytest -q                 # pruebas
python -m uvicorn src.api:app --port 8000     # Terminal 1 -> http://127.0.0.1:8000/docs
python -m streamlit run src/app.py --server.port 8501   # Terminal 2 -> http://localhost:8501
```
Atajos: `.\run.ps1 api | app | train | test`.

## Resultados (conjunto de prueba, 1 187 reseñas, usado una sola vez)
| Modelo | F1 CV (5 pliegues) | F1 prueba | ROC-AUC |
|---|---|---|---|
| Naive Bayes | 0,854 ± 0,009 | 0,842 | 0,922 |
| Regresión Logística | 0,864 ± 0,010 | 0,860 | 0,932 |
| SVM Lineal | 0,864 ± 0,009 | 0,858 | 0,930 |
| **SGD por épocas (modelo de la app)** | 0,862 ± 0,009 | **0,855** | **0,930** |

Accuracy 0,860 · mejor época 3 (early stopping en la 8) · vocabulario 3 653 términos.
Frases nuevas: 17/20 (85 %, cualitativo; son un borrador a reemplazar por frases propias). Detalle completo en `reports/`.

## Estructura
`src/` (código) · `tests/` (38 pruebas) · `data/` · `models/` · `reports/` (JSON, CSV, `figures/`).
Flujo: Streamlit → HTTP/JSON → FastAPI → `train.py` / `predictor.py` → `models/` y `reports/`.

## Capturas
Pendiente de insertar (ver sección 16.1 de la especificación).
