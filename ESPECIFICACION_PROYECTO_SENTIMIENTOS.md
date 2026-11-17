# Especificación Maestra del Proyecto — Para construir con Claude Code

**Proyecto:** Sistema Inteligente de Análisis de Sentimientos en Reseñas de Comercio Electrónico (Español)
**Asignatura:** Proyectos de Machine Learning — Período 2026-2
**Docente:** Ing. Sergio Granizo MSc.
**Autora (proyecto individual):** Fernanda Dayana Tana

> **Cómo usar este documento:** colócalo en la raíz de una carpeta vacía (o del repositorio) y pídele a Claude Code:
> *"Lee ESPECIFICACION_PROYECTO_SENTIMIENTOS.md y construye el proyecto completo siguiendo las fases en orden. Al terminar cada fase, ejecuta sus pruebas y muéstrame el resultado antes de seguir."*
>
> Este documento reemplaza la documentación anterior del proyecto. Es la única fuente de verdad.

---

## 0. Instrucciones para Claude Code (leer primero)

1. **Construye por fases** (sección 14). No pases a la siguiente fase sin que las pruebas de la actual pasen.
2. **Nunca inventes resultados.** Todas las cifras del informe (accuracy, F1, matrices, tiempos) deben salir de ejecutar el código sobre el dataset real. Donde el documento dice `[COMPLETAR]`, se rellena con la salida real.
3. **El dataset es real** (Amazon Reviews Multi, español). No generes reseñas con IA para entrenar ni evaluar. La **única** excepción son las *fixtures* pequeñas de `tests/`, usadas solo para probar que el código funciona, y deben estar marcadas como tales.
4. **Reproducibilidad:** `random_state = 42` en todo (splits, muestreo, modelos).
5. **Sin fuga de datos (data leakage):** el TF-IDF y cualquier transformación se ajustan **solo** con entrenamiento; la validación elige hiperparámetros/época; la prueba se usa **una sola vez** al final.
6. **Entorno:** Python 3.11+, scikit-learn ≥ 1.3 (usar `loss="log_loss"`, **no** `"log"`, que ya no existe). Fija versiones en `requirements.txt`.
7. **Rutas:** calcúlalas desde `Path(__file__)` para que el proyecto funcione sin importar desde qué carpeta se ejecute. Carpetas sin espacios ni tildes.
8. Código y comentarios en **español**, con docstrings y type hints. Formatea con `ruff`/`black` si está disponible.
9. Si algo de esta especificación es imposible o contradictorio con lo que encuentres en los datos, **detente y explícalo** en lugar de improvisar.

---

## 1. Requisitos del docente → cómo se cumplen (matriz de trazabilidad)

| # | Requisito (documento de indicaciones) | Cómo lo cumple el proyecto | Evidencia |
|---|---|---|---|
| 1 | Enfoque: análisis de sentimientos | Clasificación binaria Positivo/Negativo de reseñas en español | Toda la app |
| 2 | Dataset real con fuente bibliográfica | Multilingual Amazon Reviews Corpus (Keung et al., 2020), subconjunto `es` | Sección 4 + referencias |
| 3 | Aplicación en Python con interfaz gráfica | Frontend Streamlit con 5 páginas | Capturas 6–10 |
| 4a | **Entrenar el modelo, crear una API e implementarlo** | Backend **FastAPI** (`/train`, `/predict`, `/metrics`…); el entrenamiento se lanza desde la interfaz vía API | Swagger `/docs` + página Entrenamiento |
| 4b | **Configurar parámetros** | Formulario de hiperparámetros en la interfaz (épocas, α, n-gramas, max_features, tamaño de muestra, umbral…) | Captura de la página Entrenamiento |
| 4c | **Mostrar métricas del entrenamiento** | Curvas por época en vivo + página Métricas (matriz de confusión, ROC, correlación, comparación de modelos) | Página Métricas |
| 4d | **Recibir un nuevo caso** | Simulador: escribe una reseña y obtiene etiqueta, probabilidad y palabras que influyeron | Página Simulador |
| 5 | Repositorio público en GitHub con commits progresivos | Plan de commits por fase (sección 15) | Historial de GitHub |
| 6 | Presentación de 15–20 min | Guion de diapositivas (sección 17) | Diapositivas |
| — | Entregables: informe PDF, presentación, **póster** (rúbrica 20 %) | Estructura del informe (sección 16) y del póster (sección 18) | PDF, PPT, póster |
| — | "Evidencias del ejercicio en WebGoat" | Ver sección 19 (confirmar con el docente) | Correo/respuesta del docente |

---

## 2. Motivación y problema (para el informe y la exposición)

- **Problema:** las tiendas en línea reciben miles de reseñas en español. Leerlas a mano es lento, costoso e inconsistente. Las empresas (especialmente pymes latinoamericanas) necesitan detectar rápido productos con quejas, defectos o estafas.
- **¿Por qué es relevante?** El español es una de las lenguas más usadas en internet, pero la mayoría de herramientas y tutoriales de PLN están pensados para inglés. Las negaciones (*"no funciona"*, *"jamás volvería a comprar"*) y los modismos hacen que traducir al inglés pierda información.
- **Aporte de la IA:** un modelo que clasifica una reseña en milisegundos, indica su grado de certeza y **explica** qué palabras pesaron en la decisión, permitiendo priorizar la atención al cliente y el control de calidad.
- **¿Por qué este tema?** Problema real, datos reales disponibles, resultados interpretables y defendibles frente al docente (no es una "caja negra").

---

## 3. Alcance

**Incluye:**
- Clasificación binaria (Negativo = 1–2 estrellas, Positivo = 4–5 estrellas) de reseñas en español.
- Pipeline completo: ingesta → auditoría/limpieza de ruido → análisis exploratorio → selección de modelo → entrenamiento por épocas → evaluación → API → interfaz.
- Explicabilidad por predicción (contribución de cada término).

**No incluye (se declara como limitación y trabajo futuro):**
- Clase neutral (3 estrellas): se excluye porque es ambigua; se documenta cuántas se descartan.
- Detección de sarcasmo/ironía, análisis por aspectos (precio, envío, calidad).
- Modelos Transformer (BETO); se mencionan como trabajo futuro y, opcionalmente, como experimento extra (sección 8.6).
- Despliegue en la nube con usuarios reales (se ejecuta localmente).

---

## 4. Dataset

### 4.1 Fuente y citación
- **Nombre:** The Multilingual Amazon Reviews Corpus (MARC), subconjunto en español.
- **Cita APA:** Keung, P., Lu, Y., Szarvas, G., & Smith, N. A. (2020). The Multilingual Amazon Reviews Corpus. En *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)* (pp. 4563–4568). Association for Computational Linguistics.
- **Características documentadas por los autores:** reseñas en 6 idiomas recogidas entre 2015 y 2019; para cada idioma hay 200 000 reseñas de entrenamiento, 5 000 de validación y 5 000 de prueba; el corpus está **balanceado por estrellas** (20 % para cada calificación de 1 a 5). Campos: `review_id`, `product_id`, `reviewer_id`, `stars`, `review_body`, `review_title`, `language`, `product_category`.
- **Disponibilidad:** Amazon retiró la distribución oficial (en Hugging Face figura como *defunct*). Se usa una copia publicada en **Kaggle**. En el informe se debe indicar **el enlace exacto de Kaggle usado y la fecha de descarga** `[COMPLETAR]`, además de citar el paper original.
- **Licencia:** uso académico/no comercial bajo las condiciones de Amazon; prohibido intentar identificar a los autores de las reseñas. Mencionarlo en el informe.

### 4.2 Ubicación de los archivos
- Colocar los CSV crudos en `data/raw/` (por ejemplo `train.csv`, `validation.csv`, `test.csv`, o un único `amazon_reviews_es.csv`).
- El cargador debe:
  - aceptar uno o varios CSV y concatenarlos;
  - normalizar nombres de columnas (`stars`→`estrellas`, `review_body`→`cuerpo`, `review_title`→`titulo`, `language`→`idioma`, `product_category`→`categoria`);
  - si existen los tres archivos oficiales, **conservar de qué split oficial viene cada fila** (columna `split_origen`);
  - fallar con un mensaje claro si faltan `stars` o `review_body`.

### 4.3 ¿Cuántos datos usar? (decisión justificada con evidencia)
La versión anterior usaba solo 8 000 reseñas (4 000 por clase), es decir, un 5 % de lo disponible. Tras descartar las de 3 estrellas, el corpus en español ofrece unas **160 000 reseñas útiles** (≈ 80 000 por clase), así que **no hace falta conseguir datos externos: hace falta usar más de los que ya hay**.

La decisión no se toma "a ojo": se hace con una **curva de aprendizaje** (sección 8.4), entrenando con 2 000, 5 000, 10 000, 20 000, 40 000 y 80 000 reseñas por clase y midiendo el F1 en validación.
- Si la curva sigue subiendo al final → usar el máximo.
- Si se aplana → usar el tamaño donde se estabiliza (a partir de ahí, más datos solo agregan tiempo de entrenamiento).
- Valor por defecto mientras tanto: **20 000 por clase (40 000 en total)**, configurable desde la interfaz.

Esta curva es la respuesta formal a *"¿hacen falta más datos?"* y va en el informe.

### 4.4 Partición de datos (sin fuga)
**Opción A (preferida, si se tienen los tres CSV oficiales):** respetar los splits oficiales del corpus:
- `train` oficial → entrenamiento (submuestreado según 4.3);
- `validation` oficial → elegir época e hiperparámetros;
- `test` oficial → evaluación final, una sola vez.

Así los resultados son comparables con la literatura y no hay forma de contaminar la prueba.

**Opción B (si solo hay un archivo):** partición estratificada **70 % / 15 % / 15 %** (entrenamiento / validación / prueba), con `stratify=sentimiento` y `random_state=42`.

En ambos casos:
- El **balanceo** (submuestreo) se aplica solo a entrenamiento. Validación y prueba ya están balanceadas por diseño (corpus 20 % por estrella → al quitar las de 3 queda 50/50).
- Se verifica y reporta la proporción de clases en cada split.
- Se controla que **no haya duplicados entre splits** (mismo texto normalizado en train y test). Si los hay, se eliminan de train.

---

## 5. Auditoría de calidad y tratamiento del ruido

Antes de limpiar, se **mide**. Cada paso registra cuántas filas elimina o modifica, en un `reports/auditoria_datos.json` y en una tabla del informe ("embudo de limpieza").

### 5.1 Auditoría inicial (`src/data_audit.py`)
Reportar, por split:
- total de filas; nulos por columna; textos vacíos o solo espacios;
- duplicados exactos de texto y duplicados tras normalizar;
- distribución de estrellas y de categorías de producto;
- idioma declarado vs. idioma detectado (muestra de 2 000 con `langdetect` o `lingua`, si está instalado);
- longitud en caracteres y palabras (mínimo, percentiles 1/25/50/75/99, máximo);
- reseñas con mojibake o caracteres raros (`Ã©`, `�`), HTML, URLs, emojis;
- reseñas muy cortas (< 3 palabras) y extremadamente largas (> p99).

### 5.2 Tipos de ruido y tratamiento

| Tipo de ruido | Ejemplo | Tratamiento | Se documenta |
|---|---|---|---|
| Nulos o vacíos | `""`, `NaN` | Eliminar | Nº eliminadas |
| Duplicados | misma reseña repetida | Eliminar (dentro de cada split y entre splits) | Nº eliminadas |
| Idioma incorrecto | reseña en inglés o portugués en el set `es` | Filtrar por `language == "es"`; si se detecta otro idioma con alta confianza, eliminar | Nº y % |
| Codificación rota | `Ã©` en lugar de `é` | Intentar reparar (`ftfy` si está disponible); si no se puede, eliminar | Nº |
| HTML / URLs | `<br>`, `http://…` | Quitar con expresiones regulares | Nº afectadas |
| Letras repetidas | `buenooooo`, `malísimoooo` | Reducir 3+ repeticiones a una (`bueno`) | Nº afectadas |
| Números, signos y emojis | `10/10`, `!!!`, 😡 | Quitar del texto que va al modelo **después** de extraer variables derivadas (nº de exclamaciones, etc.) | — |
| Reseñas muy cortas | `ok`, `bien` | Mantener si tienen ≥ 2 tokens útiles tras limpiar; si no, eliminar | Nº |
| Reseñas extremadamente largas | > p99 | **No eliminar**; truncar a p99 tokens solo si afecta al rendimiento (justificar) | Decisión |
| Neutrales (3 estrellas) | opiniones mixtas | Excluir y contar | Nº |
| **Ruido de etiqueta** | 2 estrellas con texto positivo, o 5 estrellas con queja | No se puede limpiar automáticamente sin sesgar; se **estima** en el análisis de errores (sección 9.5) y se documenta como limitación | % estimado en muestra revisada |

### 5.3 Normalización lingüística (texto que entra al modelo)
Orden de pasos en `limpiar_texto()`:
1. Normalización Unicode NFC y minúsculas.
2. Quitar URLs y HTML.
3. Reducir letras repetidas.
4. Quitar números y signos, **conservando** tildes, `ü` y `ñ`.
5. Colapsar espacios.
6. Quitar palabras vacías con una lista propia en español **que NO elimine**:
   - negaciones: *no, ni, nunca, jamás, tampoco, nada, nadie, ningún/ninguno/a, sin*;
   - contrastes: *pero, aunque, sino*;
   - intensificadores: *muy, mucho/a/os/as, poco/a, más, menos, demasiado, bastante*.

   (La lista estándar de NLTK sí los elimina, y eso destruye frases como *"no funciona"*. Explicar esto en la defensa.)
7. Unir título y cuerpo como `titulo + ". " + cuerpo`: el título suele ser la frase más cargada de sentimiento.

**Experimento obligatorio (justifica las decisiones):** comparar en validación (a) texto sin quitar palabras vacías, (b) con la lista estándar, (c) con la lista propia que conserva negaciones. Reportar el F1 de cada opción en una tabla.

### 5.4 Variables derivadas (para la matriz de correlación y el EDA)
Calculadas sobre el texto **original**, antes de limpiar:
`longitud_caracteres`, `num_palabras`, `num_exclamaciones`, `num_interrogaciones`, `proporcion_mayusculas`, `num_negaciones`, `estrellas`, `sentimiento`.

### 5.5 Salida
- `data/processed/train.csv`, `val.csv`, `test.csv` con las columnas: `texto`, `texto_limpio`, `estrellas`, `sentimiento`, `categoria`, `split_origen`, variables derivadas.
- `reports/auditoria_datos.json` con el embudo de limpieza.

---

## 6. Análisis exploratorio de datos (EDA)

Generar `reports/figures/` (PNG a 300 DPI) y mostrar lo mismo en la página **Dashboard** de la app:
1. **Embudo de limpieza:** registros crudos → tras cada filtro → final.
2. **Balance de clases** por split.
3. **Distribución por estrellas** (1, 2, 4, 5).
4. **Distribución por categoría de producto** (top 15) y proporción Positivo/Negativo en cada una.
5. **Histograma de longitud** (palabras) por clase.
6. **Matriz de correlación de Pearson** de las variables derivadas + `sentimiento` (y, tras entrenar, `P(Positivo)` del modelo).
7. **Términos más frecuentes por clase** (unigramas y bigramas) tras la limpieza.

> Nota para la defensa: una matriz de correlación sobre las ~20 000 columnas TF-IDF no es interpretable. Por eso la correlación se hace sobre las **variables derivadas**, y la interpretabilidad del modelo se muestra con los **coeficientes aprendidos** (sección 9.4).

---

## 7. Representación del texto (variable X)

- **TF-IDF** (`TfidfVectorizer`) ajustado **solo con entrenamiento**.
- Valores de partida: `ngram_range=(1, 2)`, `max_features=20000`, `min_df=2`, `max_df=0.95`, `sublinear_tf=True`.
- Los n-gramas (2–3 palabras) capturan negaciones y expresiones: *"no funciona"*, *"deja mucho que desear"*.
- Se elige `(1,2)` vs `(1,3)` y `max_features` por validación (sección 8.2), no por intuición.

**Fundamento matemático (para la defensa):**
- $\text{tfidf}(t,d) = \text{tf}(t,d) \cdot \left(\ln\frac{1+N}{1+\text{df}(t)} + 1\right)$ (con `sublinear_tf`, $\text{tf} = 1 + \log(\text{frecuencia})$), y cada vector se normaliza con L2.
- **X:** matriz dispersa de $n \times p$ (p ≈ 20 000).
- **Y:** binaria, 0 = Negativo y 1 = Positivo.

---

## 8. Selección de modelo y entrenamiento

### 8.1 Modelos candidatos (todos sobre el mismo TF-IDF)

| Modelo | Por qué se incluye |
|---|---|
| `DummyClassifier(strategy="most_frequent")` | Línea base: demuestra que el modelo aprende algo (≈ 50 % por el balance) |
| Naive Bayes Multinomial | Clásico en texto, muy rápido |
| Regresión Logística (`lbfgs`/`liblinear`) | Lineal, probabilística, interpretable |
| SVM lineal (`LinearSVC`) | Suele ser el más fuerte en TF-IDF (Joachims, 1998) |
| **SGDClassifier(`loss="log_loss"`)** | Regresión logística optimizada por descenso de gradiente estocástico; permite **entrenar por épocas** con `partial_fit`, mostrar curvas y hacer early stopping |

### 8.2 Búsqueda de hiperparámetros
- **Validación cruzada estratificada de 5 pliegues** sobre **entrenamiento** (`GridSearchCV` o `RandomizedSearchCV` con `Pipeline(tfidf → modelo)`, para que el TF-IDF se reajuste en cada pliegue y no haya fuga).
- Métrica de selección: **F1** (macro F1 equivale con clases balanceadas; reportar ambas).
- Rejillas sugeridas (pequeñas, para que corran en una laptop):
  - TF-IDF: `ngram_range ∈ {(1,1),(1,2),(1,3)}`, `max_features ∈ {10k, 20k, 50k}`, `min_df ∈ {2, 5}`.
  - NB: `alpha ∈ {0.1, 0.5, 1.0}`.
  - LogReg: `C ∈ {0.5, 1, 4, 10}`.
  - LinearSVC: `C ∈ {0.1, 0.5, 1}`.
  - SGD: `alpha ∈ {1e-6, 1e-5, 1e-4}`, `penalty ∈ {l2, elasticnet}`.
- Reportar en una tabla: media ± desviación estándar del F1 en CV y tiempo de entrenamiento.
- Opcional: **prueba de McNemar** entre los dos mejores modelos sobre validación, para decir si la diferencia es estadísticamente significativa.

### 8.3 Entrenamiento por épocas del modelo principal (SGD)
- Por cada época: barajar el entrenamiento (semilla fija) → `partial_fit(X, y, classes=[0,1])` → calcular sobre entrenamiento **y** validación: log-loss, accuracy y F1.
- **Checkpointing:** guardar (`deepcopy`) el modelo de la época con mayor F1 en **validación** (desempate: menor log-loss).
- **Early stopping:** detener si pasan `paciencia` épocas (por defecto 5) sin mejorar. Máximo de épocas configurable (por defecto 50).
- Diagnóstico a explicar en el informe a partir de las curvas:
  - las dos pérdidas bajan → aprendiendo;
  - la de entrenamiento baja y la de validación sube → **sobreajuste** (por eso se elige la mejor época);
  - ambas altas → subajuste (más n-gramas, menor α).
- Si en la búsqueda de 8.2 la SVM o la LogReg superan claramente a SGD, se documenta con honestidad. La app puede usar el mejor modelo; las curvas por época se muestran con SGD (o con LogReg vía `warm_start` y `max_iter` incremental). Elegir y justificar.

### 8.4 Curva de aprendizaje (cantidad de datos)
- Con los hiperparámetros elegidos, entrenar con 2k, 5k, 10k, 20k, 40k y 80k reseñas por clase y medir el F1 en **validación** y el tiempo.
- Gráfica `8_curva_aprendizaje.png` con una conclusión explícita: *"a partir de N reseñas el F1 mejora menos de 0,5 puntos"* `[COMPLETAR con datos reales]`.

### 8.5 Umbral de decisión
- Por defecto 0,5. Mostrar la curva Precision–Recall vs. umbral en validación.
- Desde la app se puede ajustar el umbral (por ejemplo, para detectar más negativas a costa de falsos positivos) y ver cómo cambia la matriz de confusión.

### 8.6 Experimento opcional (solo si hay GPU o tiempo): BETO
- Ajuste fino de `dccuchile/bert-base-spanish-wwm-cased` (Cañete et al., 2020) con 1–2 épocas sobre una muestra.
- Es solo para comparación en la tabla y para "trabajos futuros". **No es requisito** y no debe bloquear el proyecto.

### 8.7 Persistencia
- `models/modelo.joblib` con `{modelo, umbral, config, fecha, version_sklearn, metricas_resumen}`.
- `models/vectorizador.joblib`.
- Usar `joblib` (no `pickle` a secas). Al cargar, comparar la versión de sklearn y avisar si difiere.

---

## 9. Evaluación (conjunto de prueba, una sola vez)

### 9.1 Métricas
Accuracy, Precision, Recall, F1 (por clase y macro), ROC-AUC, PR-AUC y Log-Loss, junto con el `classification_report` completo. Guardar todo en `reports/metricas.json`.

### 9.2 Matrices obligatorias
1. **Matriz de confusión** (conteos y % por fila) en prueba → `2_matriz_confusion.png`.
2. **Matriz de correlación** (sección 6) → `5_matriz_correlacion.png`.
3. Opcional: matriz de confusión por categoría de producto (en qué categorías falla más).

### 9.3 Curvas
`1_evolucion_epocas.png` (pérdida, accuracy y F1 por época, marcando la mejor), `3_curva_roc.png`, `curva_precision_recall.png` y `curva_calibracion.png` (¿la probabilidad de 0,8 significa realmente un 80 %?).

### 9.4 Interpretabilidad
`4_palabras_influyentes.png`: los 20 términos con mayor peso positivo y los 20 con mayor peso negativo (coeficientes aprendidos).

### 9.5 Análisis de errores (clave para un buen informe)
- Exportar `reports/errores.csv` con los falsos positivos y falsos negativos más "seguros" (mayor probabilidad equivocada).
- Revisar manualmente al menos 30 y clasificarlos en: sarcasmo, contraste ("bonito pero se rompió"), **etiqueta ruidosa** (estrellas que no coinciden con el texto), reseña sobre el envío y no sobre el producto, y otros.
- Reportar la tabla con los porcentajes. Esto estima el **ruido de etiqueta** del dataset y justifica el techo de rendimiento.

### 9.6 Validación con vocabulario nuevo
- `src/validar_vocabulario.py`: 20 frases **escritas por la autora** (no por IA), 10 positivas y 10 negativas, con negaciones (*"jamás volvería a pedirlo"*), dobles negaciones (*"sin ninguna queja"*), contrastes (*"no le tenía fe… pero me sorprendió"*) y vocabulario de quejas (*"defecto de fábrica"*, *"me estafaron"*).
- Imprimir una tabla: frase | esperado | predicho | probabilidad | ✓/✗, y el % de aciertos.
- Se presenta como **prueba cualitativa**, no como métrica del modelo (20 frases no son estadísticamente representativas).

### 9.7 Tabla final de resultados (el informe la llena con datos reales)

| Modelo | F1 CV (media ± de) | F1 validación | F1 prueba | ROC-AUC prueba | Tiempo (s) |
|---|---|---|---|---|---|
| Dummy | `[COMPLETAR]` | | | | |
| Naive Bayes | `[COMPLETAR]` | | | | |
| Regresión Logística | `[COMPLETAR]` | | | | |
| SVM Lineal | `[COMPLETAR]` | | | | |
| SGD por épocas | `[COMPLETAR]` | | | | |

---

## 10. Arquitectura del software

```text
sentimiento-resenas-es/
├── data/
│   ├── raw/                     # CSV de Kaggle (NO se sube a GitHub; ver .gitignore)
│   └── processed/               # train.csv, val.csv, test.csv (no se suben si son pesados)
├── models/                      # modelo.joblib, vectorizador.joblib
├── reports/
│   ├── figures/                 # PNG a 300 DPI para el informe
│   ├── auditoria_datos.json
│   ├── metricas.json
│   ├── historial_epocas.csv
│   └── errores.csv
├── src/
│   ├── __init__.py
│   ├── config.py                # rutas + dataclass TrainConfig con validación de rangos
│   ├── data_loader.py           # carga de CSV crudos y normalización de columnas
│   ├── data_audit.py            # auditoría de calidad (sección 5.1)
│   ├── preprocessing.py         # limpieza, etiquetado, balanceo, variables derivadas, splits
│   ├── eda.py                   # gráficas exploratorias
│   ├── model_selection.py       # CV + búsqueda de hiperparámetros + curva de aprendizaje
│   ├── train.py                 # entrenamiento por épocas, checkpointing, evaluación, guardado
│   ├── plots.py                 # todas las gráficas del modelo
│   ├── predictor.py             # inferencia + explicación por término
│   ├── validar_vocabulario.py   # 20 frases nuevas
│   ├── api.py                   # FastAPI (backend)
│   └── app.py                   # Streamlit (frontend), consume la API por HTTP
├── tests/
│   ├── conftest.py              # fixture pequeña (marcada como datos de prueba)
│   ├── test_preprocessing.py
│   ├── test_train.py
│   ├── test_predictor.py
│   └── test_api.py
├── .streamlit/config.toml       # tema oscuro corporativo
├── .gitignore
├── requirements.txt
├── Makefile o run.ps1           # comandos rápidos (Windows: run.ps1)
├── README.md
└── ESPECIFICACION_PROYECTO_SENTIMIENTOS.md
```

**Flujo:** `Streamlit (frontend) ──HTTP/JSON──► FastAPI (backend) ──► src/train.py · src/predictor.py ──► models/ · reports/`

La interfaz **no** importa el modelo directamente: todo pasa por la API. Así se cumple "backend y frontend" y "crear una API con el modelo".

---

## 11. API REST (FastAPI) — `src/api.py`

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/health` | Estado del servicio y si hay un modelo cargado |
| GET | `/config/default` | Hiperparámetros por defecto y sus rangos válidos (para construir el formulario) |
| POST | `/train` | Recibe `TrainConfig` (JSON), valida los rangos y lanza el entrenamiento en un **hilo en segundo plano**. Devuelve `job_id`. Si ya hay un entrenamiento en curso, responde **409** |
| GET | `/train/status` | Progreso: etapa actual, época actual/total, historial parcial (para las curvas en vivo), estado (`en_curso`/`completado`/`error`) y mensaje de error |
| GET | `/metrics` | Contenido de `metricas.json` del último modelo |
| GET | `/dataset/summary` | Resumen del dataset y del embudo de limpieza para el Dashboard |
| GET | `/dataset/sample?estrellas=1&n=50` | Muestra filtrada para la tabla exploratoria |
| POST | `/predict` | `{"texto": "...", "umbral": 0.5}` → etiqueta, prob_positivo, confianza, top términos con su contribución y ms de inferencia |
| POST | `/predict/batch` | Lista de textos (máx. 500) → lista de predicciones |
| GET | `/model/info` | Fecha de entrenamiento, config usada y métricas resumidas |

Requisitos de la API:
- Modelos Pydantic para petición y respuesta, con validación (texto no vacío, máx. 5 000 caracteres, umbral entre 0,05 y 0,95).
- Errores claros en español (400 parámetros inválidos, 404 si no hay modelo, 409 si ya se está entrenando).
- Tras entrenar, recargar el predictor en memoria sin reiniciar el servidor.
- Documentación automática en `http://127.0.0.1:8000/docs` (sirve como captura de evidencia de la API).
- CORS abierto solo a `localhost`.

---

## 12. Interfaz gráfica (Streamlit) — UX/UI

### 12.1 Identidad visual
- **Tema oscuro corporativo:** fondo negro medianoche `#0E1117`, paneles azul marino `#14213D`, acento vino tinto `#7B2D3B` en botones y encabezados, y texto `#F0EFEC` / `#C3C2B7`.
- **Colores de datos (constantes en toda la app y en el informe):** Positivo = **azul** `#3987E5`; Negativo = **rojo** `#E66767`. Las magnitudes van en una escala de un solo azul y las correlaciones en una escala divergente rojo↔azul con gris en el centro. No usar arcoíris.
- Gráficas interactivas con **Plotly** (tooltip al pasar el mouse) y leyenda siempre visible cuando hay 2 o más series.
- Tipografía legible; las etiquetas nunca dependen solo del color (siempre texto + ícono: ✅ Positivo / ⚠️ Negativo).
- Configurar `.streamlit/config.toml` con el tema. Barra lateral con logo/título, navegación e indicador de estado de la API (🟢 conectada / 🔴 sin conexión).

### 12.2 Páginas

**1. Inicio**
- Título, descripción breve del problema, cómo usar la app y tarjetas KPI del modelo actual (Accuracy, F1, AUC de prueba, fecha de entrenamiento).
- Si no hay modelo, mostrar un aviso con un botón que lleve a "Entrenamiento".

**2. Entrenamiento (requisitos 4a, 4b y 4c)**
- Formulario de parámetros agrupado en secciones plegables:
  - **Datos:** reseñas por clase (slider de 2 000 a 80 000) y % de validación/prueba (solo en la Opción B).
  - **TF-IDF:** n-grama máximo (1–3), `max_features` (5k–50k) y `min_df`.
  - **Modelo:** algoritmo (SGD / LogReg / SVM / NB), épocas (1–100), α (escala logarítmica), penalización, tasa de aprendizaje, paciencia del early stopping y umbral.
  - Casilla "comparar con otros modelos".
- Botones **Restaurar valores por defecto** y **Entrenar**.
- Durante el entrenamiento: barra de progreso, etapa actual y **curvas de pérdida y F1 actualizándose época a época** (consultando `/train/status` cada ~0,5 s).
- Al terminar: resumen (mejor época, F1/Accuracy en prueba, tiempo total) y un botón "Ver métricas".
- Validación en la interfaz de rangos y combinaciones inválidas, antes de llamar a la API.

**3. Métricas (requisito 4c)**
- Fila de KPIs (Accuracy, Precision, Recall, F1, AUC, Log-Loss) en prueba, con la diferencia respecto a validación.
- Pestañas:
  - **Matriz de confusión:** heatmap interactivo con conteos y %, y un slider de umbral que la recalcula en vivo.
  - **Curvas por época.**
  - **ROC / PR.**
  - **Matriz de correlación.**
  - **Palabras influyentes.**
  - **Comparación de modelos:** tabla y barras.
  - **Curva de aprendizaje.**
  - **Análisis de errores:** tabla de los errores más seguros.
- Botón para descargar `metricas.json`.

**4. Simulador (requisito 4d)**
- Área de texto para escribir una reseña, botón **Analizar** y 3–4 botones de ejemplo rápido (una reseña positiva, una negativa, una con negación y una con contraste).
- Resultado:
  - etiqueta grande con ícono;
  - medidor (gauge) de P(Positivo) con el umbral marcado;
  - % de confianza;
  - tiempo de inferencia en ms;
  - **barras de contribución por término** (azul empuja a positivo, rojo a negativo);
  - texto limpio que vio el modelo (transparencia).
- **Análisis por lote:** subir un CSV con una columna `texto` → tabla de resultados + gráfico de distribución + botón para descargar el CSV con las predicciones.
- Historial de las últimas 10 consultas en la sesión.

**5. Dashboard del dataset**
- Embudo de limpieza, balance de clases, distribución por estrellas y por categoría, e histograma de longitudes.
- Filtros en una sola fila arriba (estrellas 1–5, categoría, clase) que recalculan los KPIs y las gráficas.
- Tabla exploratoria paginada de reseñas reales.

### 12.3 Calidad de UX
- Mensajes de error amigables (por ejemplo, "La API no está encendida: ejecuta `uvicorn src.api:app`").
- `st.spinner` en las operaciones largas y `st.cache_data` para lecturas de la API que no cambian.
- Nada se rompe si todavía no hay modelo entrenado.
- Compatibilidad con versiones recientes de Streamlit: usar `width="stretch"` en lugar del parámetro obsoleto `use_container_width` (si la versión instalada no lo soporta, hacer una función auxiliar con respaldo).

---

## 13. Pruebas automatizadas (pytest)

`tests/conftest.py` crea una fixture pequeña (~200 filas) **claramente marcada como datos sintéticos de prueba de software**, nunca usada para reportar métricas.

| Archivo | Qué verifica |
|---|---|
| `test_preprocessing.py` | `limpiar_texto` conserva tildes, ñ y negaciones ("no", "nunca", "sin"); quita URLs, HTML y números; reduce "buenooo" a "bueno"; mapea 1–2→0, 4–5→1 y descarta 3; el balanceo deja 50/50; no hay duplicados entre splits; el cargador falla con un mensaje claro si faltan columnas |
| `test_train.py` | El entrenamiento corre con la fixture; la mejor época está en el rango [1, épocas]; early stopping detiene antes del máximo cuando no hay mejora; el TF-IDF se ajusta solo con entrenamiento (vocabulario ⊆ términos de train); se generan los artefactos y todas las figuras; `TrainConfig` rechaza valores fuera de rango |
| `test_predictor.py` | La probabilidad está en [0, 1]; la etiqueta es coherente con el umbral; las contribuciones vienen ordenadas por magnitud; el texto vacío lanza un error |
| `test_api.py` | `/health` responde 200; `/predict` valida la entrada (422/400); `/train` responde 409 si ya hay uno en curso; flujo completo con `TestClient`: entrenar → status completado → predict |
| `validar_vocabulario.py` | Con el modelo real: ≥ 80 % de aciertos en las 20 frases (se omite si no hay modelo entrenado) |

Comando: `pytest -q`. Agregar cobertura (`pytest --cov=src`) si está disponible y reportar el porcentaje.

---

## 14. Fases de construcción (orden para Claude Code)

| Fase | Entregable | Criterio para terminar |
|---|---|---|
| 1 | Estructura, `config.py`, `requirements.txt`, `.gitignore`, README básico | `python -c "import src"` funciona |
| 2 | `data_loader.py` + `data_audit.py` | Auditoría impresa y `auditoria_datos.json` generado con los datos reales |
| 3 | `preprocessing.py` + splits | `test_preprocessing.py` pasa; embudo de limpieza documentado |
| 4 | `eda.py` | Figuras del EDA generadas y revisadas visualmente |
| 5 | `model_selection.py` (CV, rejilla, curva de aprendizaje, experimento de palabras vacías) | Tablas de CV y curva de aprendizaje con conclusión |
| 6 | `train.py` + `plots.py` | `test_train.py` pasa; métricas en prueba impresas; figuras 1–8 generadas |
| 7 | `predictor.py` + `validar_vocabulario.py` + análisis de errores | `test_predictor.py` pasa; ≥ 80 % en las 20 frases; `errores.csv` |
| 8 | `api.py` | `test_api.py` pasa; `/docs` accesible |
| 9 | `app.py` + tema | Las 5 páginas funcionan con la API encendida; sin errores en consola |
| 10 | README final + rellenar `[COMPLETAR]` del informe con resultados reales | Todos los tests pasan; revisión de las capturas |

---

## 15. Repositorio y commits progresivos

- Repositorio **público** en GitHub, rama `main`.
- `.gitignore`: `data/raw/`, `__pycache__/`, `.venv/`, `*.joblib` pesados (o subirlos con Git LFS si el docente quiere ver el modelo), `.ipynb_checkpoints/`.
- **Un commit por fase**, a medida que se va construyendo (no un único commit final). Mensajes sugeridos:
  1. `feat: estructura del proyecto y configuración`
  2. `feat: carga y auditoría de calidad del dataset`
  3. `feat: limpieza de ruido, etiquetado, balanceo y particiones`
  4. `feat: análisis exploratorio y matriz de correlación`
  5. `feat: selección de modelo con validación cruzada y curva de aprendizaje`
  6. `feat: entrenamiento por épocas con checkpointing y métricas`
  7. `feat: predictor explicable y validación con vocabulario nuevo`
  8. `feat: API REST con FastAPI`
  9. `feat: interfaz Streamlit con entrenamiento, métricas, simulador y dashboard`
  10. `docs: README, resultados e informe`
- El README debe incluir: descripción, cita del dataset y enlace de Kaggle, instalación, cómo descargar los datos, comandos para ejecutar, capturas y resultados.

**Comandos de ejecución (para el README):**
```bash
python -m venv .venv
.venv\Scripts\activate              # Windows  (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt
python -m src.data_audit
python -m src.preprocessing
python -m src.model_selection        # opcional, tarda más
python -m src.train
python -m src.validar_vocabulario
pytest -q
uvicorn src.api:app --reload         # Terminal 1 → http://127.0.0.1:8000/docs
streamlit run src/app.py             # Terminal 2 → http://localhost:8501
```

---

## 16. Estructura del informe PDF

1. Portada (asignatura, docente, período, autora).
2. Resumen (150–200 palabras con el resultado principal).
3. Objetivos (general y específicos, tomados de las indicaciones).
4. Motivación y planteamiento del problema (sección 2).
5. Estado del arte (sección 20, con 6–8 referencias comentadas en 2–3 líneas cada una).
6. Alcance y limitaciones (sección 3).
7. Dataset: fuente, cita, licencia, descripción y **embudo de limpieza** (secciones 4–5).
8. Análisis exploratorio (sección 6) con figuras.
9. Metodología: representación TF-IDF, fórmulas, modelos candidatos, validación cruzada, entrenamiento por épocas y partición sin fuga (secciones 7–8).
10. Resultados: tabla comparativa, curvas por época, matriz de confusión, ROC, correlación, palabras influyentes, curva de aprendizaje, análisis de errores y vocabulario nuevo (sección 9).
11. Arquitectura del sistema y API (secciones 10–11).
12. Interfaz y ejercicio práctico (sección 12) con capturas.
13. Conclusiones (3–5, cada una apoyada en un número real).
14. Recomendaciones y trabajos futuros: clase neutral, BETO/Transformers, análisis por aspectos, detección de sarcasmo, despliegue en la nube, reentrenamiento con reseñas nuevas.
15. Referencias (APA 7).
16. Anexos: enlace al repositorio, capturas de commits y salida de `pytest`.

### 16.1 Capturas que hay que tomar (después de construir)
1. VS Code con el árbol de carpetas + terminal de `data_audit` (embudo de limpieza).
2. Terminal de `preprocessing` (registros finales y balance).
3. Terminal de `model_selection` (tabla de CV).
4. Terminal de `train` (épocas, mejor época, métricas de prueba, matriz de confusión).
5. Terminal de `validar_vocabulario` (tabla de 20 frases y % de aciertos).
6. Terminal de `pytest -q` con todas las pruebas en verde.
7. Swagger `http://127.0.0.1:8000/docs` con un `/predict` ejecutado.
8. App — Entrenamiento: parámetros configurados y curvas en vivo.
9. App — Métricas: matriz de confusión, ROC y matriz de correlación.
10. App — Simulador: una reseña positiva y una negativa con sus contribuciones.
11. App — Dashboard con filtros aplicados.
12. GitHub: repositorio público e historial de commits progresivos.

---

## 17. Guion de la presentación (15–20 min)

| Min | Diapositiva | Contenido |
|---|---|---|
| 0–1 | Portada | Título, autora, asignatura |
| 1–3 | Motivación | Problema, relevancia y aporte de la IA |
| 3–5 | Estado del arte | 4–5 referencias clave en una línea cada una |
| 5–6 | Alcance | Qué hace y qué no hace |
| 6–8 | Dataset y limpieza | Fuente, embudo de ruido, balance, curva de aprendizaje ("¿hacen falta más datos?") |
| 8–11 | Modelo | TF-IDF, fórmula sigmoide, comparación de modelos, épocas y early stopping |
| 11–13 | Resultados | Matriz de confusión, ROC, correlación, palabras influyentes, errores |
| 13–14 | Arquitectura | Diagrama Streamlit ↔ FastAPI ↔ modelo; estructura del proyecto |
| 14–18 | **Demo en vivo** | Entrenar con parámetros → métricas → simulador con casos del público → dashboard |
| 18–20 | Conclusiones y futuro | 3 conclusiones con números + trabajos futuros |

Preparar respuestas a preguntas típicas: ¿por qué no BERT? ¿cómo evitaste la fuga de datos? ¿por qué quitaste las de 3 estrellas? ¿qué significa la matriz de confusión? ¿por qué esa cantidad de datos? ¿qué hace el α?

---

## 18. Póster (rúbrica: entregables digitales)

Formato A3 o A2 vertical, con la misma paleta que la app:
- Título, autora e institución.
- Problema y objetivo.
- Dataset (cita y embudo de limpieza).
- Metodología (diagrama del pipeline).
- Resultados (tabla de modelos, matriz de confusión, ROC).
- Captura de la app.
- Conclusiones.
- QR al repositorio de GitHub.

---

## 19. Nota sobre "Evidencias del ejercicio en WebGoat"

Las indicaciones piden *"Evidencias del ejercicio en WebGoat (capturas + explicación)"*. WebGoat (OWASP) es una plataforma de prácticas de seguridad web y no guarda relación con este proyecto de Machine Learning, así que probablemente es un residuo de otra plantilla.
- **No asumirlo:** confirmar por escrito con el Ing. Granizo y guardar su respuesta.
- Si confirma que no aplica, la sección de evidencias del informe se llena con las capturas de la sección 16.1.
- Si dice que sí aplica, se agrega un anexo aparte. No forma parte de esta construcción.

---

## 20. Estado del arte — referencias base (APA 7)

1. Keung, P., Lu, Y., Szarvas, G., & Smith, N. A. (2020). The Multilingual Amazon Reviews Corpus. *Proceedings of EMNLP 2020*, 4563–4568. → Fuente del dataset.
2. Pang, B., Lee, L., & Vaithyanathan, S. (2002). Thumbs up? Sentiment classification using machine learning techniques. *Proceedings of EMNLP 2002*, 79–86. → Trabajo fundacional: NB, máxima entropía y SVM para sentimientos.
3. Liu, B. (2012). *Sentiment Analysis and Opinion Mining*. Morgan & Claypool. → Marco teórico del análisis de sentimientos.
4. Joachims, T. (1998). Text categorization with Support Vector Machines: Learning with many relevant features. *ECML-98*, 137–142. → Por qué los modelos lineales funcionan bien en texto disperso.
5. Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval. *Information Processing & Management, 24*(5), 513–523. → Fundamento de TF-IDF.
6. Bottou, L. (2010). Large-scale machine learning with stochastic gradient descent. *Proceedings of COMPSTAT'2010*, 177–186. → Fundamento de SGD y el entrenamiento por épocas.
7. Villena-Román, J., Lana-Serrano, S., Martínez-Cámara, E., & González-Cristóbal, J. C. (2013). TASS — Workshop on Sentiment Analysis at SEPLN. *Procesamiento del Lenguaje Natural, 50*, 37–44. → Análisis de sentimientos específico para el español.
8. Cañete, J., Chaperon, G., Fuentes, R., Ho, J.-H., Kang, H., & Pérez, J. (2020). Spanish pre-trained BERT model and evaluation data. *PML4DC Workshop at ICLR 2020*. → BETO, estado del arte con Transformers en español (trabajo futuro).
9. Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830. → Librería utilizada.
10. Fawcett, T. (2006). An introduction to ROC analysis. *Pattern Recognition Letters, 27*(8), 861–874. → Interpretación de la curva ROC y el AUC.

> Verificar cada referencia (páginas y DOI) en Google Scholar antes de entregar el informe.

---

## 21. Checklist final antes de la defensa

- [ ] Dataset real descargado; enlace de Kaggle, fecha y cita en el README y el informe.
- [ ] Embudo de limpieza con números reales.
- [ ] Experimento de palabras vacías (conservar negaciones) documentado.
- [ ] Curva de aprendizaje con conclusión sobre la cantidad de datos.
- [ ] Comparación de 5 modelos con CV.
- [ ] Partición sin fuga y prueba usada una sola vez.
- [ ] Matriz de confusión, matriz de correlación, ROC, curvas por época y palabras influyentes.
- [ ] Análisis de errores con estimación del ruido de etiqueta.
- [ ] API funcionando (`/docs`) y app con las 5 páginas sin errores.
- [ ] Entrenamiento y parámetros configurables desde la interfaz.
- [ ] `pytest` en verde.
- [ ] Commits progresivos en un GitHub público.
- [ ] Informe PDF, presentación y póster.
- [ ] Respuesta del docente sobre WebGoat guardada.
- [ ] Demo ensayada con la API y la app encendidas **antes** de la exposición (y capturas de respaldo por si falla internet o el equipo).
