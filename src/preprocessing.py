"""Limpieza, etiquetado, variables derivadas, balanceo y particiones (secciones 4-5)."""
from __future__ import annotations

import json
import re
import unicodedata

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import DATA_PROC, REPORTS, SEED, asegurar_carpetas
from .data_loader import cargar_crudo

NEGACIONES = {"no", "ni", "nunca", "jamás", "jamas", "tampoco", "nada", "nadie",
              "ningún", "ningun", "ninguno", "ninguna", "sin"}
CONTRASTES = {"pero", "aunque", "sino"}
INTENSIFICADORES = {"muy", "mucho", "mucha", "muchos", "muchas", "poco", "poca", "pocos",
                    "pocas", "más", "mas", "menos", "demasiado", "bastante"}
CONSERVAR = NEGACIONES | CONTRASTES | INTENSIFICADORES

# Lista estándar tipo NLTK (incluye negaciones e intensificadores, a propósito)
STOP_ESTANDAR = set(["de", "la", "que", "el", "en", "y", "a", "los", "del", "se", "las", "por", "un", "para", "con", "no", "una", "su", "al", "lo", "como", "más", "pero", "sus", "le", "ya", "o", "este", "sí", "porque", "esta", "entre", "cuando", "muy", "sin", "sobre", "también", "me", "hasta", "hay", "donde", "quien", "desde", "todo", "nos", "durante", "todos", "uno", "les", "ni", "contra", "otros", "ese", "eso", "ante", "ellos", "e", "esto", "mí", "antes", "algunos", "qué", "unos", "yo", "otro", "otras", "otra", "él", "tanto", "esa", "estos", "mucho", "quienes", "nada", "muchos", "cual", "poco", "ella", "estar", "estas", "algunas", "algo", "nosotros", "mi", "mis", "tú", "te", "ti", "tu", "tus", "ellas", "nosotras", "vosotros", "vosotras", "os", "mío", "mía", "míos", "mías", "tuyo", "tuya", "tuyos", "tuyas", "suyo", "suya", "suyos", "suyas", "nuestro", "nuestra", "nuestros", "nuestras", "vuestro", "vuestra", "vuestros", "vuestras", "esos", "esas", "estoy", "estás", "está", "estamos", "estáis", "están", "esté", "estés", "estemos", "estéis", "estén", "he", "has", "ha", "hemos", "habéis", "han", "soy", "eres", "es", "somos", "sois", "son", "fue", "era", "eran", "ser", "ha", "sido", "tengo", "tienes", "tiene", "tenemos", "tienen"])
# Lista propia: la estándar SIN negaciones, contrastes ni intensificadores
STOP_PROPIA = STOP_ESTANDAR - CONSERVAR

RE_URL = re.compile(r"https?://\S+|www\.\S+")
RE_HTML = re.compile(r"<[^>]+>")
RE_REPETIDAS = re.compile(r"(.)\1{2,}")
RE_NO_LETRA = re.compile(r"[^a-záéíóúüñ\s]")
RE_ESPACIOS = re.compile(r"\s+")
RE_EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿]")
RE_MOJI = re.compile("Ã.|Â.|�")


def reparar_mojibake(t: str) -> str:
    try:
        import ftfy
        return ftfy.fix_text(t)
    except ImportError:
        return t


def limpiar_texto(texto: str, stopwords: str = "propia") -> str:
    """Normaliza el texto. `stopwords`: 'propia' (conserva negaciones), 'estandar' o 'ninguna'."""
    if not isinstance(texto, str):
        return ""
    t = unicodedata.normalize("NFC", texto).lower()
    t = RE_URL.sub(" ", t)
    t = RE_HTML.sub(" ", t)
    t = RE_REPETIDAS.sub(r"\1", t)
    t = RE_NO_LETRA.sub(" ", t)
    t = RE_ESPACIOS.sub(" ", t).strip()
    if stopwords != "ninguna":
        lista = STOP_PROPIA if stopwords == "propia" else STOP_ESTANDAR
        t = " ".join(w for w in t.split() if w not in lista)
    return t


def variables_derivadas(texto: str) -> dict:
    """Variables calculadas sobre el texto ORIGINAL (antes de limpiar)."""
    letras = [c for c in texto if c.isalpha()]
    toks = re.findall(r"\w+", texto.lower())
    return {
        "longitud_caracteres": len(texto),
        "num_palabras": len(texto.split()),
        "num_exclamaciones": texto.count("!") + texto.count("¡"),
        "num_interrogaciones": texto.count("?") + texto.count("¿"),
        "proporcion_mayusculas": sum(c.isupper() for c in letras) / len(letras) if letras else 0.0,
        "num_negaciones": sum(t in NEGACIONES for t in toks),
    }


def etiquetar(estrellas: int) -> int | None:
    """1-2 -> 0 (Negativo), 4-5 -> 1 (Positivo), 3 -> None (se descarta)."""
    if estrellas in (1, 2):
        return 0
    if estrellas in (4, 5):
        return 1
    return None


def balancear(df: pd.DataFrame, col: str = "sentimiento", n_por_clase: int | None = None,
              seed: int = SEED) -> pd.DataFrame:
    """Submuestrea cada clase al mismo tamaño (el mínimo, o `n_por_clase` si es menor)."""
    n = int(df[col].value_counts().min())
    if n_por_clase:
        n = min(n, n_por_clase)
    partes = [g.sample(n=n, random_state=seed) for _, g in df.groupby(col)]
    return pd.concat(partes).sample(frac=1, random_state=seed).reset_index(drop=True)


def _detectar_idioma_ajeno(textos: pd.Series, umbral: float = 0.99) -> pd.Series:
    """True si langdetect detecta otro idioma con prob >= umbral (solo textos de >= 30 caracteres)."""
    try:
        from langdetect import DetectorFactory, detect_langs
    except ImportError:
        return pd.Series(False, index=textos.index)
    DetectorFactory.seed = SEED
    out = []
    for t in textos:
        if len(t) < 30:
            out.append(False)
            continue
        try:
            top = detect_langs(t)[0]
            out.append(top.lang != "es" and top.prob >= umbral)
        except Exception:
            out.append(False)
    return pd.Series(out, index=textos.index)


def construir_dataset(guardar: bool = True) -> dict:
    asegurar_carpetas()
    df = cargar_crudo()
    embudo = [("registros_crudos", len(df))]
    df["texto"] = df["cuerpo"].fillna("").astype(str)

    def paso(nombre: str) -> None:
        embudo.append((nombre, len(df)))

    # 1. nulos/vacíos
    df = df[df["texto"].str.strip() != ""].copy()
    paso("sin_nulos_ni_vacios")
    # 2. neutrales
    df["sentimiento"] = df["estrellas"].map(etiquetar)
    df = df[df["sentimiento"].notna()].copy()
    df["sentimiento"] = df["sentimiento"].astype(int)
    paso("sin_neutrales_3_estrellas")
    # 3. mojibake: reparar; contar afectadas
    moji = int(df["texto"].str.contains(RE_MOJI).sum())
    df["texto"] = df["texto"].map(reparar_mojibake)
    # 4. duplicados (texto normalizado)
    df["_norm"] = df["texto"].str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    df = df.drop_duplicates("_norm").copy()
    paso("sin_duplicados")
    # 5. idioma ajeno
    ajeno = _detectar_idioma_ajeno(df["texto"])
    df = df[~ajeno].copy()
    paso("solo_espanol_detectado")
    # variables derivadas (sobre el original) y texto limpio
    der = pd.DataFrame([variables_derivadas(t) for t in df["texto"]], index=df.index)
    df = pd.concat([df, der], axis=1)
    df["texto_limpio"] = df["texto"].map(limpiar_texto)
    # 6. muy cortas tras limpiar
    df = df[df["texto_limpio"].str.split().str.len() >= 2].copy()
    paso("con_al_menos_2_tokens_utiles")

    afect = {
        "mojibake_detectado": moji,
        "con_html": int(df["texto"].str.contains(RE_HTML).sum()),
        "con_url": int(df["texto"].str.contains(RE_URL).sum()),
        "con_letras_repetidas": int(df["texto"].map(lambda t: bool(RE_REPETIDAS.search(t))).sum()),
        "con_emoji": int(df["texto"].str.contains(RE_EMOJI).sum()),
    }
    if "sentimiento_original" in df:
        afect["discrepancias_con_columna_sentiment_original"] = int(
            (df["sentimiento"] != df["sentimiento_original"]).sum())

    cols = ["texto", "texto_limpio", "estrellas", "sentimiento", "categoria", "split_origen",
            "longitud_caracteres", "num_palabras", "num_exclamaciones", "num_interrogaciones",
            "proporcion_mayusculas", "num_negaciones", "_norm"]
    df = df[cols].reset_index(drop=True)

    # Particiones 70/15/15 estratificadas (opción B)
    train, resto = train_test_split(df, test_size=0.30, stratify=df["sentimiento"], random_state=SEED)
    val, test = train_test_split(resto, test_size=0.50, stratify=resto["sentimiento"], random_state=SEED)
    # sin fuga: quitar de train textos presentes en val/test
    prohibidos = set(val["texto_limpio"]) | set(test["texto_limpio"]) | set(val["_norm"]) | set(test["_norm"])
    antes = len(train)
    train = train[~(train["texto_limpio"].isin(prohibidos) | train["_norm"].isin(prohibidos))]
    duplicados_entre_splits = antes - len(train)
    # balanceo solo en entrenamiento
    antes_bal = len(train)
    train = balancear(train)
    embudo.append(("train_tras_balanceo", len(train)))
    train, val, test = (x.drop(columns="_norm").reset_index(drop=True) for x in (train, val, test))

    resumen = {
        "embudo_limpieza": [{"paso": n, "filas": int(f)} for n, f in embudo],
        "ruido_detectado": afect,
        "duplicados_entre_splits_eliminados_de_train": int(duplicados_entre_splits),
        "train_descartadas_por_balanceo": int(antes_bal - len(train)),
        "splits": {nombre: {"filas": len(x),
                            "negativo": int((x["sentimiento"] == 0).sum()),
                            "positivo": int((x["sentimiento"] == 1).sum())}
                   for nombre, x in (("train", train), ("val", val), ("test", test))},
        "particion": "Opción B: 70/15/15 estratificada, random_state=42 (dataset de un solo archivo)",
    }
    if guardar:
        train.to_csv(DATA_PROC / "train.csv", index=False)
        val.to_csv(DATA_PROC / "val.csv", index=False)
        test.to_csv(DATA_PROC / "test.csv", index=False)
        ruta = REPORTS / "auditoria_datos.json"
        base = json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else {}
        base.update(resumen)
        ruta.write_text(json.dumps(base, indent=2, ensure_ascii=False), encoding="utf-8")
    return resumen


def main() -> None:
    r = construir_dataset()
    print("EMBUDO DE LIMPIEZA")
    prev = None
    for e in r["embudo_limpieza"]:
        d = "" if prev is None else f"  ({e['filas'] - prev:+d})"
        print(f"  {e['paso']:<32}{e['filas']:>7}{d}")
        prev = e["filas"]
    print("\nRuido detectado:", json.dumps(r["ruido_detectado"], ensure_ascii=False))
    print("Duplicados entre splits eliminados de train:", r["duplicados_entre_splits_eliminados_de_train"])
    print("Descartadas por balanceo:", r["train_descartadas_por_balanceo"])
    print("\nSPLITS")
    for k, v in r["splits"].items():
        print(f"  {k:<6} {v}")


if __name__ == "__main__":
    main()
