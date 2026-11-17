"""Fixtures de prueba. DATOS SINTÉTICOS DE PRUEBA DE SOFTWARE: nunca se usan para reportar métricas."""
import random

import pandas as pd
import pytest

POS = ["excelente producto, funciona perfecto", "muy bueno, llegó rápido y encantado",
       "me encantó, buena calidad y precio", "recomendado, superó mis expectativas",
       "genial, lo volvería a comprar", "perfecto para lo que necesitaba"]
NEG = ["pésimo producto, no funciona", "muy malo, llegó roto y defectuoso",
       "no me gustó, mala calidad", "jamás volvería a comprar, una estafa",
       "horrible, no sirve para nada", "decepcionante, se rompió al primer uso"]


@pytest.fixture(scope="session")
def df_fixture() -> pd.DataFrame:
    """~240 filas sintéticas (SOLO para pruebas de software)."""
    from src.preprocessing import limpiar_texto
    rnd = random.Random(0)
    filas = []
    for i in range(120):
        filas.append({"texto": f"{rnd.choice(POS)} número {i}", "estrellas": rnd.choice([4, 5]),
                      "sentimiento": 1, "categoria": rnd.choice(["home", "toys"])})
        filas.append({"texto": f"{rnd.choice(NEG)} número {i}", "estrellas": rnd.choice([1, 2]),
                      "sentimiento": 0, "categoria": rnd.choice(["home", "toys"])})
    df = pd.DataFrame(filas)
    df["texto_limpio"] = df["texto"].map(limpiar_texto)
    return df


@pytest.fixture(scope="session")
def splits_fixture(df_fixture):
    from sklearn.model_selection import train_test_split
    tr, rest = train_test_split(df_fixture, test_size=0.4, stratify=df_fixture["sentimiento"], random_state=42)
    va, te = train_test_split(rest, test_size=0.5, stratify=rest["sentimiento"], random_state=42)
    return tr.reset_index(drop=True), va.reset_index(drop=True), te.reset_index(drop=True)
