import pandas as pd
import pytest

from src.preprocessing import balancear, construir_dataset, etiquetar, limpiar_texto


def test_conserva_tildes_enie_y_negaciones():
    t = limpiar_texto("¡No funciona NUNCA! Sin garantía, pingüino año")
    for w in ("no", "nunca", "sin", "garantía", "pingüino", "año"):
        assert w in t.split()


def test_quita_url_html_numeros():
    t = limpiar_texto("Mira http://x.com/a <br> compré 25 unidades www.foo.es")
    assert "http" not in t and "<" not in t and "br" not in t.split() and "25" not in t and "www" not in t


def test_letras_repetidas():
    assert "bueno" in limpiar_texto("buenooooo").split()


def test_stopwords_estandar_destruye_negacion():
    assert "no" not in limpiar_texto("no funciona", stopwords="estandar").split()
    assert "no" in limpiar_texto("no funciona", stopwords="propia").split()


def test_etiquetas():
    assert [etiquetar(e) for e in (1, 2, 3, 4, 5)] == [0, 0, None, 1, 1]


def test_balanceo_50_50():
    df = pd.DataFrame({"sentimiento": [0] * 30 + [1] * 10, "x": range(40)})
    b = balancear(df)
    assert b["sentimiento"].value_counts().tolist() == [10, 10]


def test_loader_falla_sin_columnas(tmp_path):
    from src.data_loader import cargar_crudo
    pd.DataFrame({"a": [1]}).to_csv(tmp_path / "x.csv", index=False)
    with pytest.raises(ValueError, match="faltan columnas"):
        cargar_crudo(tmp_path)


def test_train_balanceado_en_dataset_real():
    r = construir_dataset(guardar=False)
    assert r["splits"]["train"]["negativo"] == r["splits"]["train"]["positivo"]


def test_splits_guardados_sin_solape():
    from src.config import DATA_PROC
    if not (DATA_PROC / "train.csv").exists():
        pytest.skip("ejecuta python -m src.preprocessing")
    tr, va, te = (pd.read_csv(DATA_PROC / f"{n}.csv") for n in ("train", "val", "test"))
    assert not set(tr["texto_limpio"]) & set(te["texto_limpio"])
    assert not set(tr["texto_limpio"]) & set(va["texto_limpio"])
