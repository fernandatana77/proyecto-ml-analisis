import pytest

from src.config import TrainConfig
from src.predictor import ModeloNoDisponible, Predictor
from src.train import Rutas, entrenar


@pytest.fixture(scope="module")
def pred(splits_fixture, tmp_path_factory):
    b = tmp_path_factory.mktemp("p")
    r = Rutas(b / "m", b / "r", b / "f")
    entrenar(TrainConfig(n_por_clase=100, epocas=10, min_df=1, max_features=2000), datos=splits_fixture,
             rutas=r, guardar_figuras=False)
    return Predictor(r.modelos)


def test_probabilidad_en_rango(pred):
    r = pred.predecir("excelente producto, funciona perfecto")
    assert 0 <= r["prob_positivo"] <= 1 and 0 <= r["confianza"] <= 1


def test_etiqueta_coherente_con_umbral(pred):
    t = "excelente producto, funciona perfecto"
    p = pred.predecir(t)["prob_positivo"]
    assert pred.predecir(t, umbral=min(p + 0.01, 0.95))["etiqueta"] == ("Negativo" if p + 0.01 <= 0.95 else "Positivo")
    assert pred.predecir(t, umbral=max(p - 0.01, 0.05))["etiqueta"] == "Positivo"


def test_contribuciones_ordenadas(pred):
    c = [abs(x["contribucion"]) for x in pred.predecir("pésimo producto, no funciona nada")["top_terminos"]]
    assert c == sorted(c, reverse=True) and len(c) > 0


@pytest.mark.parametrize("malo", ["", "   ", "123 456 !!!"])
def test_texto_invalido(pred, malo):
    with pytest.raises(ValueError):
        pred.predecir(malo)


def test_sin_modelo(tmp_path):
    with pytest.raises(ModeloNoDisponible):
        Predictor(tmp_path)
