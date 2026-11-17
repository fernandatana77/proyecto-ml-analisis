import pytest

from src.config import TrainConfig
from src.train import Rutas, crear_vectorizador, entrenar


@pytest.fixture(scope="module")
def resultado(splits_fixture, tmp_path_factory):
    base = tmp_path_factory.mktemp("salida")
    rutas = Rutas(base / "models", base / "reports", base / "figs")
    cfg = TrainConfig(n_por_clase=100, epocas=15, min_df=1, max_features=2000, paciencia=5, comparar_modelos=True)
    m = entrenar(cfg, datos=splits_fixture, rutas=rutas)
    return m, rutas, cfg


def test_corre_y_mejor_epoca_en_rango(resultado):
    m, _, cfg = resultado
    assert 1 <= m["mejor_epoca"] <= cfg.epocas


def test_artefactos_y_figuras(resultado):
    m, rutas, _ = resultado
    for f in ("modelo.joblib", "vectorizador.joblib"):
        assert (rutas.modelos / f).exists()
    for f in ("metricas.json", "historial_epocas.csv", "errores.csv"):
        assert (rutas.reportes / f).exists()
    for f in ("1_evolucion_epocas.png", "2_matriz_confusion.png", "3_curva_roc.png",
              "4_palabras_influyentes.png", "6_comparacion_modelos.png"):
        assert (rutas.figuras / f).exists(), f


def test_early_stopping_detiene_antes(splits_fixture, tmp_path):
    cfg = TrainConfig(n_por_clase=100, epocas=100, min_df=1, paciencia=1, max_features=2000)
    m = entrenar(cfg, datos=splits_fixture, rutas=Rutas(tmp_path / "m", tmp_path / "r", tmp_path / "f"),
                 guardar_figuras=False)
    assert m["parada_temprana"] and m["epocas_ejecutadas"] < 100


def test_tfidf_ajustado_solo_con_train(splits_fixture):
    tr, va, _ = splits_fixture
    cfg = TrainConfig(min_df=1, max_features=5000)
    vec = crear_vectorizador(cfg).fit(tr["texto_limpio"])
    terminos_train = set()
    for t in tr["texto_limpio"]:
        w = t.split()
        for n in range(1, cfg.ngram_max + 1):
            terminos_train.update(" ".join(w[i:i + n]) for i in range(len(w) - n + 1))
    assert set(vec.vocabulary_) <= terminos_train


def test_config_rechaza_rangos():
    for malo in ({"epocas": 0}, {"umbral": 0.99}, {"alpha": -1}, {"algoritmo": "xyz"}, {"ngram_max": 5}):
        with pytest.raises(ValueError):
            TrainConfig(**malo)


@pytest.mark.parametrize("algo", ["logreg", "svm", "nb"])
def test_otros_algoritmos(algo, splits_fixture, tmp_path):
    cfg = TrainConfig(n_por_clase=100, algoritmo=algo, min_df=1, max_features=2000)
    m = entrenar(cfg, datos=splits_fixture, rutas=Rutas(tmp_path / "m", tmp_path / "r", tmp_path / "f"),
                 guardar_figuras=False)
    assert 0 <= m["prueba"]["accuracy"] <= 1
