import time

import pytest
from fastapi.testclient import TestClient

from src import api
from src.train import Rutas


@pytest.fixture()
def cliente(splits_fixture, tmp_path):
    api.configurar(Rutas(tmp_path / "m", tmp_path / "r", tmp_path / "f"), splits_fixture)
    yield TestClient(api.app)
    api.configurar(Rutas())   # restaura rutas reales


CFG = {"n_por_clase": 100, "epocas": 8, "min_df": 1, "max_features": 2000}


def _esperar(c, tmax=60):
    t0 = time.time()
    while time.time() - t0 < tmax:
        s = c.get("/train/status").json()
        if s["estado"] != "en_curso":
            return s
        time.sleep(0.2)
    raise TimeoutError


def test_health(cliente):
    r = cliente.get("/health")
    assert r.status_code == 200 and r.json()["estado"] == "ok"


def test_predict_sin_modelo_404(cliente):
    assert cliente.post("/predict", json={"texto": "hola"}).status_code == 404


def test_train_parametros_invalidos_400(cliente):
    assert cliente.post("/train", json={**CFG, "epocas": 0}).status_code == 400


def test_config_default(cliente):
    j = cliente.get("/config/default").json()
    assert "epocas" in j["rangos"] and j["defaults"]["algoritmo"] == "sgd"


def test_flujo_completo_y_409(cliente):
    r = cliente.post("/train", json={**CFG, "epocas": 40, "paciencia": 40})
    assert r.status_code == 202
    # mientras corre, otro /train responde 409 (si ya terminó, el flujo igual valida el resto)
    r2 = cliente.post("/train", json=CFG)
    assert r2.status_code in (409, 202)
    s = _esperar(cliente)
    assert s["estado"] == "completado", s
    assert cliente.get("/health").json()["modelo_cargado"] is True
    p = cliente.post("/predict", json={"texto": "excelente producto, funciona perfecto"})
    assert p.status_code == 200 and 0 <= p.json()["prob_positivo"] <= 1
    assert cliente.get("/metrics").status_code == 200
    assert cliente.get("/model/info").status_code == 200
    assert cliente.get("/metrics/detalle").status_code == 200


def test_validacion_predict(cliente):
    cliente.post("/train", json=CFG)
    _esperar(cliente)
    assert cliente.post("/predict", json={"texto": ""}).status_code == 422
    assert cliente.post("/predict", json={"texto": "x" * 5001}).status_code == 422
    assert cliente.post("/predict", json={"texto": "bueno", "umbral": 0.99}).status_code == 422
    assert cliente.post("/predict", json={"texto": "12345 !!!"}).status_code == 400
    b = cliente.post("/predict/batch", json={"textos": ["muy bueno", "pésimo, no sirve"]})
    assert b.status_code == 200 and len(b.json()["resultados"]) == 2
    assert cliente.post("/predict/batch", json={"textos": ["a"] * 501}).status_code == 422


def test_dataset(cliente):
    s = cliente.get("/dataset/summary").json()
    assert s["total"] > 0
    m = cliente.get("/dataset/sample", params={"estrellas": 1, "n": 5}).json()
    assert all(f["estrellas"] == 1 for f in m["filas"])


def test_train_409_si_ya_hay_uno_en_curso(cliente):
    with api.ESTADO.lock:
        api.ESTADO.job = {"job_id": "x", "estado": "en_curso"}
    assert cliente.post("/train", json=CFG).status_code == 409
