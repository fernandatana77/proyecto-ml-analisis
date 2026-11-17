"""Humo de la interfaz: ejecuta cada página con AppTest. Requiere la API encendida en :8000 (se omite si no)."""
from pathlib import Path

import pytest
import requests

APP = str(Path(__file__).resolve().parent.parent / "src" / "app.py")

try:
    requests.get("http://127.0.0.1:8000/health", timeout=2)
    API_ON = True
except requests.exceptions.RequestException:
    API_ON = False

pytestmark = pytest.mark.skipif(not API_ON, reason="API apagada")

PAGINAS = ["Inicio", "Entrenamiento", "Métricas", "Simulador", "Dashboard"]


@pytest.mark.parametrize("pagina", PAGINAS)
def test_pagina_sin_excepciones(pagina):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    at.sidebar.radio[0].set_value(pagina).run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]


def test_simulador_analiza():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    at.sidebar.radio[0].set_value("Simulador").run()
    at.text_area[0].set_value("No funciona nada, una estafa").run()
    [b for b in at.button if b.label == "Analizar"][0].click().run()
    assert not at.exception
    assert any("Negativo" in m.value for m in at.markdown)
