# Comandos rápidos: .\run.ps1 api | app | train | test
param([string]$cmd = "test")
switch ($cmd) {
  "api"   { python -m uvicorn src.api:app --port 8000 }
  "app"   { python -m streamlit run src/app.py --server.port 8501 }
  "train" { python -m src.train }
  "test"  { python -m pytest -q }
}
