"""AegisOT AI API (shared by the ML and Agents sub-teams).

Run:   uvicorn src.api.main:app --reload --port 8001      (or: python -m src.api.main)
Docs:  http://localhost:8001/docs
"""
from __future__ import annotations

from fastapi import FastAPI

from src.agents.agents import Runner
from src.agents.clients import Clients
from src.agents.router import build_agent_router
from src.agents.settings import Settings, get_settings


def create_app(settings: Settings | None = None, clients: Clients | None = None,
               runner: Runner | None = None) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(title="AegisOT AI API", version="0.1.0")

    @app.get("/health")
    def health():
        return {"status": "ok", "service": "aegisot-ai", "llm_enabled": settings.llm_enabled,
                "integrations": {"backend": settings.backend_mode, "anomaly": settings.anomaly_mode,
                                 "network": settings.network_mode}}

    # Agents: /api/v1/incident/analyze, /api/v1/diagnosis/analyze, /api/v1/agents/*
    app.include_router(build_agent_router(settings, clients, runner))

    # ML team hook: create `router = APIRouter()` in src/ml/router.py (e.g. POST /api/v1/anomaly/predict);
    # it is included automatically. A missing module is ignored; any other import error is raised.
    try:
        from src.ml.router import router as ml_router
    except ModuleNotFoundError as e:
        if e.name not in ("src.ml", "src.ml.router"):
            raise
    else:
        app.include_router(ml_router)
    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    s = get_settings()
    uvicorn.run("src.api.main:app", host=s.api_host, port=s.api_port, reload=False)
