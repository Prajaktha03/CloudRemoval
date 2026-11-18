from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.core.config import settings
from backend.routes.auth import router as auth_router
from backend.routes.api import router
from backend.services.auth import AuthStore
from backend.services.experiments import ExperimentStore
from backend.services.model import ModelService
from backend.services.training import TrainingJobManager


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.settings = settings
    app.state.auth_store = AuthStore(settings.database_path, settings.session_max_age_seconds)
    app.state.model_service = ModelService(settings)
    app.state.model_service.load()
    app.state.experiment_store = ExperimentStore(settings.database_path)
    app.state.training_manager = TrainingJobManager(settings.training_command)
    yield


app = FastAPI(
    title="Satellite Cloud Removal API",
    description="Multimodal cloudy-optical and SAR reconstruction with optional cloud masks.",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.frontend_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/healthz", include_in_schema=False)
async def deployment_health():
    return {"status": "ok"}


app.include_router(auth_router)
app.include_router(router)
