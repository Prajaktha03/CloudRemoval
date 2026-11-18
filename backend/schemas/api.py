from typing import Any

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    model: dict[str, Any]


class DatasetInfoResponse(BaseModel):
    configured: bool
    path: str
    file_count: int
    files_by_extension: dict[str, int]
    note: str


class ExperimentSummary(BaseModel):
    id: int
    kind: str
    created_at: str
    metrics: dict[str, Any]


class TrainingRequest(BaseModel):
    config_path: str | None = None


class TrainingStatus(BaseModel):
    job_id: str
    status: str
    started_at: str
    finished_at: str | None = None
    return_code: int | None = None
    message: str | None = None
