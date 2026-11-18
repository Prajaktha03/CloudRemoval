from io import BytesIO
from dataclasses import replace
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from backend.main import app
from backend.core.config import settings
from backend.services.evaluation import evaluate
from backend.services.experiments import ExperimentStore
from backend.services.imaging import normalize_image, read_image
from backend.services.model import ModelService
from models.generator import MultimodalUNet
from backend.core.errors import ModelUnavailableError
from backend.services.auth import AuthStore
from utils.config import ConfigDict


def test_authentication_protects_api_and_issues_revocable_cookie(tmp_path: Path):
    with TestClient(app) as client:
        app.state.auth_store = AuthStore(tmp_path / "auth.sqlite3")
        app.state.auth_store.create_user("researcher@example.org", "correct horse battery 42")

        assert client.get("/analytics/model-comparison").status_code == 401
        blocked = client.post(
            "/auth/login",
            json={"email": "researcher@example.org", "password": "correct horse battery 42"},
            headers={"Origin": "https://untrusted.example"},
        )
        assert blocked.status_code == 403

        login = client.post(
            "/auth/login",
            json={"email": "researcher@example.org", "password": "correct horse battery 42"},
            headers={"Origin": "http://localhost:3000"},
        )
        assert login.status_code == 200
        cookie = login.headers["set-cookie"].lower()
        assert "httponly" in cookie
        assert "samesite=strict" in cookie
        assert client.get("/auth/me").json()["user"]["email"] == "researcher@example.org"
        denied_write = client.post("/train", headers={"Origin": "https://untrusted.example"})
        assert denied_write.status_code == 403

        logout = client.post("/auth/logout", headers={"Origin": "http://localhost:3000"})
        assert logout.status_code == 200
        assert client.get("/auth/me").status_code == 401


@pytest.fixture
def authenticated_client(tmp_path: Path):
    with TestClient(app) as client:
        app.state.auth_store = AuthStore(tmp_path / "auth.sqlite3")
        app.state.auth_store.create_user("researcher@example.org", "correct horse battery 42")
        client.headers.update({"Origin": "http://localhost:3000"})
        response = client.post(
            "/auth/login",
            json={"email": "researcher@example.org", "password": "correct horse battery 42"},
        )
        assert response.status_code == 200
        yield client


def png_bytes(array: np.ndarray) -> bytes:
    buffer = BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


def test_health_and_dataset_info(authenticated_client):
    assert authenticated_client.get("/health").status_code == 200
    assert authenticated_client.get("/dataset/info").status_code == 200


def test_deployment_health_is_public():
    with TestClient(app) as client:
        response = client.get("/healthz")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


def test_saved_evaluation_and_training_reports_are_available(authenticated_client):
    comparison = authenticated_client.get("/analytics/model-comparison")
    history = authenticated_client.get("/analytics/training-history")

    assert comparison.status_code == 200
    assert comparison.json()["split"] == "test"
    assert comparison.json()["total_samples"] == 25
    assert history.status_code == 200
    assert history.json()["histories"]["multimodal_unet"]["epoch"]


def test_test_scene_preview_is_from_the_held_out_manifest(authenticated_client):
    metadata = authenticated_client.get("/dataset/sample")
    preview = authenticated_client.get("/dataset/sample/cloudy_optical")

    assert metadata.status_code == 200
    assert metadata.json()["scene_id"].startswith("ROI_2017_140")
    assert metadata.json()["cloudy_optical"]["bands"] == 13
    assert preview.status_code == 200
    assert preview.headers["content-type"] == "image/png"


def test_predict_reports_missing_model_and_validates_uploads(authenticated_client):
    optical = np.zeros((32, 32, 3), dtype=np.uint8)
    sar = np.zeros((32, 32, 3), dtype=np.uint8)
    response = authenticated_client.post(
        "/predict",
        files={
            "optical": ("optical.png", png_bytes(optical), "image/png"),
            "sar": ("sar.png", png_bytes(sar), "image/png"),
        },
    )
    assert response.status_code == 422


def test_evaluation_identical_images_and_shape_validation(authenticated_client):
    reference = np.full((16, 16, 3), 0.5, dtype=np.float32)
    metrics = evaluate(reference, reference.copy())
    assert metrics["psnr"] is None
    assert metrics["psnr_is_infinite"] is True
    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0
    assert abs(metrics["ssim"] - 1.0) < 1e-6
    single_band = np.full((16, 16, 1), 0.5, dtype=np.float32)
    assert evaluate(single_band, single_band.copy())["ssim"] == 1.0
    response = authenticated_client.post(
        "/evaluate",
        files={
            "reconstruction": ("recon.png", png_bytes(np.zeros((16, 16, 3), dtype=np.uint8)), "image/png"),
            "reference": ("ref.png", png_bytes(np.zeros((15, 16, 3), dtype=np.uint8)), "image/png"),
        },
    )
    assert response.status_code == 422


def test_evaluate_endpoint_returns_json_for_exact_match(authenticated_client):
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    response = authenticated_client.post(
        "/evaluate",
        files={
            "reconstruction": ("reconstruction.png", png_bytes(image), "image/png"),
            "reference": ("reference.png", png_bytes(image), "image/png"),
        },
    )
    assert response.status_code == 200
    assert response.json()["metrics"]["psnr"] is None
    assert response.json()["metrics"]["psnr_is_infinite"] is True
    assert response.json()["metrics"]["mae"] == 0.0
    assert response.json()["metrics"]["rmse"] == 0.0


def test_experiment_store_persists_only_summaries(tmp_path: Path):
    store = ExperimentStore(tmp_path / "experiments.sqlite3")
    store.add("evaluation", {"psnr": 30.0})
    assert store.list()[0]["metrics"] == {"psnr": 30.0}


def test_integer_png_normalizes_to_unit_range():
    image = read_image(png_bytes(np.full((16, 16, 3), 255, dtype=np.uint8)), "image.png")
    normalized = normalize_image(image, 3, "image")
    assert np.all(normalized == 1.0)


def test_decoded_image_limit_and_geospatial_alignment():
    image_bytes = png_bytes(np.zeros((16, 16, 3), dtype=np.uint8))
    try:
        read_image(image_bytes, "large.png", max_decoded_bytes=10)
    except ValueError as exc:
        assert "processing limit" in str(exc)
    else:
        raise AssertionError("oversized decoded image should be rejected")


def test_compatible_checkpoint_loads_and_runs_inference(tmp_path: Path):
    config = replace(
        settings,
        checkpoint_path=tmp_path / "generator.pt",
        optical_channels=3,
        sar_channels=2,
        model_base_channels=4,
        model_uses_mask=True,
    )
    model = MultimodalUNet(3, 2, use_mask=True, base_channels=4)
    torch.save({"generator": model.state_dict()}, config.checkpoint_path)

    service = ModelService(config)
    service.load()
    prediction = service.predict(
        np.zeros((32, 32, 3), dtype=np.float32),
        np.zeros((32, 32, 2), dtype=np.float32),
        np.zeros((32, 32, 1), dtype=np.float32),
    )

    assert service.status()["available"] is True
    assert prediction.shape == (32, 32, 3)
    assert np.isfinite(prediction).all()


def test_model_service_loads_training_checkpoint_configdict_safely(tmp_path: Path):
    config = replace(
        settings,
        checkpoint_path=tmp_path / "generator.pt",
        optical_channels=3,
        sar_channels=2,
        model_base_channels=4,
        model_uses_mask=True,
    )
    model = MultimodalUNet(3, 2, use_mask=True, base_channels=4)
    torch.save(
        {
            "generator": model.state_dict(),
            "config": ConfigDict({"model": ConfigDict({"base_channels": 4})}),
        },
        config.checkpoint_path,
    )

    service = ModelService(config)
    service.load()

    assert service.status()["available"] is True
    prediction = service.predict(
        np.zeros((32, 32, 3), dtype=np.float32),
        np.zeros((32, 32, 2), dtype=np.float32),
        np.zeros((32, 32, 1), dtype=np.float32),
    )
    assert prediction.shape == (32, 32, 3)


def test_predict_endpoint_returns_reconstruction_bundle(tmp_path: Path, authenticated_client):
    config = replace(
        settings,
        checkpoint_path=tmp_path / "generator.pt",
        optical_channels=3,
        sar_channels=3,
        model_base_channels=4,
        model_uses_mask=True,
    )
    model = MultimodalUNet(3, 3, use_mask=True, base_channels=4)
    torch.save(model.state_dict(), config.checkpoint_path)
    service = ModelService(config)
    service.load()
    optical = np.zeros((32, 32, 3), dtype=np.uint8)
    sar = np.zeros((32, 32, 3), dtype=np.uint8)

    app.state.settings = config
    app.state.model_service = service
    response = authenticated_client.post(
        "/predict",
        files={
            "optical": ("optical.png", png_bytes(optical), "image/png"),
            "sar": ("sar.png", png_bytes(sar), "image/png"),
        },
    )

    assert response.status_code == 200
    with zipfile.ZipFile(BytesIO(response.content)) as archive:
        assert set(archive.namelist()) == {"reconstruction.png", "metadata.json"}
        metadata = json.loads(archive.read("metadata.json"))
    assert metadata["height"] == 32
    assert metadata["channels"] == 3


def test_model_service_reports_missing_checkpoint(tmp_path: Path):
    config = replace(settings, checkpoint_path=tmp_path / "missing.pt")
    service = ModelService(config)
    service.load()
    assert service.status()["available"] is False
    try:
        service.predict(np.zeros((16, 16, 13)), np.zeros((16, 16, 2)), None)
    except ModelUnavailableError as exc:
        assert "Checkpoint not found" in str(exc)
    else:
        raise AssertionError("inference without a checkpoint must fail")
