from io import BytesIO
import csv
import json
from pathlib import Path
import zipfile
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.responses import Response
import numpy as np
from PIL import Image
from starlette.concurrency import run_in_threadpool

from backend.core.errors import InputValidationError, ModelUnavailableError
from backend.schemas.api import DatasetInfoResponse, HealthResponse, TrainingRequest, TrainingStatus
from backend.routes.auth import require_authenticated_user, require_trusted_origin
from backend.services.dataset import dataset_info
from backend.services.evaluation import evaluate
from backend.services.imaging import (
    encode_prediction,
    normalize_image,
    read_image,
    validate_aligned,
    validate_spatial_alignment,
)
from datasets.preprocessing import (
    make_rgb_preview,
    make_sar_preview,
    normalize_optical,
    normalize_sar,
)

router = APIRouter(
    dependencies=[
        Depends(require_authenticated_user),
        Depends(require_trusted_origin, use_cache=True),
    ]
)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_ROOT = PROJECT_ROOT / "outputs" / "reports"


def _test_scene_paths() -> dict[str, str]:
    manifest = PROJECT_ROOT / "data" / "manifests" / "manifest_test.csv"
    if not manifest.is_file():
        raise HTTPException(status_code=404, detail="The held-out test manifest is unavailable")
    with manifest.open("r", encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            paths = {}
            for name in ("cloudy_s2_path", "sar_s1_path", "clear_s2_path"):
                candidate = (PROJECT_ROOT / row[name].replace("\\", "/")).resolve()
                try:
                    candidate.relative_to(PROJECT_ROOT)
                except ValueError:
                    continue
                if not candidate.is_file():
                    break
                paths[name] = candidate
            if len(paths) == 3:
                return {"row": row, **paths}
    raise HTTPException(status_code=404, detail="No complete held-out scene is available")


def _read_test_scene():
    try:
        import rasterio
    except ImportError as exc:
        raise HTTPException(status_code=503, detail="Scene previews require Rasterio") from exc

    scene = _test_scene_paths()
    bands = {}
    metadata = {}
    for name, label in (
        ("cloudy_s2_path", "cloudy_optical"),
        ("sar_s1_path", "sar"),
        ("clear_s2_path", "reference"),
    ):
        with rasterio.open(scene[name]) as source:
            raw = np.moveaxis(source.read(), 0, -1)
            profile = source.profile
            metadata[label] = {
                "bands": source.count,
                "width": source.width,
                "height": source.height,
                "crs": str(source.crs) if source.crs else None,
                "bounds": list(source.bounds),
            }
            if label in {"cloudy_optical", "reference"}:
                normalized = normalize_optical(raw, nodata_value=source.nodata)
                bands[label] = make_rgb_preview(normalized)
            else:
                bands[label] = make_sar_preview(normalize_sar(raw))

    optical = metadata["cloudy_optical"]
    if optical["crs"] and "4326" in optical["crs"]:
        west, south, east, north = optical["bounds"]
        metadata["center"] = [(south + north) / 2, (west + east) / 2]
    else:
        metadata["center"] = None
    metadata["scene_id"] = f'{scene["row"]["roi"]} · patch {scene["row"]["patch_id"]}'
    metadata["acquisition_date"] = scene["row"]["cloudy_date"]
    return bands, metadata


@router.get("/analytics/model-comparison")
async def get_model_comparison():
    report = REPORTS_ROOT / "model_comparison.json"
    if not report.is_file():
        raise HTTPException(status_code=404, detail="No saved model comparison report is available")
    try:
        return json.loads(report.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=500, detail="The saved model comparison report is unreadable") from exc


@router.get("/analytics/training-history")
async def get_training_history():
    histories = {}
    for name, label in (
        ("training_history_multimodal_unet.json", "multimodal_unet"),
        ("training_history_optical_only_unet.json", "optical_only_unet"),
    ):
        path = REPORTS_ROOT / name
        if path.is_file():
            try:
                histories[label] = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise HTTPException(status_code=500, detail=f"Saved training history {name} is unreadable") from exc
    return {"histories": histories}


@router.get("/dataset/sample")
async def get_test_sample():
    _, metadata = await run_in_threadpool(_read_test_scene)
    return metadata


@router.get("/dataset/sample/{kind}")
async def get_test_sample_preview(kind: str):
    if kind not in {"cloudy_optical", "sar", "reference"}:
        raise HTTPException(status_code=404, detail="Unknown sample preview")
    previews, _ = await run_in_threadpool(_read_test_scene)
    buffer = BytesIO()
    Image.fromarray(previews[kind]).save(buffer, format="PNG")
    return Response(content=buffer.getvalue(), media_type="image/png")


async def _read_upload(upload: UploadFile, maximum_bytes: int) -> bytes:
    content = await upload.read(maximum_bytes + 1)
    if len(content) > maximum_bytes:
        raise HTTPException(status_code=413, detail=f"{upload.filename} exceeds the upload size limit")
    return content


def _to_input_error(exc: Exception) -> HTTPException:
    return HTTPException(status_code=422, detail=str(exc))


@router.get("/health", response_model=HealthResponse)
async def health(request: Request):
    model_status = request.app.state.model_service.status()
    return {"status": "ok" if model_status["available"] else "degraded", "model": model_status}


@router.get("/dataset/info", response_model=DatasetInfoResponse)
async def get_dataset_info(request: Request):
    return await run_in_threadpool(dataset_info, request.app.state.settings.dataset_root)


@router.post("/predict")
async def predict(
    request: Request,
    optical: Annotated[UploadFile, File(description="Cloudy optical image: multiband GeoTIFF or 1/3/4-band PNG/JPEG")],
    sar: Annotated[UploadFile, File(description="Corresponding SAR image with configured channel count")],
    cloud_mask: Annotated[UploadFile | None, File(description="Optional 1-band mask: 1 means obscured; 0 means clear")] = None,
):
    settings = request.app.state.settings
    files = [optical, sar] + ([cloud_mask] if cloud_mask is not None else [])
    payload_bytes = [await _read_upload(item, settings.max_upload_bytes) for item in files]
    try:
        images = await run_in_threadpool(
            lambda: [
                read_image(content, item.filename or "upload", settings.max_decoded_bytes)
                for content, item in zip(payload_bytes, files)
            ]
        )
        optical_array = normalize_image(images[0], settings.optical_channels, "Cloudy optical image")
        sar_array = normalize_image(images[1], settings.sar_channels, "SAR image")
        mask_array = None
        if cloud_mask is not None:
            mask_array = normalize_image(images[2], 1, "Cloud mask")
            if not ((mask_array == 0) | (mask_array == 1)).all():
                raise InputValidationError("Cloud mask values must be binary (0 = clear, 1 = obscured)")
        aligned = [optical_array, sar_array]
        names = ["Cloudy optical image", "SAR image"]
        if mask_array is not None:
            aligned.append(mask_array)
            names.append("Cloud mask")
        validate_aligned(*aligned, names=tuple(names))
        geospatial_alignment_verified = validate_spatial_alignment(images, tuple(names))
        result = await run_in_threadpool(
            request.app.state.model_service.predict, optical_array, sar_array, mask_array
        )
        body, media_type, filename = await run_in_threadpool(encode_prediction, result, images[0])
        metadata = {
            "filename": filename,
            "height": int(result.shape[0]),
            "width": int(result.shape[1]),
            "channels": int(result.shape[2]),
            "value_range": [float(result.min()), float(result.max())],
            "format": media_type,
            "model_device": request.app.state.model_service.status()["device"],
            "geospatial_alignment_verified": geospatial_alignment_verified,
        }
        request.app.state.experiment_store.add("prediction", metadata)
    except ModelUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except InputValidationError as exc:
        raise _to_input_error(exc) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction processing failed: {exc}") from exc

    archive = BytesIO()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr(filename, body)
        bundle.writestr("metadata.json", json.dumps(metadata, indent=2))
    return Response(
        content=archive.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="prediction.zip"'},
    )


@router.post("/evaluate")
async def evaluate_images(
    request: Request,
    reconstruction: Annotated[UploadFile, File(description="Reconstructed optical image")],
    reference: Annotated[UploadFile, File(description="Paired clear optical reference")],
):
    settings = request.app.state.settings
    limit = settings.max_upload_bytes
    contents = [await _read_upload(item, limit) for item in (reconstruction, reference)]
    try:
        images = await run_in_threadpool(
            lambda: [
                read_image(content, upload.filename or "upload", settings.max_decoded_bytes)
                for content, upload in zip(contents, (reconstruction, reference))
            ]
        )
        channels = images[0].array.shape[2]
        reconstructed = normalize_image(images[0], channels, "Reconstruction")
        clear = normalize_image(images[1], channels, "Clear reference")
        validate_aligned(reconstructed, clear, names=("Reconstruction", "Clear reference"))
        metrics = await run_in_threadpool(evaluate, reconstructed, clear)
        request.app.state.experiment_store.add("evaluation", metrics)
    except InputValidationError as exc:
        raise _to_input_error(exc) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {exc}") from exc
    return {"metrics": metrics, "data_range": 1.0, "aggregation": "all bands, channel-wise SSIM"}


@router.get("/experiments")
async def get_experiments(request: Request, limit: int = 50):
    if not 1 <= limit <= 200:
        raise HTTPException(status_code=422, detail="limit must be between 1 and 200")
    return {"experiments": await run_in_threadpool(request.app.state.experiment_store.list, limit)}


@router.post("/train", response_model=TrainingStatus, status_code=202)
async def start_training(request: Request, body: TrainingRequest | None = None):
    try:
        return await run_in_threadpool(
            request.app.state.training_manager.start,
            body.config_path if body else None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/train/{job_id}", response_model=TrainingStatus)
async def training_status(request: Request, job_id: str):
    result = request.app.state.training_manager.get(job_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Training job not found")
    return result
