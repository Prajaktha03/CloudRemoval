# Multimodal Satellite Cloud Removal

This research workspace contains the existing FastAPI/PyTorch inference API, data and training pipeline, and a Next.js App Router interface in `frontend/`. The browser consumes the API; inference and evaluation remain in the Python service. The existing API uses SQLite for small experiment summaries.

The Next.js dashboard uses the repository's saved SEN12MS-CR-TS reports and scene previews. These reports are historical measurements, not a claim about a newly uploaded scene. Check `/health` before inference: a checkpoint is usable only when the API successfully loads it with the configured model architecture.

The current backend is **not** a Node/Express + PostgreSQL service. It also has no server-side project/dataset catalog, persistent queued-job API, job progress stream, or retained input/output files. The UI labels browser-local project metadata and unavailable job analytics accordingly; it does not simulate those backend features.

## Requirements and installation

Use Python 3.11 (or a Python version supported by the installed PyTorch and Rasterio wheels). From the project root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run locally

Start the backend in one terminal:

```powershell
.\.venv\Scripts\Activate.ps1
python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

Open the generated API documentation at `http://127.0.0.1:8000/docs`.

In another terminal, configure and start the Next.js interface:

```powershell
Set-Location .\frontend
Copy-Item .env.example .env.local
npm install
npm run dev
```

Open `http://localhost:3000`. `NEXT_PUBLIC_API_URL` defaults to `http://localhost:8000`; set it in `frontend/.env.local` when the API runs elsewhere. The backend's default CORS list includes the Next.js and legacy Streamlit origins; set `FRONTEND_ORIGINS` to the exact comma-separated frontend origins for other deployments.

### Sign-in and account provisioning

The workspace requires authentication. There is no public registration endpoint: provision each account from the backend host using the interactive CLI. Create at least one account before signing in:

```powershell
python -m backend.create_admin
```

The account email and password are prompted securely (the password is not echoed). Passwords must be 12–128 characters. Password verifiers use scrypt; API sessions use random opaque tokens stored only as SHA-256 hashes in the existing SQLite database and are issued in an HttpOnly, SameSite=Strict cookie. Sessions expire after 12 hours by default; failed sign-ins are rate-limited per client address.

For HTTPS deployments, set `SESSION_COOKIE_SECURE=true`, configure `FRONTEND_ORIGINS` to the exact trusted frontend origins, and serve the API only over HTTPS. Keep the frontend and API on the same site so the strict session cookie is sent. The development default `SESSION_COOKIE_SECURE=false` is only for local HTTP.

The original Streamlit app is still available with `streamlit run app.py`.

## Frontend pages

- `/dashboard` — actual saved test-set metrics, validation history, experiment records, and a manifest-backed geospatial sample.
- `/studio` — GeoTIFF preflight, alignment checks, configured API inference, downloadable prediction archive, before/after slider, and reference-based PSNR/SSIM/MAE/RMSE when evaluation succeeds.
- `/projects` — backend dataset inventory and manifest metadata; project/file metadata created here remains in browser storage and is not uploaded.
- `/history` — successful prediction/evaluation summaries from the existing SQLite experiment API; the service does not retain failed jobs, source imagery, or durations.
- `/analytics` — saved evaluation/training reports with unavailable time/region/failure filters clearly indicated.
- `/comparison` — stored model metric summaries and CSV export; original inputs are available as previews, but model output images, inference times, and SAM are not in the report.

Run frontend checks with:

```powershell
Set-Location .\frontend
npm run typecheck
npm run lint
npm run build
```

The frontend targets the existing Python API. Project CRUD, dataset object storage, a Node/Express service, PostgreSQL persistence, and asynchronous job progress require backend work beyond this interface.

## Configuration

Configuration uses environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `MODEL_CHECKPOINT` | `outputs/checkpoints/generator.pt` | Generator state dictionary or a checkpoint containing `generator` / `model_state_dict` |
| `OPTICAL_CHANNELS` | `13` | Required optical band count |
| `SAR_CHANNELS` | `2` | Required SAR band count |
| `MODEL_BASE_CHANNELS` | `32` | U-Net base width used to construct the checkpoint architecture |
| `MODEL_USES_MASK` | `true` | Whether the checkpoint was trained with a one-channel cloud mask |
| `DATASET_ROOT` | `data/raw` | Dataset directory used by the inventory endpoint |
| `EXPERIMENTS_DB` | `data/experiments.sqlite3` | SQLite database for small experiment summaries only |
| `MAX_UPLOAD_BYTES` | `52428800` | Maximum size of each upload |
| `MAX_DECODED_BYTES` | `134217728` | Maximum decoded size of each image to limit decompression/memory risks |
| `FRONTEND_ORIGINS` | `http://localhost:3000,http://localhost:8501` | Comma-separated allowed CORS origins |
| `SESSION_COOKIE_SECURE` | `false` | Set `true` when serving over HTTPS |
| `SESSION_MAX_AGE_SECONDS` | `43200` | Maximum authenticated session lifetime |
| `TRAIN_COMMAND` | unset | Optional operator-controlled command to start a training process |

To use SEN12MS-CR-TS, download it from the [official project repository](https://github.com/PatrickTUM/SEN12MS-CR-TS) and set `DATASET_ROOT` to the extracted directory. The current `/dataset/info` endpoint inventories files only; it does not claim to validate dataset pairing or alignment. Do not pair unrelated dates or scenes.

The training adapter selects the cloudiest and clearest Sentinel-2 acquisitions per patch using its heuristic cloud detector, then pairs Sentinel-1 by acquisition index. This is not a curated ground-truth cloud mask, and the clear reference may be from a different date. Inspect the generated pairs and treat scores as experimental. Training does not silently substitute synthetic data: unreadable acquisitions are skipped with warnings, while missing datasets or no usable pairs produce an explicit error.

### Example Studio inputs

`data/sample_uploads/sen12ms_test_patch_212/` contains aligned, normalized GeoTIFFs from held-out ROI 140:

- Upload `cloudy_sentinel2.tif`, `sentinel1_sar.tif`, and `cloud_mask_heuristic.tif` for reconstruction.
- Optionally upload `clear_reference_sentinel2_2018-07-18.tif` to calculate comparison metrics.

The cloudy image is dated 2018-02-28; the SAR image is dated 2018-02-26. The mask is a heuristic and covers the full patch. The comparison image is a later observation, not a same-date ground-truth target, so its metrics are illustrative only.

## API

- `POST /auth/login`: verifies a provisioned account and sets an HttpOnly session cookie.
- `GET /auth/me`: returns the current signed-in account.
- `POST /auth/logout`: revokes the current session and clears its cookie.
- All other API endpoints require an authenticated session. State-changing requests are restricted to configured frontend origins.

- `GET /health`: API and loaded model/checkpoint status.
- `GET /dataset/info`: configured dataset path and file inventory.
- `POST /predict`: multipart `optical`, `sar`, and optional `cloud_mask` uploads. Returns a ZIP with the reconstructed image and `metadata.json`.
- `POST /evaluate`: multipart `reconstruction` and paired `reference`; returns PSNR and SSIM with data range 1.0 (`psnr` is `null` with `psnr_is_infinite: true` for an exact match).
- `GET /experiments`: saved prediction/evaluation summaries (`?limit=1..200`).
- `POST /train`: launches the configured `TRAIN_COMMAND`; returns 503 when disabled and 409 if a job is active.
- `GET /train/{job_id}`: in-process training job status.

PNG/JPEG inputs use the conventional 8-bit `[0, 255]` encoding and are normalized to `[0, 1]`. GeoTIFF inputs must already be finite, normalized to `[0, 1]`, and have exactly the configured band count. Optical, SAR, and mask dimensions must match. Masks are binary: `1` is obscured and `0` is clear. If the model expects a mask and none is uploaded, an all-zero mask is used.

Predictions retain multiband output: GeoTIFF for GeoTIFF optical inputs (with the input geospatial profile), PNG for 1/3/4-channel image inputs, otherwise NPZ. The response archive contains a metadata record. GeoTIFF compatibility depends on the source profile being writable by Rasterio.

## Validation

Run the focused API and service tests:

```powershell
python -m pytest tests -q
```

Tests exercise API status, incompatible upload validation, missing-checkpoint behavior, metrics, PNG normalization, and SQLite summary persistence. They do not assert scientific performance or substitute synthetic data for real imagery.

## Important limits

- Checkpoints, dataset files, and saved reports are workspace artifacts. Verify their provenance, loading status, and evaluation protocol before research use.
- The provided U-Net defines the expected model architecture; a supplied checkpoint must match its shape/configuration.
- Training is disabled unless `TRAIN_COMMAND` is explicitly configured by the backend operator. Training command output is captured in job status, but training code and data adapters are not part of this backend-only scaffold.
- Dataset pairing, geospatial alignment, nodata-aware preprocessing, and correct dataset-specific scaling must be validated before research use.
- SQLite stores only small metadata summaries, never satellite images or checkpoints.
