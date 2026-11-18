from dataclasses import dataclass
import io
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

from backend.core.errors import InputValidationError

try:
    import rasterio
    from rasterio.io import MemoryFile
except ImportError:
    rasterio = None
    MemoryFile = None


@dataclass
class ImageData:
    array: np.ndarray
    filename: str
    profile: dict[str, Any] | None = None


def read_image(
    content: bytes,
    filename: str,
    max_decoded_bytes: int = 128 * 1024 * 1024,
) -> ImageData:
    suffix = Path(filename).suffix.lower()
    if not content:
        raise InputValidationError(f"{filename}: uploaded file is empty")
    if suffix in {".tif", ".tiff"}:
        if rasterio is None or MemoryFile is None:
            raise InputValidationError("GeoTIFF support requires rasterio to be installed")
        try:
            with MemoryFile(content) as memory_file:
                with memory_file.open() as dataset:
                    decoded_bytes = dataset.width * dataset.height * dataset.count * max(
                        np.dtype(dtype).itemsize for dtype in dataset.dtypes
                    )
                    if decoded_bytes > max_decoded_bytes:
                        raise InputValidationError(
                            f"{filename}: decoded image exceeds the {max_decoded_bytes}-byte processing limit"
                        )
                    array = dataset.read()
                    profile = dataset.profile.copy()
        except InputValidationError:
            raise
        except Exception as exc:
            raise InputValidationError(f"{filename}: invalid or unreadable GeoTIFF") from exc
        array = np.moveaxis(array, 0, -1)
        return ImageData(array=array, filename=filename, profile=profile)

    if suffix not in {".png", ".jpg", ".jpeg"}:
        raise InputValidationError(f"{filename}: use PNG, JPEG, or GeoTIFF")
    try:
        with Image.open(io.BytesIO(content)) as image:
            if image.mode == "P":
                raise InputValidationError(f"{filename}: palette-encoded images are not supported")
            mode_sizes = {
                "1": 1,
                "L": 1,
                "P": 1,
                "LA": 1,
                "RGB": 1,
                "RGBA": 1,
                "CMYK": 1,
                "I;16": 2,
                "I;16B": 2,
                "I;16L": 2,
                "I": 4,
                "F": 4,
            }
            decoded_bytes = image.width * image.height * len(image.getbands()) * mode_sizes.get(image.mode, 4)
            if decoded_bytes > max_decoded_bytes:
                raise InputValidationError(
                    f"{filename}: decoded image exceeds the {max_decoded_bytes}-byte processing limit"
                )
            array = np.asarray(image)
    except InputValidationError:
        raise
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise InputValidationError(f"{filename}: invalid or unreadable image") from exc
    if array.ndim == 2:
        array = array[..., np.newaxis]
    return ImageData(array=array, filename=filename)


def normalize_image(image: ImageData, expected_channels: int, name: str) -> np.ndarray:
    array = image.array
    if array.ndim != 3:
        raise InputValidationError(f"{name} must be a 2D image with channel bands")
    if array.shape[2] != expected_channels:
        raise InputValidationError(
            f"{name} has {array.shape[2]} bands; the configured model requires {expected_channels}"
        )
    if array.shape[0] < 16 or array.shape[1] < 16:
        raise InputValidationError(f"{name} must be at least 16 by 16 pixels")
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise InputValidationError(f"{name} contains non-numeric or non-finite values")

    if image.profile is None and np.issubdtype(array.dtype, np.integer) and array.dtype != np.uint8:
        raise InputValidationError(f"{name} PNG/JPEG must use 8-bit integer samples; use GeoTIFF for multispectral data")
    array = array.astype(np.float32)
    # Integer PNG/JPEG values conventionally encode [0, 255]; GeoTIFF values
    # must already use the documented [0, 1] model input scale.
    if image.profile is None and np.issubdtype(image.array.dtype, np.integer):
        array /= float(np.iinfo(image.array.dtype).max)
    if array.min() < 0 or array.max() > 1:
        raise InputValidationError(f"{name} values must be normalized to [0, 1]")
    return array


def validate_aligned(*arrays: np.ndarray, names: tuple[str, ...]) -> None:
    reference_shape = arrays[0].shape[:2]
    for array, name in zip(arrays[1:], names[1:]):
        if array.shape[:2] != reference_shape:
            raise InputValidationError(f"{name} dimensions must match the cloudy optical image")


def validate_spatial_alignment(images: list[ImageData], names: tuple[str, ...]) -> bool:
    profiles = [image.profile for image in images]
    georeferenced = [
        (profile, name) for profile, name in zip(profiles, names) if profile is not None
    ]
    if not georeferenced:
        return False
    reference, reference_name = georeferenced[0]
    reference_transform = reference.get("transform")
    for profile, name in georeferenced[1:]:
        if profile.get("crs") != reference.get("crs"):
            raise InputValidationError(f"{name} CRS must match {reference_name}")
        transform = profile.get("transform")
        if reference_transform is None or transform is None:
            continue
        if not np.allclose(tuple(transform), tuple(reference_transform), rtol=0, atol=1e-9):
            raise InputValidationError(f"{name} geotransform must match {reference_name}")
    return (
        len(georeferenced) == len(profiles)
        and reference.get("crs") is not None
        and reference_transform is not None
    )


def encode_prediction(array: np.ndarray, input_image: ImageData) -> tuple[bytes, str, str]:
    if input_image.profile is not None:
        if rasterio is None or MemoryFile is None:
            raise RuntimeError("GeoTIFF support requires rasterio")
        profile = input_image.profile.copy()
        profile.update(
            count=array.shape[2],
            dtype="float32",
            height=array.shape[0],
            width=array.shape[1],
        )
        with MemoryFile() as memory_file:
            with memory_file.open(**profile) as dataset:
                dataset.write(np.moveaxis(array.astype(np.float32), -1, 0))
            return memory_file.read(), "image/tiff", "reconstruction.tif"

    if array.shape[2] in {1, 3, 4}:
        output = np.clip(array * 255, 0, 255).astype(np.uint8)
        if output.shape[2] == 1:
            output = output[:, :, 0]
        buffer = io.BytesIO()
        Image.fromarray(output).save(buffer, format="PNG")
        return buffer.getvalue(), "image/png", "reconstruction.png"

    buffer = io.BytesIO()
    np.savez_compressed(buffer, reconstruction=array.astype(np.float32))
    return buffer.getvalue(), "application/octet-stream", "reconstruction.npz"
