"""FastAPI web app for selectable Vietnamese dialect classifiers."""

from __future__ import annotations

import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from src.inference.predict import (
    DEFAULT_CNN_CHECKPOINT_PATH,
    DEFAULT_PHOWHISPER_CACHE_DIR,
    DEFAULT_PHOWHISPER_CHECKPOINT_PATH,
    DEFAULT_SVM_MODEL_PATH,
    SUPPORTED_MODELS,
    is_model_loaded,
    load_phowhisper_model,
    load_model,
    load_svm_model,
    loaded_device,
    normalize_model_name,
    predict,
)


STATIC_ROOT = Path(__file__).parent / "static"
NO_STORE_HEADERS = {
    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
    "Pragma": "no-cache",
}
MODEL_DISPLAY_NAMES = {
    "cnn": "Lightweight CNN",
    "svm": "SVM MFCC baseline",
    "phowhisper": "PhoWhisper",
}
MODEL_DESCRIPTIONS = {
    "cnn": "Log-Mel spectrogram model.",
    "svm": "MFCC mean/std traditional baseline.",
    "phowhisper": "PhoWhisper-base dialect classifier.",
}
MODEL_CONFIG: dict[str, Any] = {}


def env_path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default)))


def env_flag(name: str, default: bool = True) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def configure_models() -> None:
    default_model = normalize_model_name(os.environ.get("DEFAULT_MODEL", "cnn"))
    MODEL_CONFIG.clear()
    MODEL_CONFIG.update(
        {
            "default_model": default_model,
            "cnn_checkpoint_path": env_path(
                "CNN_CHECKPOINT_PATH",
                DEFAULT_CNN_CHECKPOINT_PATH,
            ),
            "svm_model_path": env_path("SVM_MODEL_PATH", DEFAULT_SVM_MODEL_PATH),
            "phowhisper_checkpoint_path": env_path(
                "PHOWHISPER_CHECKPOINT_PATH",
                DEFAULT_PHOWHISPER_CHECKPOINT_PATH,
            ),
            "phowhisper_cache_dir": env_path(
                "PHOWHISPER_CACHE_DIR",
                DEFAULT_PHOWHISPER_CACHE_DIR,
            ),
            "cnn_device": os.environ.get("CNN_DEVICE", "auto"),
            "phowhisper_device": os.environ.get(
                "PHOWHISPER_DEVICE",
                os.environ.get("CNN_DEVICE", "auto"),
            ),
            "phowhisper_local_files_only": env_flag(
                "PHOWHISPER_LOCAL_FILES_ONLY",
                True,
            ),
        }
    )


@asynccontextmanager
async def lifespan(_app: FastAPI):
    configure_models()
    yield


app = FastAPI(
    title="Vietnamese Dialect Identification",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_ROOT / "index.html", headers=NO_STORE_HEADERS)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "default_model": MODEL_CONFIG.get("default_model", "cnn"),
        "models": model_options(),
    }


@app.get("/models")
def models() -> dict[str, Any]:
    return {
        "default_model": MODEL_CONFIG.get("default_model", "cnn"),
        "models": model_options(),
    }


@app.post("/predict")
async def predict_upload(file: UploadFile, model: str = Form("cnn")):
    suffix = Path(file.filename or "").suffix
    if not suffix or len(suffix) > 12:
        suffix = ".audio"
    temporary_path: Path | None = None
    try:
        model_name = normalize_model_name(model)
        ensure_model_loaded(model_name)
        with tempfile.NamedTemporaryFile(
            prefix="dialect-upload-",
            suffix=suffix,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            while chunk := await file.read(1024 * 1024):
                temporary.write(chunk)
        result = predict(temporary_path, model_name=model_name)
        return {
            "model": model_name,
            "model_label": MODEL_DISPLAY_NAMES[model_name],
            "prediction": result["predicted_label"],
            "confidence": result["confidence"],
            "probabilities": result["probabilities"],
            "score_type": result["score_type"],
            "device": loaded_device(model_name),
        }
    except (FileNotFoundError, OSError, ValueError) as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})
    except RuntimeError as exc:
        return JSONResponse(status_code=500, content={"error": str(exc)})
    finally:
        await file.close()
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def model_options() -> list[dict[str, Any]]:
    if not MODEL_CONFIG:
        configure_models()
    paths = {
        "cnn": MODEL_CONFIG["cnn_checkpoint_path"],
        "svm": MODEL_CONFIG["svm_model_path"],
        "phowhisper": MODEL_CONFIG["phowhisper_checkpoint_path"],
    }
    options: list[dict[str, Any]] = []
    for name in SUPPORTED_MODELS:
        artifact_exists = paths[name].exists()
        cache_ready = True
        if name == "phowhisper":
            cache_ready = MODEL_CONFIG["phowhisper_cache_dir"].exists()
        loaded = is_model_loaded(name)
        device = loaded_device(name) if loaded else ""
        options.append(
            {
                "name": name,
                "label": MODEL_DISPLAY_NAMES[name],
                "description": MODEL_DESCRIPTIONS[name],
                "artifact_path": paths[name].as_posix(),
                "available": artifact_exists and cache_ready,
                "loaded": loaded,
                "device": device,
                "default": name == MODEL_CONFIG["default_model"],
            }
        )
    return options


def ensure_model_loaded(model_name: str) -> None:
    if not MODEL_CONFIG:
        configure_models()
    if is_model_loaded(model_name):
        return
    if model_name == "cnn":
        load_model(
            MODEL_CONFIG["cnn_checkpoint_path"],
            device=MODEL_CONFIG["cnn_device"],
        )
        return
    if model_name == "svm":
        load_svm_model(MODEL_CONFIG["svm_model_path"])
        return
    if model_name == "phowhisper":
        load_phowhisper_model(
            MODEL_CONFIG["phowhisper_checkpoint_path"],
            device=MODEL_CONFIG["phowhisper_device"],
            cache_dir=MODEL_CONFIG["phowhisper_cache_dir"],
            local_files_only=MODEL_CONFIG["phowhisper_local_files_only"],
        )
        return
    raise ValueError(f"Unsupported model: {model_name}")
