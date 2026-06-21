"""FastAPI web app for the lightweight CNN dialect classifier."""

from __future__ import annotations

import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from src.inference.predict import (
    DEFAULT_CHECKPOINT_PATH,
    load_model,
    loaded_device,
    predict,
)


STATIC_ROOT = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    checkpoint_path = Path(
        os.environ.get("CNN_CHECKPOINT_PATH", str(DEFAULT_CHECKPOINT_PATH))
    )
    device = os.environ.get("CNN_DEVICE", "auto")
    load_model(checkpoint_path, device=device)
    yield


app = FastAPI(
    title="Vietnamese Dialect Identification",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_ROOT / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "device": loaded_device()}


@app.post("/predict")
async def predict_upload(file: UploadFile):
    suffix = Path(file.filename or "").suffix
    if not suffix or len(suffix) > 12:
        suffix = ".audio"
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix="dialect-upload-",
            suffix=suffix,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            while chunk := await file.read(1024 * 1024):
                temporary.write(chunk)
        result = predict(temporary_path)
        return {
            "prediction": result["predicted_label"],
            "confidence": result["confidence"],
            "probabilities": result["probabilities"],
        }
    except (FileNotFoundError, OSError, ValueError) as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})
    except RuntimeError as exc:
        return JSONResponse(status_code=500, content={"error": str(exc)})
    finally:
        await file.close()
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
