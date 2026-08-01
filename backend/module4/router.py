"""
backend/module4/router.py — Module 4's HTTP surface on the shared server.

`module4/app.py` registered its routes as decorators on its own module-level
`FastAPI()` instance, at `/`, `/health` and `/predict`. Two changes were
needed to host it alongside Modules 1 and 2:

  * an `APIRouter` instead of a second `FastAPI` app, so the routes attach to
    the one server;
  * namespaced paths — its `/` collided head-on with the workout dashboard's
    own `/`, and bare `/health` / `/predict` are too generic for a server that
    now hosts three modules.

    GET  /api/nutrition/health
    POST /api/nutrition/predict

Inference runs in a worker thread. The original handler was `async def` but
called the fully synchronous, ~2 s `analyze()` inline, which on a shared
server would block the event loop — stalling every Module 1 / Module 2
WebSocket frame for the duration of a nutrition request.
"""

import io
import logging
import time

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from starlette.concurrency import run_in_threadpool

from backend.module4 import model
from backend.module4.settings import HF_REPO, HF_TOKEN, MAX_UPLOAD_MB, USE_TTA

log = logging.getLogger("nutri")

router = APIRouter(prefix="/api/nutrition", tags=["nutrition"])


@router.get("/health")
def health():
    return {
        "ok": model.is_loaded(),
        "repo": HF_REPO,
        "device": model.DEVICE,
        "tta": USE_TTA,
        "pseudo_depth": model.S.get("depth_pipe") is not None,
        "max_upload_mb": MAX_UPLOAD_MB,
        "token_set": HF_TOKEN is not None,
        # Weights load on first predict, not at boot — see model.ensure_loaded.
        "model_loaded": model.is_loaded(),
    }


@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    t0 = time.perf_counter()
    filename = file.filename or "<unknown>"
    log.info("[NUTRI] request received  file=%s", filename)

    data = await file.read()
    size_kb = len(data) / 1024
    log.info("[NUTRI] file read         size=%.1f KB", size_kb)

    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        log.warning("[NUTRI] upload rejected   size=%.1f KB > limit=%d MB",
                    size_kb, MAX_UPLOAD_MB)
        raise HTTPException(413, f"Image too large (max {MAX_UPLOAD_MB} MB)")

    try:
        img = Image.open(io.BytesIO(data))
        img.load()
        log.info("[NUTRI] image decoded     mode=%s  size=%dx%d",
                 img.mode, img.width, img.height)
    except Exception as exc:
        log.error("[NUTRI] image decode failed: %s", exc)
        raise HTTPException(400, "File is not a readable image")

    # First call pays the model download/load; both it and inference are
    # blocking, so both go to a worker thread.
    try:
        if not model.is_loaded():
            log.info("[NUTRI] loading model (first request; this can take a while)")
            await run_in_threadpool(model.ensure_loaded)
    except Exception as exc:
        log.exception("[NUTRI] model load failed: %s", exc)
        raise HTTPException(
            503, f"Nutrition model unavailable: {exc}. The weights download "
                 f"from Hugging Face on first use — check the server's network "
                 f"access and try again.")

    try:
        log.info("[NUTRI] running inference  tta=%s  depth=%s  device=%s",
                 USE_TTA, model.S.get("depth_pipe") is not None, model.DEVICE)
        t1 = time.perf_counter()
        result = await run_in_threadpool(model.analyze, img)
        t2 = time.perf_counter()
        tot = result["totals"]
        log.info("[NUTRI] done  inference=%.2fs  total=%.2fs  kcal=%.0f  "
                 "fat=%.1fg  carb=%.1fg  prot=%.1fg  ingredients=%d",
                 t2 - t1, t2 - t0, tot["calories"], tot["fat_g"],
                 tot["carbs_g"], tot["protein_g"], len(result["ingredients"]))
        return JSONResponse(result)
    except Exception as exc:
        log.exception("[NUTRI] inference failed: %s", exc)
        raise HTTPException(500, f"Inference failed: {exc}")
