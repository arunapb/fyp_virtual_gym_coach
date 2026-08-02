"""
backend/main.py — the single FastAPI app: one port, one frontend, one
WebSocket, Module 1 and Module 2 running frame-by-frame in parallel.

Import strategy
----------------
This process imports Module 2's `config`/`src` tree (unmodified) exactly as
`module2/api.py` does for itself — inserting module2/'s own directory at the
front of `sys.path` before importing anything from it. Module 1's `src`/
`config` never loads in this process at all; it only ever runs inside the
per-job worker process (see backend/module1_worker.py), which keeps the two
same-named `src`/`config` packages from ever coexisting in one interpreter.

Endpoints mostly mirror `module2/api.py` (job listing/result/video/files/audio
are Module 2's existing behaviour, verbatim), with two differences: uploading
goes through `POST /api/videos` (no exercise pre-selection — Module 1 decides
live) instead of `POST /api/jobs`, and the stream socket runs
`MergedAnalysisSession` instead of `LiveAnalysisSession` so every frame event
also carries Module 1's readout for that frame.
"""

import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
REPO_ROOT = BACKEND_DIR.parent
MODULE2_DIR = REPO_ROOT / "module2"
sys.path.insert(0, str(MODULE2_DIR))

import asyncio  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from contextlib import asynccontextmanager  # noqa: E402

from fastapi import (  # noqa: E402
    FastAPI, File, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect,
)
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from starlette.middleware.gzip import GZipMiddleware  # noqa: E402

from config import (  # noqa: E402 — Module 2's config, unmodified
    ALLOWED_VIDEO_EXTS, BASELINE_FRAMES, MAX_CONCURRENT_JOBS, MAX_FRAMES,
    MAX_UPLOAD_BYTES, STREAM_ATTACH_TIMEOUT_S, UPLOAD_CHUNK_BYTES,
)
from src import series  # noqa: E402 — Module 2's, unmodified
from src.audio_bridge import report_missing_clips, resolve_clip  # noqa: E402
from src.jobs import (  # noqa: E402
    MODE_STREAM, STATUS_DONE, STATUS_QUEUED, STATUS_RUNNING, JobStore,
)
from src.ranged import ranged_file_response  # noqa: E402
from src.session import EXERCISE_REGISTRY, _pretty_label  # noqa: E402
from src.signal_source import SIGNAL_NULL, SIGNAL_REST  # noqa: E402

from backend import module1_worker, user_store  # noqa: E402
from backend.live_session import MergedAnalysisSession  # noqa: E402
from backend.module3.router import router as meals_router  # noqa: E402
from backend.module4.router import router as nutrition_router  # noqa: E402

BACKEND_FRONTEND_DIR = BACKEND_DIR / "frontend"

_SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]+$")
VIDEO_MEDIA_TYPES = {".mp4": "video/mp4", ".webm": "video/webm"}
AUDIO_MEDIA_TYPES = {".mp3": "audio/mpeg", ".wav": "audio/wav",
                     ".ogg": "audio/ogg", ".flac": "audio/flac",
                     ".m4a": "audio/mp4", ".aiff": "audio/aiff"}

# Placeholder exercise label for a Module-1-driven job: the real exercise is
# decided live, frame by frame, not chosen at upload time.
AUTO_EXERCISE_LABEL = "Auto"


def create_app() -> FastAPI:
    store = JobStore()
    slots: dict = {}

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        report_missing_clips()
        yield
        store.shutdown()

    def analysis_slot() -> asyncio.Semaphore:
        if "sem" not in slots:
            slots["sem"] = asyncio.Semaphore(MAX_CONCURRENT_JOBS)
        return slots["sem"]

    app = FastAPI(title="Virtual Gym Coach", docs_url="/api/docs",
                  redoc_url=None, lifespan=lifespan)
    app.add_middleware(GZipMiddleware, minimum_size=1024)
    app.state.store = store

    # ── UI ───────────────────────────────────────────────────────────────────
    app.mount("/static", StaticFiles(directory=str(BACKEND_FRONTEND_DIR)), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return FileResponse(os.path.join(BACKEND_FRONTEND_DIR, "index.html"))

    # Demo sign-in screen. There is NO server-side auth behind this — it is a
    # browser-only gate (frontend/auth.js) that exists to open the demo on a
    # login and to give the mandatory goal-setup popup a moment to appear.
    # Every route on this server, including this one, is open.
    @app.get("/login", response_class=HTMLResponse)
    def login_page():
        return FileResponse(os.path.join(BACKEND_FRONTEND_DIR, "login.html"))

    # ── Modules 3 and 4: their own pages, not part of the Module 1 + ────────
    # Module 2 workout pipeline. They share this server and nothing else.
    #
    # Module 4 (nutrition analysis) is fully standalone. Module 3 (meal plan)
    # has one link back to the workout side: every finished analysis appends
    # its exercise to Module 3's log, which moves the 7-day average behind the
    # calorie target shown here — see backend/module3/exercise_log.py.
    @app.get("/nutrition", response_class=HTMLResponse)
    def nutrition_page():
        return FileResponse(os.path.join(BACKEND_FRONTEND_DIR, "nutrition.html"))

    @app.get("/meals", response_class=HTMLResponse)
    def meals_page():
        return FileResponse(os.path.join(BACKEND_FRONTEND_DIR, "meals.html"))

    app.include_router(nutrition_router)
    app.include_router(meals_router)

    # ── Demo users ───────────────────────────────────────────────────────────
    # Username only, no password, no session — see backend/user_store.py. All
    # this buys is that two people using the demo do not inherit each other's
    # workout log, meal preferences and goal weight.
    @app.post("/api/users/login")
    def user_login(payload: dict):
        username = str(payload.get("username", "")).strip()
        if not username:
            raise HTTPException(400, "A username is required.")
        record = user_store.ensure(username)
        return {
            "username": record["username"],
            "slug": record["slug"],
            # True the first time a name is seen — the browser uses it to say
            # so, and it is why their log and preferences start empty.
            "new_user": record["new"],
        }

    @app.get("/api/users")
    def users():
        return {"users": user_store.list_users()}

    # ── Metadata ─────────────────────────────────────────────────────────────
    @app.get("/api/exercises")
    def exercises():
        items = []
        for label, cls in EXERCISE_REGISTRY.items():
            items.append({
                "id": label,
                "name": _pretty_label(label),
                "calibration": cls.calibration_pose_description,
                "charts": series.charts_for(label),
            })
        return {
            "exercises": items,
            "calibration_frames": BASELINE_FRAMES,
            "signals": _signal_vocabulary(),
            "limits": {"max_upload_bytes": MAX_UPLOAD_BYTES, "max_frames": MAX_FRAMES},
        }

    # ── Upload (Module-1 style: no exercise pre-selection) ──────────────────
    @app.post("/api/videos", status_code=201)
    async def upload_video(video: UploadFile = File(...)):
        ext = os.path.splitext(video.filename or "")[1].lower()
        if ext not in ALLOWED_VIDEO_EXTS:
            raise HTTPException(
                400, f"Unsupported file type '{ext or 'unknown'}'. "
                     f"Accepted: {', '.join(sorted(ALLOWED_VIDEO_EXTS))}.")

        job = store.create(AUTO_EXERCISE_LABEL,
                           os.path.basename(video.filename or "upload"), mode=MODE_STREAM)
        upload_path = os.path.join(job.upload_dir, f"source{ext}")
        written = 0
        try:
            with open(upload_path, "wb") as fh:
                while True:
                    chunk = await video.read(UPLOAD_CHUNK_BYTES)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > MAX_UPLOAD_BYTES:
                        raise HTTPException(
                            413, f"Video is larger than the "
                                 f"{MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.")
                    fh.write(chunk)
        except HTTPException:
            store.delete(job.id)
            raise
        finally:
            await video.close()

        if written == 0:
            store.delete(job.id)
            raise HTTPException(400, "The uploaded file was empty.")

        job.video_path = upload_path
        asyncio.get_running_loop().call_later(
            STREAM_ATTACH_TIMEOUT_S, _reap_unattached, store, job.id)
        return {"job_id": job.id}

    @app.get("/api/jobs")
    def list_jobs():
        return {"jobs": store.list()}

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: str):
        job = _require(store, job_id)
        return job.as_dict(store.queue_position(job))

    # ── Live analysis: Module 1 + Module 2, frame by frame ──────────────────
    @app.websocket("/api/jobs/{job_id}/stream")
    async def job_stream(websocket: WebSocket, job_id: str, user: str = ""):
        # `user` is the demo username from the query string — a WebSocket
        # handshake cannot carry custom headers, which is how every other route
        # receives it (X-Demo-User). It only decides whose exercise log this
        # workout lands in; see backend/user_store.py for why this is not auth.
        await websocket.accept()
        job = store.get(job_id)
        if job is None or job.video_path is None:
            await _ws_error(websocket, "No such job (it may have been evicted).")
            return
        if job.mode != MODE_STREAM:
            await _ws_error(websocket, "This job was created for batch analysis.")
            return
        if job.status != STATUS_QUEUED:
            await _ws_error(websocket, f"This analysis is {job.status}, not "
                                       f"waiting to start. Upload the video "
                                       f"again to re-run it.")
            return
        if not store.claim(job):
            await _ws_error(websocket, "This analysis is already being watched "
                                       "in another tab.")
            return

        # A fresh worker process per job guarantees a fresh Module 1 session
        # (no state carried over from a previous upload) with no explicit
        # reset messaging needed.
        pool = ProcessPoolExecutor(max_workers=1, initializer=module1_worker.init_worker)
        session = MergedAnalysisSession(store, job, job.video_path, pool,
                                        MAX_FRAMES, username=(user or "").strip() or None)
        receiver = asyncio.create_task(session.receive(websocket))
        try:
            semaphore = analysis_slot()
            if semaphore.locked():
                await _ws_send(websocket, {"type": "status", "status": "queued",
                                           "detail": "Another analysis is using "
                                                     "the CPU; starting shortly."})
            async with semaphore:
                if job.cancel_event.is_set():
                    await _ws_send(websocket, {"type": "status", "status": "cancelled"})
                    return
                await _ws_send(websocket, {"type": "status", "status": "running",
                                           "signals": _signal_vocabulary()})
                await session.pump(websocket)
            if session.error:
                await _ws_error(websocket, session.error)
            elif session.cancelled:
                await _ws_send(websocket, {"type": "status", "status": "cancelled"})
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            receiver.cancel()
            job.cancel_event.set()
            pool.shutdown(wait=False, cancel_futures=True)
            await _close_quietly(websocket)

    @app.get("/api/jobs/{job_id}/result")
    def job_result(job_id: str):
        job = _require(store, job_id)
        if job.status != STATUS_DONE:
            raise HTTPException(409, f"Job is {job.status}, not finished.")
        path = os.path.join(job.directory, "result.json")
        if not os.path.exists(path):
            raise HTTPException(404, "Result file is missing.")
        return FileResponse(path, media_type="application/json")

    @app.get("/api/jobs/{job_id}/video")
    def job_video(job_id: str, request: Request):
        job = _require(store, job_id)
        name = (job.summary or {}).get("video", {}).get("file")
        if not name:
            raise HTTPException(404, "No annotated video for this job yet.")
        path = _job_file(job, name)
        media = VIDEO_MEDIA_TYPES.get(os.path.splitext(name)[1].lower(),
                                      "application/octet-stream")
        return ranged_file_response(path, request, media)

    @app.get("/api/jobs/{job_id}/files/{name}")
    def job_file(job_id: str, name: str, request: Request):
        job = _require(store, job_id)
        path = _job_file(job, name)
        return ranged_file_response(path, request, "text/csv", filename=name)

    @app.delete("/api/jobs/{job_id}")
    def delete_job(job_id: str):
        job = _require(store, job_id)
        if job.status == STATUS_RUNNING:
            store.cancel(job)
            return JSONResponse({"cancelled": True})
        store.delete(job_id)
        return JSONResponse({"deleted": True})

    # ── Audio clips ──────────────────────────────────────────────────────────
    @app.get("/api/audio/{clip}")
    def audio_clip(clip: str, request: Request):
        path = resolve_clip(clip)
        if path is None or not os.path.exists(path):
            raise HTTPException(404, f"Unknown audio clip '{clip}'.")
        media = AUDIO_MEDIA_TYPES.get(os.path.splitext(path)[1].lower(),
                                      "application/octet-stream")
        return ranged_file_response(path, request, media)

    return app


def _signal_vocabulary():
    return ([{"label": SIGNAL_NULL, "name": "Null", "kind": "idle"},
             {"label": SIGNAL_REST, "name": "Rest", "kind": "idle"}]
            + [{"label": label, "name": _pretty_label(label), "kind": "exercise"}
               for label in EXERCISE_REGISTRY])


def _reap_unattached(store, job_id):
    job = store.get(job_id)
    if job is not None and not job.attached and job.status == STATUS_QUEUED:
        store.delete(job_id)


async def _ws_send(websocket, payload):
    try:
        await websocket.send_json(payload)
    except Exception:
        pass


async def _ws_error(websocket, message):
    await _ws_send(websocket, {"type": "error", "message": message})
    await _close_quietly(websocket)


async def _close_quietly(websocket):
    try:
        await websocket.close()
    except Exception:
        pass


def _require(store, job_id):
    job = store.get(job_id)
    if job is None:
        raise HTTPException(404, "No such job (it may have been evicted).")
    return job


def _job_file(job, name):
    if not name or not _SAFE_NAME.fullmatch(name):
        raise HTTPException(400, "Invalid file name.")
    root = os.path.realpath(job.directory)
    path = os.path.realpath(os.path.join(root, name))
    if os.path.commonpath([root, path]) != root or not os.path.isfile(path):
        raise HTTPException(404, "No such file for this job.")
    return path


app = create_app()

# Run via `python backend/run.py`, not `uvicorn backend.main:app` from the CLI
# and not `python backend/main.py` directly — see backend/run.py's docstring
# for why this file must never be the multiprocessing entry point.
