"""
webapp/server.py — the FastAPI application.

Endpoints
─────────
  GET    /                          the single-page UI
  GET    /api/exercises             exercises the pipeline actually implements
  POST   /api/jobs                  upload a video, create an analysis
  GET    /api/jobs                  every job in this server session
  GET    /api/jobs/{id}             status + progress (+ summary once finished)
  WS     /api/jobs/{id}/stream      run the analysis, frame by frame, live
  GET    /api/jobs/{id}/result      full result document, timeline included
  GET    /api/jobs/{id}/video       annotated video (range requests supported)
  GET    /api/jobs/{id}/files/{name} download one generated CSV
  DELETE /api/jobs/{id}             cancel and/or delete a job
  GET    /api/audio/{clip}          a coaching clip, for the browser to play

`POST /api/jobs` does not start anything on its own.  In the default `stream`
mode it stores the video and waits for the WebSocket above to attach, which is
what makes the analysis start when the user is watching it; `batch` mode queues
it on the worker pool for a client that only wants the finished document.

The exercise list is DERIVED from `session.EXERCISE_REGISTRY`, never hardcoded —
the same discipline `session.rest_banner_text()` follows — so registering a new
exercise makes it selectable in the browser with no change to this file.
"""

import asyncio
import os
import re
from contextlib import asynccontextmanager

from fastapi import (
    FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.gzip import GZipMiddleware

from config import (
    ALLOWED_VIDEO_EXTS, BASELINE_FRAMES, FRONTEND_DIR, MAX_CONCURRENT_JOBS,
    MAX_FRAMES, MAX_UPLOAD_BYTES, STREAM_ATTACH_TIMEOUT_S, UPLOAD_CHUNK_BYTES,
)
from src import series
from src.audio_bridge import report_missing_clips, resolve_clip
from src.jobs import (
    MODE_BATCH, MODE_STREAM, STATUS_DONE, STATUS_QUEUED, STATUS_RUNNING, JobStore,
)
from src.live_session import LiveAnalysisSession
from src.ranged import ranged_file_response
from src.session import EXERCISE_REGISTRY, _pretty_label
from src.signal_source import SIGNAL_NULL, SIGNAL_REST, LiveSignalSource

_SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]+$")

VIDEO_MEDIA_TYPES = {".mp4": "video/mp4", ".webm": "video/webm"}
AUDIO_MEDIA_TYPES = {".mp3": "audio/mpeg", ".wav": "audio/wav",
                     ".ogg": "audio/ogg", ".flac": "audio/flac",
                     ".m4a": "audio/mp4", ".aiff": "audio/aiff"}


def create_app() -> FastAPI:
    store = JobStore()
    # Admission control for live runs, held for the duration of one analysis.
    # Constructed lazily inside the endpoint's own loop rather than here, so the
    # app object stays importable outside a running event loop (uvicorn --reload
    # and the OpenAPI dump both do that).
    slots: dict = {}

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        # The browser does the playing here, so nothing else in this process
        # would ever notice an empty assets/audio/ — see report_missing_clips.
        report_missing_clips()
        yield
        # Stop the worker pool and signal any in-flight analysis to unwind, so
        # Ctrl+C does not leave a MediaPipe run holding the job folder open.
        store.shutdown()

    def analysis_slot() -> asyncio.Semaphore:
        if "sem" not in slots:
            slots["sem"] = asyncio.Semaphore(MAX_CONCURRENT_JOBS)
        return slots["sem"]

    app = FastAPI(title="Module 2 — Pose Correction", docs_url="/api/docs",
                  redoc_url=None, lifespan=lifespan)
    # The result document is mostly repetitive per-frame text, which compresses
    # by well over an order of magnitude.
    app.add_middleware(GZipMiddleware, minimum_size=1024)
    app.state.store = store

    # ── UI ───────────────────────────────────────────────────────────────────
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

    # ── Metadata ─────────────────────────────────────────────────────────────
    @app.get("/api/exercises")
    def exercises():
        """
        Implemented exercises, derived from the registry the pipeline routes on.

        `EXERCISE_REGISTRY` is the single source of truth for what
        `SessionController` can actually activate, so an exercise that exists as
        a class but is not registered correctly never appears as a choice.
        """
        items = []
        for label, cls in EXERCISE_REGISTRY.items():
            items.append({
                "id": label,
                "name": _pretty_label(label),
                "calibration": cls.calibration_pose_description,
                # Shipped up front so the live view can draw an exercise's charts
                # — axes, thresholds and all — from the first frame, and can
                # rebuild them if the signal names a different exercise partway
                # through.  Waiting for the result document would mean no charts
                # until the run had already finished.
                "charts": series.charts_for(label),
            })
        return {
            "exercises": items,
            "calibration_frames": BASELINE_FRAMES,
            "signals": _signal_vocabulary(),
            "limits": {"max_upload_bytes": MAX_UPLOAD_BYTES,
                       "max_frames": MAX_FRAMES},
        }

    # ── Jobs ─────────────────────────────────────────────────────────────────
    @app.post("/api/jobs", status_code=201)
    async def create_job(video: UploadFile = File(...), exercise: str = Form(...),
                         mode: str = Form(MODE_STREAM)):
        if exercise not in EXERCISE_REGISTRY:
            raise HTTPException(400, f"'{exercise}' is not an implemented exercise.")
        if mode not in (MODE_STREAM, MODE_BATCH):
            raise HTTPException(400, f"'{mode}' is not a valid mode.")

        ext = os.path.splitext(video.filename or "")[1].lower()
        if ext not in ALLOWED_VIDEO_EXTS:
            raise HTTPException(
                400, f"Unsupported file type '{ext or 'unknown'}'. "
                     f"Accepted: {', '.join(sorted(ALLOWED_VIDEO_EXTS))}.")

        job = store.create(exercise, os.path.basename(video.filename or "upload"),
                           mode=mode)
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

        if mode == MODE_BATCH:
            store.submit(job, upload_path)
        else:
            # Held, not queued: a streamed job starts when its socket attaches.
            # If the tab never opens one — the upload response was the last thing
            # that happened — the video is reclaimed rather than left on disk.
            job.video_path = upload_path
            asyncio.get_running_loop().call_later(
                STREAM_ATTACH_TIMEOUT_S, _reap_unattached, store, job.id)
        return job.as_dict(store.queue_position(job))

    @app.get("/api/jobs")
    def list_jobs():
        return {"jobs": store.list()}

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: str):
        job = _require(store, job_id)
        return job.as_dict(store.queue_position(job))

    # ── Live analysis ────────────────────────────────────────────────────────
    @app.websocket("/api/jobs/{job_id}/stream")
    async def job_stream(websocket: WebSocket, job_id: str):
        """
        Analyse the job's video, emitting every frame's findings as it is made.

        The socket is the RUN, not a view of one: nothing is processed until a
        client is here to watch, and closing the tab stops the analysis at the
        next frame boundary.  See src/live_session.py for the wire format and
        for why the producer runs in its own thread behind a bounded queue.
        """
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
            # A second run over the same folder would interleave two sets of
            # writes into one annotated video and one set of CSVs.
            await _ws_error(websocket, "This analysis is already being watched "
                                       "in another tab.")
            return

        # Starts on the exercise the upload form chose, and is settable per frame
        # from here on — the seam Module 1 takes over (src/signal_source.py).
        signal = LiveSignalSource(job.exercise)
        session = LiveAnalysisSession(store, job, job.video_path, signal)
        receiver = asyncio.create_task(session.receive(websocket))
        try:
            semaphore = analysis_slot()
            if semaphore.locked():
                await _ws_send(websocket, {"type": "status", "status": "queued",
                                           "detail": "Another analysis is using "
                                                     "the CPU; starting shortly."})
            async with semaphore:
                if job.cancel_event.is_set():
                    await _ws_send(websocket, {"type": "status",
                                               "status": "cancelled"})
                    return
                await _ws_send(websocket, {"type": "status", "status": "running",
                                           "signals": _signal_vocabulary()})
                await session.pump(websocket)
            if session.error:
                await _ws_error(websocket, session.error)
            elif session.cancelled:
                await _ws_send(websocket, {"type": "status", "status": "cancelled"})
        except (WebSocketDisconnect, RuntimeError):
            # A client that walks away mid-run is an ordinary end of session,
            # not a server fault: the send that discovers the dead socket raises
            # WebSocketDisconnect, or RuntimeError if the close frame had
            # already been processed.  `pump` unwinds its worker either way.
            pass
        finally:
            receiver.cancel()
            job.cancel_event.set()
            await _close_quietly(websocket)

    @app.get("/api/jobs/{job_id}/result")
    def job_result(job_id: str):
        job = _require(store, job_id)
        if job.status != STATUS_DONE:
            raise HTTPException(409, f"Job is {job.status}, not finished.")
        path = os.path.join(job.directory, "result.json")
        if not os.path.exists(path):
            raise HTTPException(404, "Result file is missing.")
        # Served straight from disk: it is already JSON, and re-encoding a
        # multi-megabyte timeline on every request would be pure waste.
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
            # Let the worker unwind at its next frame boundary; it closes the
            # video writer and CSV handles in a finally block, then removes the
            # folder itself.  Deleting the directory from under it would leave
            # half-written files on Windows, where an open handle blocks removal.
            store.cancel(job)
            return JSONResponse({"cancelled": True})
        store.delete(job_id)
        return JSONResponse({"deleted": True})

    # ── Audio clips ──────────────────────────────────────────────────────────
    @app.get("/api/audio/{clip}")
    def audio_clip(clip: str, request: Request):
        """
        Serve a coaching clip by basename, resolved through the same index the
        desktop player uses, so the browser plays the identical file.
        """
        path = resolve_clip(clip)
        if path is None or not os.path.exists(path):
            raise HTTPException(404, f"Unknown audio clip '{clip}'.")
        media = AUDIO_MEDIA_TYPES.get(os.path.splitext(path)[1].lower(),
                                      "application/octet-stream")
        return ranged_file_response(path, request, media)

    return app


def _signal_vocabulary():
    """
    Every Module-1 label this build can act on, in the order a UI should offer
    them: the two idle states, then the implemented exercises.

    Derived from `EXERCISE_REGISTRY` for the same reason the exercise list is —
    a label the browser can send but `SessionController` cannot route is a
    control that silently does nothing.
    """
    return ([{"label": SIGNAL_NULL, "name": "Null", "kind": "idle"},
             {"label": SIGNAL_REST, "name": "Rest", "kind": "idle"}]
            + [{"label": label, "name": _pretty_label(label), "kind": "exercise"}
               for label in EXERCISE_REGISTRY])


def _reap_unattached(store, job_id):
    """Drop a streamed job whose browser never opened its socket."""
    job = store.get(job_id)
    if job is not None and not job.attached and job.status == STATUS_QUEUED:
        store.delete(job_id)


async def _ws_send(websocket, payload):
    """
    Best-effort status message.

    Every caller is reporting on a run that has already happened or is about to;
    none of them can do anything useful if the client has gone, and a send to a
    closed socket raising here would turn a normal disconnect into a logged
    server error.  The frame stream itself is NOT sent this way — there, a
    failed send is exactly the disconnect signal `pump` needs.
    """
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
        pass                    # already closed by the client, or mid-handshake


def _require(store, job_id):
    job = store.get(job_id)
    if job is None:
        raise HTTPException(404, "No such job (it may have been evicted).")
    return job


def _job_file(job, name):
    """
    Resolve a generated file inside a job folder, refusing anything else.

    Two independent checks: the name must be a plain filename, and the resolved
    path must still sit inside the job directory.  The second catches anything
    the first misses (symlinks, unusual separators) rather than trusting one
    pattern to be exhaustive.
    """
    if not name or not _SAFE_NAME.fullmatch(name):
        raise HTTPException(400, "Invalid file name.")
    root = os.path.realpath(job.directory)
    path = os.path.realpath(os.path.join(root, name))
    if os.path.commonpath([root, path]) != root or not os.path.isfile(path):
        raise HTTPException(404, "No such file for this job.")
    return path


app = create_app()
