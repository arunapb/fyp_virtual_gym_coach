import json
import os
import uuid
import asyncio
from pathlib import Path

import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.session import GymCoachSession
from src.video_io import get_video_capture, read_frame, release_capture
from src.state_rules import _load_model as _load_state_model
from src.exercise_rules import _load_exercise_model

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
FRONTEND_DIR = BASE_DIR / "frontend"
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

app = FastAPI(title="Virtual Gym Coach — Module 1 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "state_model_loaded": _load_state_model() is not None,
        "exercise_model_loaded": _load_exercise_model() is not None,
    }


@app.post("/api/videos")
async def upload_video(file: UploadFile = File(...)):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext or 'unknown'}")

    video_id = f"{uuid.uuid4().hex}{ext}"
    dest_path = UPLOAD_DIR / video_id

    with open(dest_path, "wb") as out_file:
        while chunk := await file.read(1024 * 1024):
            out_file.write(chunk)

    return {"video_id": video_id}


def _safe_upload_path(video_id: str) -> Path:
    """Resolve a video_id to a path strictly inside UPLOAD_DIR (no traversal)."""
    candidate = (UPLOAD_DIR / video_id).resolve()
    if UPLOAD_DIR.resolve() not in candidate.parents:
        raise ValueError("invalid video_id")
    return candidate


async def _read_control_message(websocket: WebSocket, stop_event: asyncio.Event):
    """Background task: listen for the client's {"type": "stop"} message (or a
    disconnect) and signal stop_event when it arrives."""
    try:
        while not stop_event.is_set():
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                stop_event.set()
                return
            text = message.get("text")
            if text:
                try:
                    data = json.loads(text)
                except ValueError:
                    continue
                if data.get("type") == "stop":
                    stop_event.set()
                    return
    except WebSocketDisconnect:
        stop_event.set()


async def _handle_webcam_mode(websocket: WebSocket, session: GymCoachSession):
    """Client streams binary JPEG frames; we process each one as it arrives
    and also accept a JSON {"type": "stop"} control message at any time."""
    while True:
        message = await websocket.receive()
        if message["type"] == "websocket.disconnect":
            return

        frame_bytes = message.get("bytes")
        if frame_bytes:
            update = session.process_frame_bytes(frame_bytes)
            if update:
                await websocket.send_json(update)
            continue

        text = message.get("text")
        if text:
            try:
                data = json.loads(text)
            except ValueError:
                continue
            if data.get("type") == "stop":
                return


async def _handle_video_mode(websocket: WebSocket, session: GymCoachSession, video_path: Path):
    """We drive the frame loop ourselves (reading the uploaded file), paced
    to roughly the video's native FPS, while a background task listens for
    an early client-initiated stop."""
    cap = get_video_capture(str(video_path))
    stop_event = asyncio.Event()
    listener_task = asyncio.create_task(_read_control_message(websocket, stop_event))

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_delay = 1.0 / fps if fps > 0 else 1.0 / 30.0

    try:
        while not stop_event.is_set():
            success, frame = read_frame(cap)
            if not success:
                break
            update = session.process_frame(frame, include_preview=True)
            await websocket.send_json(update)
            await asyncio.sleep(frame_delay)
    finally:
        release_capture(cap)
        listener_task.cancel()
        try:
            video_path.unlink(missing_ok=True)
        except OSError:
            pass


@app.websocket("/ws/session")
async def ws_session(websocket: WebSocket):
    await websocket.accept()

    try:
        first_message = await websocket.receive_json()
    except (ValueError, WebSocketDisconnect):
        await websocket.close(code=1003)
        return

    if first_message.get("type") != "start":
        await websocket.send_json({"type": "error", "message": "First message must be {'type': 'start', ...}"})
        await websocket.close(code=1003)
        return

    mode = first_message.get("mode")
    session = GymCoachSession()
    disconnected = False

    try:
        if mode == "webcam":
            await _handle_webcam_mode(websocket, session)
        elif mode == "video":
            video_id = first_message.get("video_id", "")
            try:
                video_path = _safe_upload_path(video_id)
            except ValueError:
                await websocket.send_json({"type": "error", "message": "Invalid video_id"})
                await websocket.close(code=1003)
                return
            if not video_path.exists():
                await websocket.send_json({"type": "error", "message": "Unknown video_id — upload it via POST /api/videos first"})
                await websocket.close(code=1003)
                return
            await _handle_video_mode(websocket, session, video_path)
        else:
            await websocket.send_json({"type": "error", "message": f"Unknown mode: {mode!r} (use 'webcam' or 'video')"})
            await websocket.close(code=1003)
            return
    except WebSocketDisconnect:
        disconnected = True

    if not disconnected:
        try:
            await websocket.send_json({"type": "summary", "summary": session.get_summary()})
        except WebSocketDisconnect:
            disconnected = True

    if not disconnected:
        await websocket.close()


# Serve the local test frontend last, so it doesn't shadow the /api and /ws routes above.
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
