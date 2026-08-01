"""
src/live_session.py — the WebSocket a browser watches an analysis through.

The pipeline is CPU-bound, synchronous and about four times slower than the clip
it is analysing.  Making the browser wait for the whole run before showing
anything is what this module exists to stop: `analyse_stream` finishes one frame
at a time, and every frame is on the wire the moment it is done, so the overlay,
the risk zones, the charts and the coaching audio all appear while the rest of
the video is still being processed.

Three problems that shape the design
────────────────────────────────────
**Blocking code in an async server.**  The frame loop cannot be made async — it
is MediaPipe and OpenCV.  So it runs in its own thread and hands events to the
event loop through a queue, rather than blocking the loop that is also serving
every other request.

**Backpressure.**  The producer can outrun a socket (a backgrounded tab, a phone
on a weak link).  The queue is therefore SMALL and BOUNDED: once it is full the
analysis thread blocks in `put`, which throttles the run to the speed of the
client instead of growing an unbounded backlog of JPEG frames in memory.  A deep
queue would be worse than useless here — it would only let the picture the user
is watching drift further behind the frame actually being analysed.

**Admission.**  Inference is single-threaded per landmarker, so a second
concurrent run mostly steals cycles from the first.  Sessions wait on an
`asyncio.Semaphore` of MAX_CONCURRENT_JOBS *before* a thread is started, and are
told they are queued in the meantime; a waiting client then costs one suspended
coroutine rather than a thread parked on a queue.

Wire format
───────────
Control messages are JSON text in both directions.  Frames are ONE binary
message each:

    [4-byte big-endian header length][UTF-8 JSON header][JPEG bytes]

One message rather than a JSON message followed by a binary one, so a header can
never be paired with the wrong image, and binary rather than base64 inside the
JSON, which would inflate every frame by a third for nothing.

  server -> client   {"type": "meta",   ...}          once, before frame 0
                     {"type": "status", ...}          queued / running / cancelled
                     <binary frame>                   one per analysed frame
                     {"type": "done",   "summary": …} result document, timeline
                                                      omitted — the client has
                                                      been given every frame of
                                                      it already
                     {"type": "error",  "message": …}

  client -> server   {"type": "signal", "label": "Squat" | "Rest" | "Null" | …}
                     {"type": "stop"}

The `signal` message is the Module-1 seam.  Today the browser sends it; when
Module 1 is integrated its classifier becomes the producer of exactly these
labels and nothing else in this file changes.
"""

import asyncio
import json
import queue
import struct
import threading
import traceback

from config import STREAM_QUEUE_FRAMES
from src.analysis import (
    EVENT_DONE, EVENT_FRAME, EVENT_META, AnalysisCancelled, analyse_stream,
)
from src.data_export import json_default

# Sentinel pushed by the producer so the consumer knows the run is over, whether
# it ended by completing, cancelling or raising.
_END = object()

HEADER_STRUCT = struct.Struct(">I")


def encode_frame(header: dict, image: bytes | None) -> bytes:
    """Pack one frame event into the single binary message described above."""
    blob = json.dumps(header, default=json_default).encode("utf-8")
    return HEADER_STRUCT.pack(len(blob)) + blob + (image or b"")


class LiveAnalysisSession:
    """
    One video, one socket: the producer thread, the queue between them, and the
    two coroutines that drain it and listen for control messages.

    The job's existing cancel event is reused as the single stop signal, so a
    `DELETE /api/jobs/{id}` and a closed browser tab unwind through the same
    path the batch worker already uses.
    """

    def __init__(self, store, job, video_path, signal_source):
        self.store = store
        self.job = job
        self.video_path = video_path
        self.signal = signal_source
        self.cancel = job.cancel_event
        self._queue = queue.Queue(maxsize=STREAM_QUEUE_FRAMES)
        self.error: str | None = None
        self.cancelled = False

    # ── Producer (worker thread) ─────────────────────────────────────────────
    def _produce(self):
        """
        Drive the frame loop, pushing every event into the bounded queue.

        Runs off the event loop entirely.  `queue.put` blocking when the client
        is slow is the intended behaviour, not a stall to be engineered away.
        """
        try:
            for event in analyse_stream(self.video_path, self.signal,
                                        self.job.directory, stream_frames=True,
                                        progress=self._progress,
                                        should_cancel=self.cancel.is_set):
                if event["type"] == EVENT_DONE:
                    # Persisted BEFORE the client is told the run finished, so a
                    # browser that immediately asks for /result or reloads the
                    # page never races the write.
                    self.store.record_result(self.job, event["result"])
                self._queue.put(event)
        except AnalysisCancelled:
            self.cancelled = True
            self.store.record_cancelled(self.job)
        except BaseException as exc:                   # surfaced to the browser
            self.error = str(exc) or exc.__class__.__name__
            self.store.record_failure(self.job, self.error)
            traceback.print_exc()
        finally:
            self._queue.put(_END)

    def _progress(self, done, total):
        self.job.frames_done = done
        # A container may not declare a frame count (or may lie about it); keep
        # the bar honest by growing the denominator instead of pinning progress
        # at 100% while frames keep arriving.
        self.job.frames_total = max(total, done)

    # ── Consumer (event loop) ────────────────────────────────────────────────
    async def pump(self, websocket):
        """Forward every event to the socket until the run ends or the client goes."""
        loop = asyncio.get_running_loop()
        thread = threading.Thread(target=self._produce, daemon=True,
                                  name=f"stream-{self.job.id}")
        self.store.mark_running(self.job)
        thread.start()
        ended = False
        try:
            while True:
                event = await loop.run_in_executor(None, self._queue.get)
                if event is _END:
                    ended = True
                    break
                await self._send(websocket, event)
        finally:
            self.cancel.set()
            await loop.run_in_executor(None, self._reap, thread, ended)

    def _reap(self, thread, ended):
        """
        Wait for the producer to stop, draining the queue if it has not already.

        A run that reached `_END` has nothing left to drain, and reading the
        queue again would block on a sentinel that is never coming — the hang
        this argument exists to prevent.

        The other case is a client that went away mid-run.  There, cancelling is
        not enough on its own: the producer may be blocked in `put` on a full
        queue, where it never reaches its cancel check.  Emptying the queue is
        what lets it get back there and unwind.
        """
        if not ended:
            while self._queue.get() is not _END:
                pass
        thread.join(timeout=10.0)

    async def _send(self, websocket, event):
        kind = event["type"]
        if kind == EVENT_FRAME:
            image = event.pop("image", None)
            await websocket.send_bytes(encode_frame(event, image))
        elif kind == EVENT_DONE:
            # The timeline is the one thing deliberately withheld: it is the
            # bulk of the document and the client assembled it frame by frame as
            # the run went.  It stays in result.json for a reload or a download.
            result = event["result"]
            await self._send_json(websocket, {
                "type": EVENT_DONE,
                "summary": {k: v for k, v in result.items() if k != "timeline"},
            })
        else:
            await self._send_json(websocket, event)

    @staticmethod
    async def _send_json(websocket, payload):
        await websocket.send_text(json.dumps(payload, default=json_default))

    # ── Control channel ──────────────────────────────────────────────────────
    async def receive(self, websocket):
        """
        Apply control messages until the client disconnects.

        Never sends: the pump owns the outbound half of the socket, and Starlette
        does not support two coroutines writing to one connection.
        """
        try:
            while True:
                message = await websocket.receive_json()
                kind = message.get("type")
                if kind == "signal":
                    label = message.get("label")
                    if isinstance(label, str) and label:
                        self.signal.set(label)
                elif kind == "stop":
                    self.cancel.set()
                    return
        except asyncio.CancelledError:
            raise
        except Exception:
            # Disconnect, malformed JSON, anything: the client is no longer
            # driving this run, so stop it rather than analysing to the end of a
            # video nobody is watching.
            self.cancel.set()
