"""
src/jobs.py — analysis jobs: queueing, progress, storage and eviction.

A job is one uploaded video and the folder of artefacts produced from it.  It can
be driven two ways, and both end with the same files on disk:

  * **streamed** (`MODE_STREAM`, what the browser does) — a WebSocket attaches
    and watches the run frame by frame; see src/live_session.py.  The job waits
    for that socket rather than for a worker, because there is no point analysing
    a video into a live view nobody has opened yet.
  * **batch** (`MODE_BATCH`) — queued onto a worker thread, no client attached,
    poll `GET /api/jobs/{id}` for progress.  A 30 s clip at 30 fps is 900 frames
    and the pipeline sustains roughly 11 fps with rendering on the reference
    machine (docs/E7_PERFORMANCE.md), so this is still far too slow to run inside
    a request.

Concurrency: one analysis at a time, whichever mode it is in.  MediaPipe
inference is CPU-bound and single-threaded per landmarker, so running two videos
at once halves the speed of each rather than adding throughput.
"""

import json
import os
import shutil
import threading
import time
import traceback
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from src.analysis import AnalysisCancelled, analyse
from src.data_export import json_default
from config import (
    MAX_CONCURRENT_JOBS, MAX_RETAINED_JOBS, OUTPUTS_DIR, UPLOADS_DIR,
)

# Analysis results live under outputs/, keeping the top-level outputs/ folder
# free for the desktop app's own CSVs.
JOB_OUTPUTS_DIR = os.path.join(OUTPUTS_DIR, "jobs")

STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"

MODE_STREAM = "stream"
MODE_BATCH = "batch"


@dataclass
class Job:
    id: str
    exercise: str
    filename: str
    # Two folders per job, so each top-level directory means what it is named:
    # the posted video lands in uploads/, everything produced from it in outputs/.
    upload_dir: str
    directory: str                       # outputs/jobs/<id> — analysis artefacts
    mode: str = MODE_STREAM
    video_path: str | None = None
    status: str = STATUS_QUEUED
    frames_done: int = 0
    frames_total: int = 0
    error: str | None = None
    summary: dict | None = None          # result document minus the timeline
    created: float = 0.0
    attached: bool = False               # a stream socket has claimed this job
    cancel_event: threading.Event = field(default_factory=threading.Event)

    @property
    def progress(self) -> float:
        if self.status == STATUS_DONE:
            return 1.0
        if self.frames_total <= 0:
            return 0.0
        return min(self.frames_done / self.frames_total, 0.999)

    def as_dict(self, queue_position=None) -> dict:
        return {
            "id": self.id,
            "exercise": self.exercise,
            "filename": self.filename,
            "mode": self.mode,
            "status": self.status,
            "frames_done": self.frames_done,
            "frames_total": self.frames_total,
            "progress": round(self.progress, 4),
            "error": self.error,
            "queue_position": queue_position,
            "summary": self.summary,
        }


class JobStore:
    """In-memory registry over on-disk job folders, with a bounded work queue."""

    def __init__(self):
        self._jobs: "OrderedDict[str, Job]" = OrderedDict()
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_JOBS,
                                        thread_name_prefix="analysis")
        os.makedirs(JOB_OUTPUTS_DIR, exist_ok=True)
        os.makedirs(UPLOADS_DIR, exist_ok=True)

    # ── Lifecycle ────────────────────────────────────────────────────────────
    def create(self, exercise: str, filename: str, mode: str = MODE_STREAM) -> Job:
        job_id = uuid.uuid4().hex[:12]
        upload_dir = os.path.join(UPLOADS_DIR, job_id)
        directory = os.path.join(JOB_OUTPUTS_DIR, job_id)
        os.makedirs(upload_dir, exist_ok=True)
        os.makedirs(directory, exist_ok=True)
        job = Job(id=job_id, exercise=exercise, filename=filename, mode=mode,
                  upload_dir=upload_dir, directory=directory, created=_now())
        with self._lock:
            self._jobs[job_id] = job
        return job

    def submit(self, job: Job, video_path: str) -> None:
        """Queue a batch job onto the worker pool.  Streamed jobs never come here."""
        job.video_path = video_path
        self._pool.submit(self._run, job, video_path)

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def queue_position(self, job: Job) -> int | None:
        """How many jobs are ahead of a queued one (0 = next to start)."""
        if job.status != STATUS_QUEUED:
            return None
        with self._lock:
            waiting = [j for j in self._jobs.values()
                       if j.status == STATUS_QUEUED and j.created < job.created]
        return len(waiting)

    def list(self) -> list:
        with self._lock:
            jobs = list(self._jobs.values())
        return [j.as_dict(self.queue_position(j)) for j in reversed(jobs)]

    def cancel(self, job: Job) -> None:
        job.cancel_event.set()
        if job.status == STATUS_QUEUED:
            job.status = STATUS_CANCELLED

    def delete(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.pop(job_id, None)
        if job is None:
            return False
        job.cancel_event.set()
        _rmtree(job)
        return True

    def claim(self, job: Job) -> bool:
        """
        Take exclusive ownership of a streamed job for one socket.

        Returns False if another socket already has it, so a duplicated tab or a
        replayed job id cannot start a second analysis over the same folder — two
        runs would interleave their writes into one annotated video and one set
        of CSVs.
        """
        with self._lock:
            if job.attached or job.status != STATUS_QUEUED:
                return False
            job.attached = True
            return True

    def shutdown(self) -> None:
        with self._lock:
            for job in self._jobs.values():
                job.cancel_event.set()
        self._pool.shutdown(wait=False, cancel_futures=True)

    # ── Outcomes (shared by the batch worker and the live session) ───────────
    def mark_running(self, job: Job) -> None:
        job.status = STATUS_RUNNING

    def record_result(self, job: Job, result: dict) -> None:
        """Persist a finished analysis and publish its summary."""
        with open(os.path.join(job.directory, "result.json"),
                  "w", encoding="utf-8") as fh:
            json.dump(result, fh, default=json_default)
        job.summary = {k: v for k, v in result.items() if k != "timeline"}
        job.frames_done = result["video"]["frames"]
        job.frames_total = result["video"]["frames"]
        job.status = STATUS_DONE
        self._finalise(job)

    def record_cancelled(self, job: Job) -> None:
        job.status = STATUS_CANCELLED
        _rmtree(job)
        with self._lock:
            self._jobs.pop(job.id, None)

    def record_failure(self, job: Job, message: str) -> None:
        job.status = STATUS_FAILED
        job.error = message
        self._finalise(job)

    def _finalise(self, job: Job) -> None:
        # The upload is only needed while the job runs; the annotated video
        # replaces it afterwards, and originals are the bulk of the disk use.
        if job.video_path:
            _unlink(job.video_path)
        # Eviction runs on the failure path too, so a run of failures cannot
        # accumulate job folders indefinitely.
        self._evict_old()

    # ── Batch worker ─────────────────────────────────────────────────────────
    def _run(self, job: Job, video_path: str) -> None:
        if job.cancel_event.is_set():
            job.status = STATUS_CANCELLED
            return
        self.mark_running(job)

        def progress(done, total):
            job.frames_done = done
            # A container may not declare a frame count (or may lie about it);
            # keep the bar honest by growing the denominator instead of pinning
            # progress at 100% while frames keep arriving.
            job.frames_total = max(total, done)

        try:
            result = analyse(video_path, job.exercise, job.directory,
                             progress=progress,
                             should_cancel=job.cancel_event.is_set)
            self.record_result(job, result)
        except AnalysisCancelled:
            self.record_cancelled(job)
        except Exception as exc:                       # surfaced to the browser
            self.record_failure(job, str(exc) or exc.__class__.__name__)
            traceback.print_exc()

    def _evict_old(self) -> None:
        """Drop the oldest finished jobs once MAX_RETAINED_JOBS is exceeded."""
        with self._lock:
            finished = [j for j in self._jobs.values()
                        if j.status in (STATUS_DONE, STATUS_FAILED)]
            excess = len(finished) - MAX_RETAINED_JOBS
            victims = finished[:excess] if excess > 0 else []
            for job in victims:
                self._jobs.pop(job.id, None)
        for job in victims:
            _rmtree(job)


def _rmtree(job):
    """
    Remove both of a job's folders — the upload and the analysis output.

    Retried briefly, and reported if it still fails.  On Windows an open handle
    makes removal fail outright rather than deferring it, and a response still
    unwinding can hold the annotated video for a moment after the request ends.
    `ignore_errors=True` alone would swallow that and leave debris with nothing
    in the log to explain it.
    """
    for path in (job.upload_dir, job.directory):
        for attempt in range(4):
            try:
                shutil.rmtree(path)
                break
            except FileNotFoundError:
                break
            except OSError as exc:
                if attempt == 3:
                    # flush: this runs on a worker thread whose stdout is block
                    # buffered when the server's output is redirected to a file,
                    # which would otherwise hold the warning back indefinitely.
                    print(f"[JOBS] WARNING: could not remove '{path}': {exc}",
                          flush=True)
                else:
                    time.sleep(0.25)


def _now():
    return time.time()


def _unlink(path):
    try:
        os.remove(path)
    except OSError:
        pass
