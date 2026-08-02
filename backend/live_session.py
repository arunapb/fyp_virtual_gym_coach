"""
backend/live_session.py — runs backend/merged_analysis.py's generator behind
the same producer-thread / bounded-queue / backpressure machinery Module 2's
own `LiveAnalysisSession` (module2/src/live_session.py) already implements,
reused via subclassing rather than duplicated.

`pump`, `_reap`, `_send` and the wire format are inherited unchanged. Only
`_produce` is overridden, because it must call `merged_analysis.analyse_stream_
merged()` (Module 1 driving Module 2 frame-by-frame, see that file) instead of
Module 2's own file-only `analyse_stream()`. The override's body mirrors the
parent's `_produce` line for line except for that one call — see
module2/src/live_session.py to compare.
"""

import traceback

from src.analysis import EVENT_DONE, AnalysisCancelled  # module2's, unmodified
from src.live_session import _END, LiveAnalysisSession  # module2's, unmodified

from backend.merged_analysis import analyse_stream_merged


class _NoopSignal:
    """
    Placeholder for `LiveAnalysisSession.signal`, which our overridden
    `_produce` never reads (Module 1 drives the signal directly inside
    `analyse_stream_merged`). Only exists so `receive()` (inherited,
    unmodified) doesn't crash if a client ever sends a manual
    `{"type":"signal",...}` control message — the merged frontend doesn't
    expose that control, but a stray message should be a no-op, not a crash.
    """

    def set(self, _label):
        return False


class MergedAnalysisSession(LiveAnalysisSession):
    def __init__(self, store, job, video_path, pool, max_frames):
        super().__init__(store, job, video_path, signal_source=_NoopSignal())
        self._pool = pool
        self._max_frames = max_frames

    def _produce(self):
        try:
            for event in analyse_stream_merged(
                self.video_path, self._pool, self.job.directory,
                stream_frames=True, progress=self._progress,
                should_cancel=self.cancel.is_set, max_frames=self._max_frames,
            ):
                if event["type"] == EVENT_DONE:
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
