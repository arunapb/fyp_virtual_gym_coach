"""
webapp/video_out.py — write the annotated frames to a file a browser can play.

Why this needs care.  OpenCV's default `mp4v` fourcc produces MPEG-4 Part 2,
which no current browser decodes in a <video> element — the file downloads but
plays as a black rectangle.  Which encoders are actually available depends on
how OpenCV was built and on the host: on this machine the bundled FFmpeg fails
to load libopenh264 and OpenCV silently falls back to the Windows Media
Foundation backend, which does encode H.264 correctly.  `isOpened()` alone is
not proof of that, because it can report success on a writer that then produces
nothing usable.

So each candidate codec is PROBED: open a writer, push a real frame, close it,
and require a non-trivial file.  The first codec that survives is used.
"""

import os

import cv2
import numpy as np

from config import VIDEO_CODECS, VIDEO_MAX_LONG_SIDE

# A writer that opened but produced only a container header yields a few hundred
# bytes.  One encoded frame of real video comfortably exceeds this.
_MIN_PROBE_BYTES = 512


def even(n: int) -> int:
    """
    Round a dimension up to the nearest even number.

    H.264 with 4:2:0 chroma cannot represent an odd width or height, and the
    encoder simply fails on one.  Only the OUTPUT image is affected — the
    analysis already ran on the untouched frame at its true size, so no
    measurement, normalisation or threshold changes.
    """
    return n + (n % 2)


def output_size(width: int, height: int, max_long_side: int = VIDEO_MAX_LONG_SIDE):
    """
    Encoding size for a source frame: aspect preserved, long side capped, both
    dimensions even.  Frames already within the cap are returned unchanged.
    """
    longest = max(width, height)
    if longest <= max_long_side:
        return even(width), even(height)
    scale = max_long_side / longest
    return even(max(int(round(width * scale)), 2)), even(max(int(round(height * scale)), 2))


def pad_to_even(frame):
    """Pad the bottom/right edge by at most one pixel so the frame encodes."""
    h, w = frame.shape[:2]
    h2, w2 = even(h), even(w)
    if (h2, w2) == (h, w):
        return frame
    padded = np.zeros((h2, w2, 3), dtype=frame.dtype)
    padded[:h, :w] = frame
    return padded


class AnnotatedVideoWriter:
    """
    Codec-probing video writer that also caps the output resolution.

    `path_stem` is the path without an extension; the extension is chosen with
    the codec (`.mp4` for H.264, `.webm` for VP8), so `self.path` after
    construction is the file that was actually written.
    """

    def __init__(self, path_stem: str, fps: float, size):
        self.source_width, self.source_height = size
        self.width, self.height = output_size(*size)
        self.scaled = (self.width, self.height) != (even(size[0]), even(size[1]))
        self.fps = fps if fps and fps > 0 else 25.0
        self.path = None
        self.codec = None
        self.codec_label = None
        self._writer = None
        self._open(path_stem)

    def _open(self, path_stem):
        probe = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        # A mid-grey gradient rather than flat black: a constant frame can encode
        # to almost nothing, which would look like the failure this probe tests.
        probe[:, :, 1] = np.linspace(0, 255, self.width, dtype=np.uint8)

        for fourcc, ext, label in VIDEO_CODECS:
            candidate = f"{path_stem}{ext}"
            writer = cv2.VideoWriter(candidate, cv2.VideoWriter_fourcc(*fourcc),
                                     self.fps, (self.width, self.height))
            if not writer.isOpened():
                writer.release()
                _unlink(candidate)
                continue
            writer.write(probe)
            writer.release()
            if os.path.exists(candidate) and os.path.getsize(candidate) > _MIN_PROBE_BYTES:
                _unlink(candidate)
                self._writer = cv2.VideoWriter(
                    candidate, cv2.VideoWriter_fourcc(*fourcc),
                    self.fps, (self.width, self.height))
                if self._writer.isOpened():
                    self.path, self.codec, self.codec_label = candidate, fourcc, label
                    return
                self._writer.release()
                self._writer = None
            _unlink(candidate)

        raise RuntimeError(
            "No usable video encoder found. OpenCV could not open a writer for "
            + ", ".join(c[0] for c in VIDEO_CODECS)
            + ". Install an OpenCV build with FFmpeg encoding support.")

    def write(self, frame):
        """
        Encode one annotated frame, downscaling first if the source exceeded the
        resolution cap.  INTER_AREA is the correct filter for shrinking — it
        averages the source pixels rather than sampling them, so the thin
        skeleton lines and the badge text stay legible instead of aliasing.
        """
        if self.scaled:
            frame = cv2.resize(frame, (self.width, self.height),
                               interpolation=cv2.INTER_AREA)
        self._writer.write(pad_to_even(frame))

    def release(self):
        if self._writer is not None:
            self._writer.release()
            self._writer = None


def _unlink(path):
    try:
        os.remove(path)
    except OSError:
        pass
