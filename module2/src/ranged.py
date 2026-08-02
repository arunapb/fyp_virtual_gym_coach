"""
webapp/ranged.py — HTTP range support for the annotated video.

Browsers seek in a <video> by issuing `Range: bytes=…` and expect `206 Partial
Content`.  A server that always answers `200` with the whole file leaves the
scrubber working only for the part already buffered — which would break the one
interaction this app depends on most, jumping to the frame where a cue fired.

Starlette's FileResponse has handled ranges only in recent versions, so this
implements the header parsing directly rather than depending on the version that
happens to be installed.  Single-range requests only, which is all any browser
sends for media playback.
"""

import os
import re

from starlette.responses import Response, StreamingResponse

_RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)")
_CHUNK = 1024 * 256


def ranged_file_response(path: str, request, media_type: str,
                         filename: str | None = None) -> Response:
    """
    Serve `path`, honouring a single `Range` header when present.

    BOTH the full-file and the partial reply stream through `_iter_range`.  The
    obvious alternative for the full-file case, Starlette's `FileResponse`, LEAKS
    THE FILE HANDLE when the client aborts mid-download — and a browser aborts
    that first open-ended request as a matter of routine, the moment it has
    enough of the video to switch to explicit byte ranges.  On Windows an open
    handle blocks deletion, so deleting a job silently left its annotated video
    behind for the lifetime of the server.  Owning the handle here means the
    generator's `finally` closes it on disconnect.
    """
    file_size = os.path.getsize(path)
    headers = {"accept-ranges": "bytes"}
    if filename:
        headers["content-disposition"] = f'attachment; filename="{filename}"'

    parsed = _parse_range(request.headers.get("range"), file_size)
    if parsed == "unsatisfiable":
        return Response(status_code=416,
                        headers={"content-range": f"bytes */{file_size}"})

    if parsed is None:                       # whole file, 200
        start, end, status = 0, max(file_size - 1, 0), 200
    else:                                    # partial, 206
        start, end = parsed
        status = 206
        headers["content-range"] = f"bytes {start}-{end}/{file_size}"

    headers["content-length"] = str(end - start + 1 if file_size else 0)
    return StreamingResponse(_iter_range(path, start, end), status_code=status,
                             media_type=media_type, headers=headers)


def _parse_range(header, file_size):
    """(start, end) inclusive, None for "no range", or "unsatisfiable"."""
    if not header:
        return None
    match = _RANGE_RE.fullmatch(header.strip())
    if not match:
        return None
    raw_start, raw_end = match.group(1), match.group(2)
    if raw_start == "" and raw_end == "":
        return None
    if raw_start == "":                       # suffix form: last N bytes
        length = int(raw_end)
        if length <= 0:
            return "unsatisfiable"
        start, end = max(file_size - length, 0), file_size - 1
    else:
        start = int(raw_start)
        end = int(raw_end) if raw_end else file_size - 1
    end = min(end, file_size - 1)
    if start > end or start >= file_size:
        return "unsatisfiable"
    return start, end


def _iter_range(path, start, end):
    """
    Yield `path` from `start` to `end` inclusive, NEVER holding the file open
    across a `yield`.

    The obvious shape — open once, seek, then yield chunks in a loop — strands
    the handle when the client disconnects: the generator is suspended at a
    `yield` that no one will resume, and neither `with` nor `try/finally` runs
    until it is closed or collected, which a cancelled ASGI task does not
    reliably do.  On Windows that open handle then blocks deleting the file, so
    a deleted job left its annotated video on disk for the life of the server.

    Reopening per chunk removes the failure mode outright rather than trying to
    time the cleanup: between yields there is simply no handle to strand.  At
    256 KB a chunk that is a handful of opens for a typical range request, which
    is nothing next to the disk read itself.
    """
    pos, remaining = start, end - start + 1
    while remaining > 0:
        with open(path, "rb") as fh:
            fh.seek(pos)
            chunk = fh.read(min(_CHUNK, remaining))
        if not chunk:
            break
        pos += len(chunk)
        remaining -= len(chunk)
        yield chunk
