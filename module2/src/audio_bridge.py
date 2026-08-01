"""
webapp/audio_bridge.py — carries the audio-coaching layer across to the browser.

The desktop app plays a clip through the speakers the instant the controller
decides on it.  That cannot work here: analysis runs at roughly 11 fps on this
class of machine (docs/E7_PERFORMANCE.md) while the video it describes may be
30 fps, so "now" during processing is not "now" during playback.

So the decision layer is kept intact and only the OUTPUT is redirected:

  * `CueRecorder` stands in for `AudioPlayer` inside the real
    `FeedbackController`.  Every dwell window, cooldown, hysteresis rule and
    priority resolution runs unchanged — the controller simply hands its chosen
    clip to a recorder instead of a sound device.
  * The recorded (frame, clip) pairs are sent to the browser, which plays the
    genuine MP3 files at the matching video timestamps.

Result: identical cue decisions to a live session, heard at the right moment.
"""

import os

from src.audio_feedback import AudioPlayer
from config import AUDIO_CLIPS_DIR, EXPECTED_AUDIO_CLIPS

_clip_index_cache = None


class CueRecorder:
    """
    A drop-in `AudioPlayer` for offline analysis: satisfies the `play`/`stop`
    interface `FeedbackController` calls, produces no sound, and loads no audio
    device or codec (so the server runs fine on a headless box with no speakers).

    The controller already returns the clip it chose each frame, so this class
    exists to make that call harmless rather than to be the source of truth.
    """

    def __init__(self):
        self.played = []

    def play(self, filename):
        self.played.append(filename)

    def stop(self):
        pass


def clip_index() -> dict:
    """
    Map clip basename -> absolute path for everything under `assets/audio`.

    Reuses `AudioPlayer._index_clips` rather than re-walking the tree here, so
    the web server resolves a cue name to exactly the same file the desktop app
    would play — including clips that live in a per-exercise subfolder such as
    "assets/audio/bicep curl/" — and inherits its duplicate-basename check
    instead of silently serving whichever copy happened to be found first.

    An EMPTY result is not cached.  The clips are assets a user drops into
    `assets/audio/` by hand, quite possibly while the server is already running;
    caching "there are none" would then serve 404s for the rest of the process
    even after the files appeared, and the only cure would be a restart nobody
    would think to try.  Re-walking a directory that is still empty costs
    nothing, and the walk stops recurring the moment the first clip lands.
    """
    global _clip_index_cache
    if not _clip_index_cache:
        index, _n_dirs = AudioPlayer._index_clips(AUDIO_CLIPS_DIR)
        _clip_index_cache = {name: os.path.abspath(path)
                             for name, path in index.items()}
    return _clip_index_cache


def resolve_clip(basename: str) -> str | None:
    """Absolute path of a clip, or None if it is not present in the index."""
    if not basename or os.path.sep in basename or "/" in basename or ".." in basename:
        return None                     # never let a request escape the clip dir
    return clip_index().get(basename)


def missing_clips() -> list:
    """Which of `EXPECTED_AUDIO_CLIPS` are not on disk, in config order."""
    index = clip_index()
    return [name for name in EXPECTED_AUDIO_CLIPS if name not in index]


def report_missing_clips() -> list:
    """
    Print what the coaching audio resolved to, and what it did not.

    The web server never constructs an `AudioPlayer` — cue decisions go to a
    `CueRecorder` and the browser does the playing — so it never sees the
    startup inventory the desktop app prints.  Without this, a missing clip
    surfaces only as a 404 in the browser's developer console, at the moment the
    cue fires, which is a poor way to learn that an asset folder is empty.

    Returns the missing names so a caller can act on them; nothing here fails,
    because a silent cue is a degraded session, not a broken one.
    """
    missing = missing_clips()
    found = len(EXPECTED_AUDIO_CLIPS) - len(missing)
    print(f"[AUDIO] {found}/{len(EXPECTED_AUDIO_CLIPS)} coaching clips found "
          f"in '{AUDIO_CLIPS_DIR}'.", flush=True)
    if missing:
        print(f"[AUDIO] WARNING: {len(missing)} clip(s) missing — those cues "
              f"will be shown on screen but play no sound:", flush=True)
        for name in missing:
            print(f"[AUDIO]            {name}", flush=True)
        print(f"[AUDIO] See {os.path.join(AUDIO_CLIPS_DIR, 'README.md')} for "
              f"where each file belongs.", flush=True)
    return missing


def cue_caption(clip: str) -> str:
    """
    Human-readable caption for a clip filename, e.g.
    "correct_knees_out.mp3" -> "Knees out".

    Derived from the filename rather than a hand-written table so a clip added
    to config.py shows up in the web UI with no further wiring.  The leading
    category word ("correct", "encourage", "system", "press") is dropped because
    the UI already shows the category as a separate badge.
    """
    stem = os.path.splitext(os.path.basename(clip or ""))[0]
    parts = stem.split("_")
    if len(parts) > 1 and parts[0] in ("correct", "encourage", "system", "press"):
        parts = parts[1:]
    text = " ".join(parts).strip()
    return text[:1].upper() + text[1:] if text else stem


def cue_category(clip: str) -> str:
    """
    Which of the controller's three tiers a clip belongs to: "system",
    "corrective" or "encouragement".  Used only for colour-coding the cue log.
    """
    stem = os.path.basename(clip or "")
    if stem.startswith("system_"):
        return "system"
    if stem.startswith("encourage_"):
        return "encouragement"
    return "corrective"
