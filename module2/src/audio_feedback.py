"""
audio_feedback.py — Module 2 audio coaching layer (LIVE mode only).

This module turns per-frame analysis state (zones, rep phase, rep quality,
landmark visibility) into spoken coaching cues, played back as short
pre-rendered MP3 clips.  It is a READ-ONLY consumer: it never modifies zone
classification, the rep counter, or any derivation script.

Two public pieces
─────────────────
  AudioPlayer        — decodes all clips to NumPy once at startup and plays them
                       non-blocking via sounddevice (one clip at a time).
  FeedbackController — per-frame state machine.  update(...) is called once per
                       frame and returns at most ONE clip filename (or None),
                       playing it through the AudioPlayer.

Master priority (highest → lowest), so two clips never overlap:
  1. SYSTEM STATUS  — no pose / out of frame / key joints low-confidence.
  2. CORRECTIVE     — sustained red zones, phase-aware.
  3. ENCOURAGEMENT  — only when nothing needs correcting and form is good.

Exercise-aware
──────────────
The corrective cue mapping (which zone channels exist, their phases, and the
clip(s) for each) is supplied per exercise via configure().  System-status and
encouragement cues are generic and shared across exercises.  Signal gating:
  NULL     → silent
  REST     → system-status only
  EXERCISE → full pipeline

Backward compatibility: a FeedbackController() with no configure() call defaults
to the squat cue mapping (from config) and the EXERCISE gate, so the legacy call
path behaves exactly as before this refactor.

All dwell windows, cooldowns, priority order, and clip lists are named constants
in config.py.
"""

import os
import time

from config import (
    AUDIO_CLIPS_DIR, EXPECTED_AUDIO_CLIPS, AUDIO_MIN_CUE_GAP_SEC,
    CORRECTIVE_DWELL_SEC, CORRECTIVE_COOLDOWN_SEC, AUDIO_CORRECTIVE_PRIORITY,
    FPS_REFERENCE, frames_at,
    SYSTEM_STATUS_BAD_DWELL_SEC, SYSTEM_STATUS_GOOD_DWELL_SEC,
    SYSTEM_STATUS_COOLDOWN_SEC,
    ENCOURAGE_MILESTONE_INTERVAL, ENCOURAGE_STREAK_LENGTH, ENCOURAGE_COOLDOWN_SEC,
    AUDIO_SYSTEM_NO_POSE_CLIPS, AUDIO_SYSTEM_FACE_CAMERA,
    AUDIO_CORRECT_KNEES_OUT, AUDIO_CORRECT_GO_DEEPER, AUDIO_CORRECT_NOT_SO_DEEP,
    AUDIO_CORRECT_TRUNK_CLIPS,
    AUDIO_ENCOURAGE_MILESTONE_CLIPS, AUDIO_ENCOURAGE_STREAK_CLIPS,
    LANDMARK_MIN_VISIBILITY,
    IDX_HIP_L, IDX_HIP_R, IDX_KNEE_L, IDX_KNEE_R,
)
from src.pose_utils import landmarks_reliable

# Audio backends are imported defensively so that a missing system library
# degrades to "no sound" with a warning rather than crashing the demo.
try:
    import soundfile as sf
    import sounddevice as sd
    _AUDIO_LIBS_OK = True
    _AUDIO_IMPORT_ERR = None
except Exception as exc:                      # pragma: no cover - environment dependent
    _AUDIO_LIBS_OK = False
    _AUDIO_IMPORT_ERR = exc


# Key lower-body landmarks used to decide SYSTEM-STATUS case (b): the user is in
# frame but turned so that hips/knees are low-confidence.  Same set the rep
# counter gates on, so audio and rep logic agree on "reliable".
_LOWER_BODY_IDX = [IDX_HIP_L, IDX_HIP_R, IDX_KNEE_L, IDX_KNEE_R]

# Signal-gate labels (mirror session.SignalState values; kept as bare strings to
# avoid a circular import).
GATE_NULL = "NULL"
GATE_REST = "REST"
GATE_EXERCISE = "EXERCISE"


class NullAudioController:
    """
    No-op audio controller used when AUDIO_FEEDBACK_ENABLED is False.  Satisfies
    the interface the SessionController calls (configure / reset / update /
    shutdown) without loading clips or producing sound.
    """

    def configure(self, cue_mapping=None, signal_state=GATE_EXERCISE):
        pass

    def reset(self):
        pass

    def update(self, *args, **kwargs):
        return None

    def shutdown(self):
        pass


def default_corrective_channels():
    """
    The squat corrective-cue channels, in priority order, built from config.

    This is BOTH the FeedbackController's backward-compatible default AND the
    source SquatExercise.audio_cue_mapping reuses — a single definition, so the
    legacy path and the refactored path make identical decisions.

    Each channel: {name, zone_keys, phases, clips}.  A channel is "red" this
    frame if ANY of its zone_keys reads "red".  Multi-clip channels rotate.
    """
    spec = {
        "valgus": (["valgus_l", "valgus_r"], {"DESCENDING", "BOTTOM"},
                   [AUDIO_CORRECT_KNEES_OUT]),
        # STANDING is in both depth channels' phase sets because both cues are
        # armed at REP COMPLETION, by which point the phase has returned to
        # STANDING — and that is also the natural moment to be told what to do
        # differently on the next rep.
        #
        # For "go deeper" it is the only workable gate: a squat that stops short
        # never enters BOTTOM, which is precisely the fault, so gating on BOTTOM
        # alone made the cue unreachable for the one case it exists to serve.
        "depth":  (["depth"], {"BOTTOM", "STANDING"},
                   [AUDIO_CORRECT_GO_DEEPER]),
        # Too deep — reachable live at BOTTOM as well, since an over-deep
        # position is knowable while it is happening, unlike a short one.
        "depth_excess": (["depth_excess"], {"BOTTOM", "STANDING"},
                         [AUDIO_CORRECT_NOT_SO_DEEP]),
        "trunk":  (["trunk"], {"DESCENDING", "BOTTOM", "ASCENDING"},
                   list(AUDIO_CORRECT_TRUNK_CLIPS)),
    }
    return [
        {"name": name, "zone_keys": spec[name][0],
         "phases": spec[name][1], "clips": spec[name][2]}
        for name in AUDIO_CORRECTIVE_PRIORITY
    ]


# =============================================================================
# PLAYBACK
# =============================================================================

# Extensions the recursive clip index will pick up.  Restricting by extension
# keeps OS cruft that legitimately appears in several folders (Thumbs.db,
# desktop.ini, .gitkeep) from being misread as a duplicate clip.
_AUDIO_EXTS = {".mp3", ".wav", ".ogg", ".flac", ".m4a", ".aiff"}


class DuplicateClipError(RuntimeError):
    """
    Raised at startup when one clip basename exists in more than one directory.

    The clip cache is keyed by BASENAME so that a cue name like
    "correct_elbows_pinned.mp3" resolves wherever the file lives.  That makes a
    duplicate genuinely ambiguous: one copy would silently shadow the other and
    the wrong (or a stale) clip would play, with nothing in the log to explain
    it.  Failing loudly at startup is far cheaper to diagnose than that.
    """


class AudioPlayer:
    """
    Decodes every expected clip to a NumPy array once, then plays them
    non-blocking.  sounddevice.play() internally stops any prior playback, so
    starting a new clip always cuts off the previous one — a hard guarantee
    that only one clip is ever audible at a time.

    Clips are discovered RECURSIVELY under `clips_dir` and cached by basename,
    so exercises may keep their clips in their own subfolder (for example
    "assets/audio/bicep curl/") without the cue mapping needing to know where a
    file physically sits.  Discovery is still fully EAGER: every clip is decoded
    during __init__, before any exercise exists, so playback never allocates or
    blocks mid-session.
    """

    @staticmethod
    def _index_clips(clips_dir):
        """
        Map basename -> full path for every audio file under `clips_dir`.

        Returns (index, directories_scanned).  Directory names containing
        spaces work naturally: os.walk yields real path components and they are
        recombined with os.path.join, never shell-quoted or split on whitespace.

        Raises DuplicateClipError if any basename occurs twice (see that class).
        """
        index, dupes, n_dirs = {}, {}, 0
        for root, _dirs, files in os.walk(clips_dir):
            n_dirs += 1
            for fn in sorted(files):
                if os.path.splitext(fn)[1].lower() not in _AUDIO_EXTS:
                    continue
                path = os.path.join(root, fn)
                if fn in index:
                    dupes.setdefault(fn, [index[fn]]).append(path)
                else:
                    index[fn] = path

        if dupes:
            for name, paths in sorted(dupes.items()):
                print(f"[AUDIO] ERROR: duplicate clip basename '{name}' found in "
                      f"{len(paths)} locations:")
                for p in paths:
                    print(f"[AUDIO]            {p}")
            raise DuplicateClipError(
                f"Duplicate audio clip basename(s) under '{clips_dir}': "
                f"{', '.join(sorted(dupes))}. The clip cache is keyed by "
                f"basename, so one copy would silently shadow the other — "
                f"remove or rename one of each pair.")
        return index, n_dirs

    def __init__(self, clips_dir=AUDIO_CLIPS_DIR, expected=EXPECTED_AUDIO_CLIPS):
        self.enabled = _AUDIO_LIBS_OK
        self._cache = {}     # filename -> (ndarray, samplerate)

        if not self.enabled:
            print(f"[AUDIO] WARNING: audio libraries unavailable "
                  f"({_AUDIO_IMPORT_ERR}). Audio feedback disabled.")
            return

        if not os.path.isdir(clips_dir):
            print(f"[AUDIO] WARNING: clips directory '{clips_dir}' does not exist.")
        index, n_dirs = self._index_clips(clips_dir)

        loaded, missing, failed = 0, [], []
        for name in expected:
            path = index.get(name)
            if path is None:
                missing.append(name)
                continue
            try:
                data, sr = sf.read(path, dtype="float32")
                self._cache[name] = (data, sr)
                loaded += 1
            except Exception as exc:                  # pragma: no cover
                failed.append(f"{name} ({exc})")

        print(f"[AUDIO] Loaded {loaded}/{len(expected)} clips from '{clips_dir}' "
              f"({n_dirs} director{'y' if n_dirs == 1 else 'ies'}).")
        for name in missing:
            print(f"[AUDIO] WARNING: missing clip '{name}' — its cue will be skipped.")
        for item in failed:
            print(f"[AUDIO] WARNING: failed to decode {item} — cue skipped.")

    def play(self, filename):
        """Start non-blocking playback of `filename`.  No-op if unavailable."""
        if not self.enabled:
            return
        entry = self._cache.get(filename)
        if entry is None:
            return                       # missing/failed clip — already warned
        data, sr = entry
        try:
            sd.stop()                    # cut any currently-playing clip
            sd.play(data, sr)
        except Exception as exc:                      # pragma: no cover
            print(f"[AUDIO] WARNING: playback failed for '{filename}': {exc}")

    def stop(self):
        """Stop any current playback (called on shutdown)."""
        if self.enabled:
            try:
                sd.stop()
            except Exception:                         # pragma: no cover
                pass


# =============================================================================
# FEEDBACK CONTROLLER
# =============================================================================

class FeedbackController:
    """
    Per-frame coaching decision maker.

    Call update(...) exactly once per frame (regardless of whether a pose was
    detected).  It updates all sustained-condition / hysteresis / edge state,
    then resolves the master priority and returns the single clip filename it
    played this frame (or None).

    Exercise wiring: configure(cue_mapping, signal_state) sets the corrective
    channels and the signal gate, and resets all per-session state.  Defaults
    reproduce the pre-refactor squat behaviour.
    """

    def __init__(self, player=None, fps=FPS_REFERENCE):
        self.player = player if player is not None else AudioPlayer()
        self._channels = default_corrective_channels()
        self._signal_state = GATE_EXERCISE
        # Red-frame dwell derived from the SOURCE frame rate, so a fault has to
        # persist for the same REAL time before it is cued regardless of how the
        # clip was recorded (see config.py's TIMEBASE).
        self._dwell_frames = frames_at(CORRECTIVE_DWELL_SEC, fps)
        self.reset()

    # -------------------------------------------------------------------------
    # Configuration / lifecycle
    # -------------------------------------------------------------------------

    def configure(self, cue_mapping=None, signal_state=GATE_EXERCISE):
        """
        Set the active exercise's corrective channels (or keep current if None)
        and the signal gate, then fully reset per-session state.  Called by the
        SessionController on every transition into Null / Rest / an exercise.
        """
        if cue_mapping is not None:
            self._channels = list(cue_mapping.channels)
        self._signal_state = signal_state
        self.reset()

    def reset(self):
        """Reinitialise EVERY timer/counter/pointer to a fresh slate."""
        # System-status hysteresis
        self._bad_since = None
        self._good_since = None
        self._system_active = False
        self._last_system_play = float("-inf")
        # Corrective — per-channel dwell counters + cooldowns
        names = [ch["name"] for ch in self._channels]
        self._red_frames = {n: 0 for n in names}
        self._last_corrective_play = {n: float("-inf") for n in names}
        # Encouragement edge-trigger + cooldown
        self._prev_rep_count = 0
        self._last_encourage_play = float("-inf")
        # Global quiet gap
        self._last_cue_play = float("-inf")
        # Clip-rotation pointers (system no-pose, encouragement, + each channel)
        self._rot = {"no_pose": 0, "milestone": 0, "streak": 0}
        for n in names:
            self._rot[n] = 0

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def update(self, pose_detected, landmarks, zones, rep_phase,
               rep_count, rep_history, now=None):
        """
        Advance the controller by one frame and play at most one clip.

        Gating by the configured signal state:
          NULL     → returns None immediately (silent).
          REST     → only system-status may fire.
          EXERCISE → full pipeline (system > corrective > encouragement).

        Returns the clip filename played this frame, or None.
        """
        if self._signal_state == GATE_NULL:
            return None
        if now is None:
            now = time.monotonic()

        # ── 1. Condition tracking (every frame) ──────────────────────────────
        self._update_system_hysteresis(pose_detected, landmarks, now)

        if self._signal_state == GATE_EXERCISE:
            # Track corrective dwell + rep edge every frame so that a
            # higher-priority preemption can't desync these counters.
            self._update_corrective_dwell(zones)
            new_rep_completed = rep_count > self._prev_rep_count
            self._prev_rep_count = rep_count
        else:
            new_rep_completed = False

        # ── 2. Master priority resolution ────────────────────────────────────
        # SYSTEM STATUS (highest) — active for REST and EXERCISE; bypasses the
        # global quiet gap so a safety cue can always preempt.
        clip = self._resolve_system(pose_detected, now)
        if clip is not None:
            return self._fire(clip, "system", now, "_last_system_play")

        if self._signal_state == GATE_REST:
            return None        # Rest: system-status only

        # Below system status, honour the global quiet gap so cues never stack.
        if now - self._last_cue_play < AUDIO_MIN_CUE_GAP_SEC:
            return None

        # CORRECTIVE
        channel = self._corrective_winner(rep_phase, now)
        if channel is not None:
            clip = self._corrective_clip(channel)
            self._last_corrective_play[channel["name"]] = now
            return self._fire(clip, f"corrective:{channel['name']}", now)

        # ENCOURAGEMENT (lowest)
        if new_rep_completed:
            clip, label = self._resolve_encouragement(rep_count, rep_history, now)
            if clip is not None:
                self._last_encourage_play = now
                return self._fire(clip, label, now)

        return None

    def shutdown(self):
        """Stop playback on program exit."""
        self.player.stop()

    # -------------------------------------------------------------------------
    # Condition tracking (runs every frame)
    # -------------------------------------------------------------------------

    def _update_system_hysteresis(self, pose_detected, landmarks, now):
        """
        Two-threshold hysteresis on detection quality.

        detection is BAD when no pose is detected OR a pose is detected but the
        key lower-body landmarks are below the visibility threshold.  The alert
        becomes active only after BAD holds for SYSTEM_STATUS_BAD_DWELL_SEC, and
        clears only after GOOD holds for SYSTEM_STATUS_GOOD_DWELL_SEC.
        """
        bad = (not pose_detected) or (
            not landmarks_reliable(landmarks, _LOWER_BODY_IDX,
                                   LANDMARK_MIN_VISIBILITY))

        if bad:
            self._good_since = None
            if self._bad_since is None:
                self._bad_since = now
            if (not self._system_active
                    and now - self._bad_since >= SYSTEM_STATUS_BAD_DWELL_SEC):
                self._system_active = True
        else:
            self._bad_since = None
            if self._good_since is None:
                self._good_since = now
            if (self._system_active
                    and now - self._good_since >= SYSTEM_STATUS_GOOD_DWELL_SEC):
                self._system_active = False

    def _update_corrective_dwell(self, zones):
        """Count consecutive red frames per corrective channel."""
        if not zones:
            for ch in self._channels:
                self._red_frames[ch["name"]] = 0
            return
        for ch in self._channels:
            is_red = any(zones.get(k) == "red" for k in ch["zone_keys"])
            name = ch["name"]
            self._red_frames[name] = self._red_frames[name] + 1 if is_red else 0

    # -------------------------------------------------------------------------
    # Category resolvers
    # -------------------------------------------------------------------------

    def _resolve_system(self, pose_detected, now):
        """
        Return the system clip to play, or None.

        Fires only while the alert is active (bad dwell satisfied), the current
        frame is still bad, and the long cooldown has elapsed.  no-pose →
        alternate move-in/step-back; pose present but low-confidence → face the
        camera.
        """
        if not self._system_active:
            return None
        if self._bad_since is None:        # currently good — don't nag
            return None
        if now - self._last_system_play < SYSTEM_STATUS_COOLDOWN_SEC:
            return None
        if not pose_detected:
            return self._next("no_pose", AUDIO_SYSTEM_NO_POSE_CLIPS)
        return AUDIO_SYSTEM_FACE_CAMERA

    def _corrective_winner(self, rep_phase, now):
        """
        Return the single corrective channel allowed to fire this frame, or None.

        A channel qualifies when its red streak has met CORRECTIVE_DWELL_SEC
        worth of frames, the current rep phase permits it, and its per-cue
        cooldown has elapsed.  Channels are evaluated in configured priority order.
        """
        for ch in self._channels:
            name = ch["name"]
            if self._red_frames[name] < self._dwell_frames:
                continue
            if rep_phase not in ch["phases"]:
                continue
            if now - self._last_corrective_play[name] < CORRECTIVE_COOLDOWN_SEC:
                continue
            return ch
        return None

    def _corrective_clip(self, channel):
        clips = channel["clips"]
        if len(clips) == 1:
            return clips[0]
        return self._next(channel["name"], clips)

    def _resolve_encouragement(self, rep_count, rep_history, now):
        """
        Decide an encouragement clip on a rep-completion frame, or (None, None).

        Conditional on form quality — never praises red.  Cooldown-gated.  If
        both triggers qualify on the same rep, the clean-streak cue wins (it is
        the more earned, specific praise).
        """
        if now - self._last_encourage_play < ENCOURAGE_COOLDOWN_SEC:
            return None, None

        streak = self._tail_nonred_streak(rep_history)
        if streak > 0 and streak % ENCOURAGE_STREAK_LENGTH == 0:
            return self._next("streak", AUDIO_ENCOURAGE_STREAK_CLIPS), "encourage:streak"

        if rep_count > 0 and rep_count % ENCOURAGE_MILESTONE_INTERVAL == 0:
            recent = rep_history[-ENCOURAGE_MILESTONE_INTERVAL:]
            if recent and all(r.get("rep_quality") != "red" for r in recent):
                return (self._next("milestone", AUDIO_ENCOURAGE_MILESTONE_CLIPS),
                        "encourage:milestone")

        return None, None

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _tail_nonred_streak(rep_history):
        """Count consecutive non-red reps at the end of rep_history."""
        n = 0
        for rep in reversed(rep_history):
            if rep.get("rep_quality") == "red":
                break
            n += 1
        return n

    def _next(self, key, clips):
        """Return the next clip in a rotation list and advance its pointer."""
        clip = clips[self._rot[key] % len(clips)]
        self._rot[key] += 1
        return clip

    def _fire(self, clip, label, now, cooldown_attr=None):
        """Play `clip`, stamp cooldowns, log, and return it."""
        if cooldown_attr is not None:
            setattr(self, cooldown_attr, now)
        self._last_cue_play = now          # global quiet-gap timestamp
        self.player.play(clip)
        print(f"[AUDIO] play {clip}  ({label})")
        return clip
