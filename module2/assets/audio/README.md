# Coaching audio clips

Drop the 23 clips listed below into these folders. Both front ends read the
same files: `desktop_app.py` plays them through the speakers, and the web app
serves them to the browser at `/api/audio/<basename>` so the cue is heard at the
frame the controller decided on it.

**A missing clip is not fatal.** `AudioPlayer` logs a warning at startup and
`play()` no-ops on a name it has not cached, so the cue still appears in the UI
and the analysis is unaffected — it is simply silent. The web app prints the
same warning when it starts, so you find out before you run an analysis rather
than from a 404 in the browser console.

## Where each clip goes

The loader walks this tree **recursively** and keys clips by **basename**, so
the folders are organisation only — a file works wherever it sits under
`assets/audio/`. Keep the layout below anyway: it is what `config.py` describes,
and it makes it obvious which exercise a corrective belongs to.

Because the index is keyed by basename, **no two clips may share a filename**,
even in different folders. A duplicate raises `DuplicateClipError` at startup
rather than letting one copy silently shadow the other.

### `assets/audio/` — system, squat, and encouragement (13)

Shared across every exercise except the squat correctives, which have no
subfolder of their own because the squat was the first exercise built.

| File | Fires when |
|---|---|
| `system_move_into_frame.mp3` | the user is out of frame |
| `system_step_back.mp3` | the user is too close to fit |
| `system_face_camera.mp3` | the camera view is not frontal |
| `correct_knees_out.mp3` | squat — knee valgus in the red zone |
| `correct_go_deeper.mp3` | squat — depth short of threshold |
| `correct_slow_down.mp3` | squat — descent too fast |
| `correct_chest_up.mp3` | squat — trunk deviation red |
| `correct_stay_balanced.mp3` | squat — left/right asymmetry red |
| `encourage_keep_going.mp3` | generic encouragement |
| `encourage_doing_well.mp3` | generic encouragement |
| `encourage_halfway.mp3` | generic encouragement |
| `encourage_strong_finish.mp3` | generic encouragement |
| `encourage_excellent_form.mp3` | a streak of clean repetitions |

The encouragement pool must stay **exercise-neutral** — every clip in it can
play during any exercise. (`encourage_nice_depth.mp3` was removed from the pool
on 2026-07-27 for this reason: praising "nice depth" during a bicep curl is
wrong. If you still have that file you may keep it here; it is not in
`EXPECTED_AUDIO_CLIPS` and is simply never selected.)

### `assets/audio/bicep curl/` — bicep curl correctives (5)

| File | Fires when |
|---|---|
| `correct_elbows_pinned.mp3` | elbow drift red |
| `correct_stop_swinging.mp3` | body swing red |
| `correct_match_arms.mp3` | asymmetry red |
| `correct_full_extension.mp3` | range of motion short at the bottom |
| `correct_higher.mp3` | range of motion short at the top |

### `assets/audio/shoulder press/` — shoulder press correctives (5)

| File | Fires when |
|---|---|
| `press_elbows_in.mp3` | elbow flare red |
| `press_dont_arch.mp3` | body swing red |
| `press_match_arms.mp3` | asymmetry red |
| `press_lock_out.mp3` | lockout short on a repetition |
| `press_all_the_way.mp3` | stalled while lowering |

## Format

MP3 is what the filenames assume, but the loader accepts anything `soundfile`
can decode — `.wav`, `.ogg`, `.flac`, `.m4a`, `.aiff` — and resolves by
basename, so the extension in `config.py` is the one it looks for. Clips are
decoded to NumPy once at startup and played from memory, so keep them short
(1–2 seconds); a long clip delays the next cue rather than queueing behind it.

## Check what is loaded

```bash
python -c "from src.audio_bridge import report_missing_clips; report_missing_clips()"
```

Prints one line per missing clip and a count of what resolved. The same check
runs automatically when `python app.py` starts.

## Source of truth

The list above is generated from `config.py` — `EXPECTED_AUDIO_CLIPS`,
`AUDIO_CURL_CORRECTIVE_CLIPS` and `AUDIO_PRESS_CORRECTIVE_CLIPS`. If you add a
cue there, add its file here.
