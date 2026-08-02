/* =============================================================================
   Virtual Gym Coach — merged web UI

   Module 1's readout (state / exercise / confidence, updated every frame) and
   Module 2's dashboard (overlay, zones, rep counter, charts, cue log, coaching
   audio) share this one page, one video view and one WebSocket. This file is
   Module 2's frontend (module2/frontend/app.js) with the upload step swapped
   for Module 1's auto-detect flow and a small Module 1 readout panel added —
   everything else (frame decoding, charts, zone strip, rep table, cue log,
   review/scrub mode) is reused unchanged, since the merged backend keeps
   Module 2's exact WebSocket wire format and job endpoints.

   A session has two halves and this file drives both of them through the SAME
   panels, because they show the same things about the same frames:

     LIVE     — a WebSocket delivers each annotated frame and its findings the
                moment the server has them.  The canvas, the info bar, the zone
                strip, the charts, the rep table and the coaching audio all
                update as the analysis advances through the video.
     REVIEW   — when the run finishes, the canvas is swapped for the recorded
                video and the same panels are re-driven from the timeline the
                live half accumulated, now scrubbable.

   So `state.result` is built up frame by frame during the live half, in exactly
   the shape the server's result document has.  When the run ends, the server's
   authoritative summary is merged over it and nothing has to be re-fetched or
   re-rendered from scratch.

   Frame indexing in review is `round(currentTime * fps)` against that timeline,
   so the panels never drift from the pixels.
   ============================================================================= */

(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);

  const COLORS = {
    zone: { green: "#78ff00", yellow: "#ffdc00", red: "#ff5050" },
    series: ["#3987e5", "#d95926"],          // validated categorical slots 1-2
    band: { yellow: "#fab219", red: "#d03b3b", gate: "#898781" },
    grid: "#2c2c2a",
    axis: "#383835",
    muted: "#898781",
    text: "#c3c2b7",
    surface: "#16171a",
    playhead: "#ffffff",
  };

  // How often the canvases are rebuilt while frames are arriving.  The chart and
  // strip redraws are O(frames), so doing one per arriving frame would spend
  // more time drawing the past than the server spends analysing the present.
  const REDRAW_INTERVAL_MS = 180;

  const state = {
    exercises: [],
    chartSpecs: {},        // exercise id -> chart definitions, from /api/exercises
    signals: [],
    exercise: null,        // set live once Module 1/2 lock onto one — never chosen up front
    file: null,
    jobId: null,
    socket: null,
    mode: "idle",          // "live" | "review" | "idle"
    result: null,
    fps: 25,
    expectedFrames: 0,     // declared frame count, for a stable live x-axis
    frame: -1,
    activeExercise: null,
    activeSignal: null,
    repHeader: [],
    cueCursor: 0,
    charts: [],
    strip: null,
    cueTimer: null,
    started: 0,
    dirty: false,
    lastRedraw: 0,
    pendingImage: null,
    decoding: false,
  };

  const audioEl = new Audio();
  audioEl.preload = "auto";

  // ── View switching ────────────────────────────────────────────────────────
  const VIEWS = ["view-upload", "view-session", "view-error"];
  function showView(id) {
    VIEWS.forEach((v) => { $(v).hidden = v !== id; });
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  // ── Formatting ────────────────────────────────────────────────────────────
  const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);

  function timecode(seconds) {
    if (!isFinite(seconds) || seconds < 0) seconds = 0;
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${m}:${String(s).padStart(2, "0")}`;
  }

  function fmtBytes(bytes) {
    const mb = bytes / (1024 * 1024);
    return mb >= 1 ? `${mb.toFixed(1)} MB` : `${(bytes / 1024).toFixed(0)} KB`;
  }

  function fmtCell(value) {
    if (value === null || value === undefined || value === "") return "—";
    if (typeof value === "boolean") return value ? "yes" : "no";
    if (typeof value === "number") {
      return Number.isInteger(value) ? String(value) : value.toFixed(2);
    }
    return String(value);
  }

  function escapeHtml(text) {
    return String(text).replace(/[&<>"]/g, (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  }

  function argmax(counts) {
    let best = null;
    Object.keys(counts).forEach((k) => {
      if (best === null || counts[k] > counts[best]) best = k;
    });
    return best;
  }

  // ── Boot ──────────────────────────────────────────────────────────────────
  async function boot() {
    wireUpload();
    wirePlayer();
    // The topbar's "New analysis" button is gone; the "Workout analysis" tab
    // is the way back to the upload screen. `resetToUpload` is still the
    // error view's recovery path.
    $("retry-btn").addEventListener("click", resetToUpload);
    $("stop-btn").addEventListener("click", stopAnalysis);
    $("audio-toggle").addEventListener("change", (e) => {
      if (!e.target.checked) audioEl.pause();
    });

    try {
      const meta = await fetch("/api/exercises").then((r) => r.json());
      state.exercises = meta.exercises;
      state.signals = meta.signals || [];
      meta.exercises.forEach((ex) => { state.chartSpecs[ex.id] = ex.charts || []; });
      renderUploadInfo(meta);
    } catch (err) {
      showFailure("Could not reach the analysis server. Is it still running?");
    }
    requestAnimationFrame(tick);
  }

  /**
   * Upload-step info panel. No exercise picker: Module 1 decides which
   * exercise is being performed live, frame by frame, once the video starts
   * streaming — the calibration note below is written generically rather
   * than naming one exercise's start pose.
   */
  function renderUploadInfo(meta) {
    const mb = Math.round(meta.limits.max_upload_bytes / (1024 * 1024));
    $("drop-limits").textContent =
      `MP4, MOV, AVI, MKV or WebM · up to ${mb} MB · up to ${meta.limits.max_frames.toLocaleString()} frames`;

    const names = state.exercises.map((ex) => ex.name).join(", ");
    $("calibration-note").innerHTML = `
      <li>Once Module 1 identifies your exercise, the first
          <strong>${meta.calibration_frames} detected frames</strong> of it
          calibrate your body proportions — hold the starting position steady
          when you begin.</li>
      <li>Module 2 currently supports: <strong>${escapeHtml(names)}</strong>.
          Other exercises Module 1 recognises still show in its panel, but
          form correction won't activate for them yet.</li>
      <li>Film from the <strong>front</strong> with your whole body in frame.
          Every threshold was derived from front-view data, and frames with
          low-confidence hips or knees are skipped by the rep counter.</li>`;
    refreshSubmit();
  }

  // ── Upload ────────────────────────────────────────────────────────────────
  function wireUpload() {
    const drop = $("dropzone");
    const input = $("file-input");

    drop.addEventListener("click", () => input.click());
    drop.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); }
    });
    ["dragenter", "dragover"].forEach((evt) =>
      drop.addEventListener(evt, (e) => {
        e.preventDefault();
        drop.classList.add("dragover");
      }));
    ["dragleave", "drop"].forEach((evt) =>
      drop.addEventListener(evt, (e) => {
        e.preventDefault();
        drop.classList.remove("dragover");
      }));
    drop.addEventListener("drop", (e) => {
      if (e.dataTransfer.files.length) setFile(e.dataTransfer.files[0]);
    });
    input.addEventListener("change", () => {
      if (input.files.length) setFile(input.files[0]);
    });
    $("clear-file").addEventListener("click", () => {
      input.value = "";
      setFile(null);
    });
    $("upload-form").addEventListener("submit", submitJob);
  }

  function setFile(file) {
    state.file = file;
    const chosen = $("file-chosen");
    chosen.hidden = !file;
    if (file) {
      chosen.querySelector(".chosen-name").textContent = file.name;
      chosen.querySelector(".chosen-size").textContent = fmtBytes(file.size);
    }
    $("upload-error").hidden = true;
    refreshSubmit();
  }

  function refreshSubmit() {
    $("analyse-btn").disabled = !state.file;
  }

  /**
   * Post the video, then open the socket that runs it.
   *
   * The upload only STORES the file — nothing is analysed until the socket
   * attaches, so the run and the thing watching it start together and the user
   * never waits on a result that is already being computed out of sight.
   * No exercise field: Module 1 decides that live.
   */
  async function submitJob(event) {
    event.preventDefault();
    if (!state.file) return;

    const body = new FormData();
    body.append("video", state.file);

    $("analyse-btn").disabled = true;
    $("upload-error").hidden = true;

    try {
      const res = await fetch("/api/videos", { method: "POST", body });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Upload failed.");
      state.jobId = data.job_id;
      beginLiveView();
      openStream(data.job_id);
    } catch (err) {
      const box = $("upload-error");
      box.textContent = err.message;
      box.hidden = false;
      $("analyse-btn").disabled = false;
    }
  }

  // ══════════════════════════════════════════════════════════════════════════
  //  LIVE HALF
  // ══════════════════════════════════════════════════════════════════════════

  function beginLiveView() {
    state.mode = "live";
    state.result = null;
    state.frame = -1;
    state.activeExercise = null;
    state.activeSignal = null;
    state.repHeader = [];
    state.charts = [];
    state.strip = null;
    state.started = performance.now();

    $("live-canvas").hidden = true;
    $("player").hidden = true;
    $("shell-placeholder").hidden = false;
    $("shell-placeholder").textContent = "Connecting…";
    $("live-badge").hidden = false;
    $("live-transport").hidden = false;
    $("review-transport").hidden = true;
    $("downloads-panel").hidden = true;
    $("breakdown-panel").hidden = true;
    $("zone-strip").hidden = true;
    // Stop is re-armed here, since stopAnalysis() disables it while the server
    // finishes the frame it is on.
    $("stop-btn").disabled = false;
    $("stop-btn").textContent = "Stop";
    $("charts").innerHTML = "";
    $("rep-table").innerHTML = "";
    $("cue-log").innerHTML = "";
    $("summary-tiles").innerHTML = "";
    $("info-bar").innerHTML = "";
    $("overall-banner").textContent = "—";
    $("live-reps").textContent = "0";
    $("live-phase").textContent = "—";
    $("live-last").textContent = "";
    $("progress-fill").style.width = "0%";
    $("live-stat").textContent = "starting…";
    $("live-rate").textContent = "";
    $("rep-hint").textContent = "Repetitions appear here as the counter closes them.";
    resetModule1Panel();
    renderSignalBar();
    showView("view-session");
  }

  function openStream(jobId) {
    const scheme = location.protocol === "https:" ? "wss:" : "ws:";
    // The demo username rides in the query string: a WebSocket handshake
    // cannot carry the X-Demo-User header the other routes use. It decides
    // only whose exercise log this workout is written to when it finishes.
    const who = window.Auth ? Auth.username() : "";
    const query = who ? `?user=${encodeURIComponent(who)}` : "";
    const socket = new WebSocket(
      `${scheme}//${location.host}/api/jobs/${jobId}/stream${query}`);
    socket.binaryType = "arraybuffer";
    state.socket = socket;

    socket.onmessage = (event) => {
      if (typeof event.data === "string") onControlMessage(JSON.parse(event.data));
      else onFrameMessage(event.data);
    };
    socket.onerror = () => {
      if (state.mode === "live") showFailure("The connection to the analyser dropped.");
    };
    socket.onclose = () => {
      state.socket = null;
      if (state.mode === "live") {
        // A close with no `done` and no `error` means the server went away
        // mid-run; say so rather than leaving a half-drawn view up.
        showFailure("The analysis stopped before it finished.");
      }
    };
  }

  function sendControl(payload) {
    if (state.socket && state.socket.readyState === WebSocket.OPEN) {
      state.socket.send(JSON.stringify(payload));
    }
  }

  /**
   * Ask the server to stop, then WAIT.
   *
   * This used to close the socket and jump straight back to the upload screen,
   * throwing away everything that had been analysed. The server now treats
   * Stop as an ending rather than a failure (backend/merged_analysis.py): it
   * finishes the frame it is on, closes the video and CSVs, and sends the same
   * `done` summary a full run sends, covering the part it got through. So the
   * socket stays open and the normal review flow takes over.
   */
  function stopAnalysis() {
    const button = $("stop-btn");
    button.disabled = true;
    button.textContent = "Stopping…";
    $("live-stat").textContent = "Finishing the current frame…";
    sendControl({ type: "stop" });
  }

  function onControlMessage(message) {
    switch (message.type) {
      case "status":
        if (message.status === "queued") {
          $("shell-placeholder").textContent =
            message.detail || "Queued behind another analysis…";
        } else if (message.status === "running") {
          $("shell-placeholder").textContent = "Waiting for the first frame…";
          if (message.signals) { state.signals = message.signals; renderSignalBar(); }
        } else if (message.status === "cancelled") {
          // The merged pipeline no longer reports this for a user-pressed Stop
          // — that arrives as a normal `done`. This is the fallback for a job
          // killed some other way (DELETE /api/jobs/{id}, or a shutdown), and
          // even then it keeps whatever was collected rather than discarding it.
          state.mode = "idle";
          if (state.result && state.result.video.frames > 0) enterReview(null);
          else showFailure("The analysis was cancelled before any frames were read.");
        }
        break;
      case "meta":
        onMeta(message);
        break;
      case "done":
        enterReview(message.summary);
        break;
      case "error":
        state.mode = "idle";
        showFailure(message.message || "The analysis failed.");
        break;
      default:
        break;
    }
  }

  function onMeta(meta) {
    state.fps = meta.fps || 25;
    // A declared frame count fixes the x-axis for the whole run, so the traces
    // sweep left to right like a monitor instead of being rescaled every time a
    // frame arrives.  Containers that declare nothing fall back to a growing
    // axis, which is the honest thing to show when the length is unknown.
    state.expectedFrames = meta.declared_frames > 0
      ? Math.min(meta.declared_frames, meta.max_frames) : 0;
    state.result = freshResult(meta);
    setChartsFor(state.exercise);
  }

  function freshResult(meta) {
    return {
      exercise: null,
      video: {
        file: null, codec: meta.codec, fps: meta.fps, frames: 0,
        width: meta.width, height: meta.height,
        source_width: meta.source_width, source_height: meta.source_height,
        scaled: meta.width !== meta.source_width, duration: 0, truncated: false,
      },
      processing: { seconds: 0, fps: 0, realtime_ratio: 0 },
      calibration: {
        done: false, frames_required: meta.calibration_frames, restarts: 0,
        completed_frame: null, completed_time: null, clean: true,
        rejected_frames: null, values: {},
      },
      detection: { pose_frames: 0, total_frames: 0, rate: 0 },
      view: { counts: {}, dominant: null },
      reps: { count: 0, header: [], rows: [], quality: { green: 0, yellow: 0, red: 0 } },
      zones: { counts: {} },
      cues: [],
      charts: [],
      segments: [],
      artifacts: {},
      timeline: [],
    };
  }

  /**
   * Unpack one frame message.
   *
   *   [4-byte big-endian header length][UTF-8 JSON header][JPEG bytes]
   *
   * One message rather than two, so a header can never be shown against the
   * wrong image (see src/live_session.py).
   */
  function onFrameMessage(buffer) {
    const view = new DataView(buffer);
    const headerLength = view.getUint32(0);
    const header = JSON.parse(
      new TextDecoder().decode(new Uint8Array(buffer, 4, headerLength)));
    const imageOffset = 4 + headerLength;
    if (buffer.byteLength > imageOffset) {
      queueImage(new Uint8Array(buffer, imageOffset));
    }
    ingestFrame(header);
  }

  /**
   * Show the newest frame, never a backlog of stale ones.
   *
   * `createImageBitmap` is asynchronous, so frames can arrive faster than they
   * decode.  Only the latest is kept pending: dropping an intermediate PICTURE
   * costs nothing, since every frame's numbers were already recorded by
   * `ingestFrame` before it got here, and queueing them all would make the
   * canvas lag further behind the panels beside it.
   */
  function queueImage(bytes) {
    state.pendingImage = new Blob([bytes], { type: "image/jpeg" });
    if (!state.decoding) drainImages();
  }

  async function drainImages() {
    state.decoding = true;
    const canvas = $("live-canvas");
    const ctx = canvas.getContext("2d");
    while (state.pendingImage) {
      const blob = state.pendingImage;
      state.pendingImage = null;
      try {
        const bitmap = await createImageBitmap(blob);
        if (canvas.width !== bitmap.width || canvas.height !== bitmap.height) {
          canvas.width = bitmap.width;
          canvas.height = bitmap.height;
        }
        ctx.drawImage(bitmap, 0, 0);
        bitmap.close();
        if (canvas.hidden) {
          canvas.hidden = false;
          $("shell-placeholder").hidden = true;
        }
      } catch (err) {
        /* a preview that would not decode is not worth failing a run over */
      }
    }
    state.decoding = false;
  }

  /**
   * Fold one frame's findings into the accumulating result document.
   *
   * This is where the live half earns the review half for free: everything is
   * written into the same shape the server's result document has, so the panels
   * that render a finished analysis render an unfinished one unchanged.
   */
  function ingestFrame(h) {
    const r = state.result;
    if (!r) return;

    updateModule1Panel(h.module1);

    if (h.ex !== state.activeExercise) onExerciseChanged(h.ex);
    if (h.signal !== state.activeSignal) {
      state.activeSignal = h.signal;
      highlightSignal(h.signal);
    }

    r.timeline.push({
      f: h.f, t: h.t, pose: h.pose, zones: h.zones, phase: h.phase,
      reps: h.reps, view: h.view, info: h.info, metrics: h.metrics,
      signal: h.signal, ex: h.ex, banner: h.banner,
    });
    state.frame = r.timeline.length - 1;

    r.video.frames = r.timeline.length;
    r.video.duration = Number((h.t || 0).toFixed(2));
    r.detection.total_frames = r.timeline.length;
    if (h.pose) {
      r.detection.pose_frames += 1;
      if (h.view) {
        r.view.counts[h.view] = (r.view.counts[h.view] || 0) + 1;
        r.view.dominant = argmax(r.view.counts);
      }
    }
    r.detection.rate = r.detection.pose_frames / r.detection.total_frames;

    Object.keys(h.zones || {}).forEach((channel) => {
      const bucket = r.zones.counts[channel] ||
        (r.zones.counts[channel] = { green: 0, yellow: 0, red: 0 });
      bucket[h.zones[channel]] += 1;
    });

    if (h.calib) {
      r.calibration.restarts = h.calib.restarts;
      if (h.calib.done && !r.calibration.done) {
        r.calibration.done = true;
        r.calibration.completed_frame = h.f;
        r.calibration.completed_time = Number((h.t || 0).toFixed(2));
      }
    }

    if (h.rep) appendRep(h.rep, h.rep_header);
    if (h.cue) {
      r.cues.push(h.cue);
      appendCue(h.cue);
      showCueBanner(h.cue);
      playCue(h.cue);
    }

    updateLivePanel(r.timeline[state.frame]);
    updateLiveTransport(h);
    state.dirty = true;
  }

  // ── Module-1 panel ────────────────────────────────────────────────────────
  function resetModule1Panel() {
    $("m1-state").textContent = "—";
    $("m1-state").className = "m1-tile-value";
    $("m1-exercise").textContent = "—";
    $("m1-confidence").textContent = "—";
    setM1ProbBar($("m1-prob-active-bar"), $("m1-prob-active-value"), null);
    setM1ProbBar($("m1-prob-rest-bar"), $("m1-prob-rest-value"), null);
    setM1ProbBar($("m1-prob-null-bar"), $("m1-prob-null-value"), null);
  }

  function setM1ProbBar(barEl, valueEl, prob) {
    if (prob === null || prob === undefined) {
      barEl.style.width = "0%";
      valueEl.textContent = "—";
      return;
    }
    const pct = prob * 100;
    barEl.style.width = `${pct.toFixed(0)}%`;
    valueEl.textContent = `${pct.toFixed(0)}%`;
  }

  /** Driven by the `module1` field the merged backend attaches to every frame. */
  function updateModule1Panel(m1) {
    if (!m1) return;
    const st = m1.state || "null";
    const stateEl = $("m1-state");
    stateEl.textContent = st.toUpperCase();
    stateEl.className = `m1-tile-value state-${st}`;
    if (m1.exercise) {
      $("m1-exercise").textContent = m1.exercise.replace(/_/g, " ");
    } else if (m1.label === "VERIFYING..." && m1.early_guess) {
      // Not locked yet, but Module 2 may already be calibrating on this guess
      // (see backend/signal_mapping.py) — show it rather than a bare "verifying".
      $("m1-exercise").textContent = `verifying: ${m1.early_guess.replace(/_/g, " ")}…`;
    } else if (m1.label === "VERIFYING...") {
      $("m1-exercise").textContent = "verifying…";
    } else {
      $("m1-exercise").textContent = "—";
    }
    $("m1-confidence").textContent =
      (typeof m1.confidence === "number" && m1.confidence > 0)
        ? `${m1.confidence.toFixed(1)}%` : "—";
    const probs = m1.state_probs || {};
    setM1ProbBar($("m1-prob-active-bar"), $("m1-prob-active-value"), probs.active);
    setM1ProbBar($("m1-prob-rest-bar"), $("m1-prob-rest-value"), probs.rest);
    setM1ProbBar($("m1-prob-null-bar"), $("m1-prob-null-value"), probs.null);
  }

  function onExerciseChanged(exercise) {
    state.activeExercise = exercise;
    const r = state.result;
    if (r) {
      r.exercise = exercise;
      // A new exercise means a new rep counter with its own columns, so the
      // table starts again rather than mixing two schemas under one header.
      r.reps = { count: 0, header: [], rows: [],
                 quality: { green: 0, yellow: 0, red: 0 } };
      state.repHeader = [];
      $("rep-table").innerHTML = "";
    }
    setChartsFor(exercise);
  }

  function setChartsFor(exercise) {
    const specs = state.chartSpecs[exercise] || [];
    if (state.result) state.result.charts = specs;
    buildCharts(specs);
  }

  function updateLiveTransport(h) {
    const r = state.result;
    const done = r.timeline.length;
    const total = state.expectedFrames;
    const fraction = total > 0 ? clamp(done / total, 0, 1) : 0;
    $("progress-fill").style.width = total > 0 ? `${(fraction * 100).toFixed(1)}%` : "100%";
    $("progress-bar-wrap").setAttribute("aria-valuenow", Math.round(fraction * 100));

    const seconds = (performance.now() - state.started) / 1000;
    r.processing.seconds = Number(seconds.toFixed(2));
    r.processing.fps = seconds > 0 ? Number((done / seconds).toFixed(2)) : 0;
    r.processing.realtime_ratio = seconds > 0 && state.fps
      ? Number(((done / seconds) / state.fps).toFixed(3)) : 0;

    $("live-stat").textContent = total > 0
      ? `${done.toLocaleString()} / ${total.toLocaleString()} frames · ${timecode(h.t)}`
      : `${done.toLocaleString()} frames · ${timecode(h.t)}`;
    $("live-rate").textContent =
      `${r.processing.fps} fps · ${r.processing.realtime_ratio}x real time`;
  }

  // ── Module-1 signal control (manual override — hidden, unused by default) ─
  function renderSignalBar() {
    // Module 1's live classification drives Module 2 automatically now, so
    // this control stays out of the UI. Left wired rather than removed, so a
    // future debugging build can flip it back on with a one-line change.
    $("signal-bar").hidden = true;
  }

  function highlightSignal(label) {
    $("signal-buttons").querySelectorAll(".signal-btn").forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.label === label);
    });
  }

  // ══════════════════════════════════════════════════════════════════════════
  //  REVIEW HALF
  // ══════════════════════════════════════════════════════════════════════════

  /**
   * Swap the live canvas for the recording, keeping every panel in place.
   *
   * The server's summary is merged over what the live half accumulated: the
   * two agree on every number, but the summary is the authority (it has the
   * per-segment breakdown, the true processing time, and the artefact names),
   * while the timeline is only ever assembled here.
   */
  function enterReview(summary) {
    const r = state.result;
    if (!r) { showFailure("The analysis produced no frames."); return; }
    state.mode = "review";
    Object.keys(summary || {}).forEach((key) => { r[key] = summary[key]; });
    state.fps = r.video.fps || state.fps;
    state.cueCursor = 0;
    state.frame = -1;

    $("live-badge").hidden = true;
    $("live-transport").hidden = true;
    $("review-transport").hidden = false;
    $("signal-bar").hidden = true;
    $("live-canvas").hidden = true;
    $("shell-placeholder").hidden = true;
    $("downloads-panel").hidden = false;

    const badge = $("codec-badge");
    badge.textContent =
      `${r.video.width}x${r.video.height} · ${r.video.fps} fps · ${r.video.codec}`;
    badge.title = r.video.scaled
      ? `Analysed at full resolution (${r.video.source_width}x${r.video.source_height}); ` +
        `the review copy is downscaled for playback.`
      : "Review copy is at the source resolution.";

    // A clip that hit the frame ceiling, or one the user stopped, was analysed
    // only up to that point. Saying so matters: every count below — reps, cues,
    // durations — is a count for the analysed part, not the whole file.
    const truncated = $("truncated-badge");
    truncated.hidden = !(r.video.truncated || r.video.stopped);
    truncated.textContent = r.video.stopped
      ? `stopped early · ${r.video.frames.toLocaleString()} frames analysed`
      : `analysed first ${r.video.frames.toLocaleString()} frames only`;

    renderTiles(r);
    renderBreakdown(r);
    renderRepTable(r);
    renderCueLog(r);
    buildCharts(r.charts);
    buildZoneStrip(r);
    renderDownloads(r);

    const player = $("player");
    player.hidden = false;
    player.src = `/api/jobs/${state.jobId}/video`;
    player.load();

    // Canvas sizing needs the element on screen, so lay out after the swap.
    requestAnimationFrame(() => { resizeAll(); syncToVideo(true); });
  }

  function resetToUpload() {
    state.mode = "idle";
    if (state.socket) {
      state.socket.onclose = null;
      state.socket.close();
      state.socket = null;
    }
    stopAudio();
    const player = $("player");
    player.pause();
    player.removeAttribute("src");
    player.load();
    state.jobId = null;
    state.result = null;
    state.frame = -1;
    state.exercise = null;
    state.charts = [];
    state.strip = null;
    $("analyse-btn").disabled = !state.file;
    showView("view-upload");
  }

  function showFailure(message) {
    state.mode = "idle";
    if (state.socket) {
      state.socket.onclose = null;
      state.socket.close();
      state.socket = null;
    }
    $("fail-message").textContent = message;
    showView("view-error");
  }

  // ── Panels ────────────────────────────────────────────────────────────────
  function renderTiles(r) {
    const q = r.reps.quality;
    const detection = `${(r.detection.rate * 100).toFixed(1)}%`;

    // The view detector is a pure observer — it changes no threshold or cue —
    // so this tile reports what it saw and why that matters, rather than
    // implying the analysis was adjusted for it.
    const view = r.view.dominant || "unknown";
    const viewTotal = Object.values(r.view.counts).reduce((a, b) => a + b, 0);
    const viewShare = viewTotal
      ? Math.round((r.view.counts[view] || 0) / viewTotal * 100) : 0;
    const VIEW_NOTES = {
      front: "matches the view every threshold was derived from",
      diagonal: "part-profile — frontal-plane channels lose accuracy",
      side: "profile — left/right geometry is not measurable",
      unknown: "too ambiguous to call confidently",
    };
    const VIEW_TONES = { front: "good", diagonal: "warning", side: "critical",
                         unknown: "warning" };
    const cal = r.calibration;
    const calValue = !cal.done ? "not reached"
      : cal.clean ? "clean" : "fallback";
    const calNote = !cal.done
      ? `fewer than ${cal.frames_required} usable frames`
      : cal.restarts
        ? `${cal.restarts} restart${cal.restarts > 1 ? "s" : ""} · locked at ${cal.completed_time}s`
        : `locked at ${cal.completed_time}s`;

    const tiles = [
      { label: "Reps counted", value: r.reps.count,
        note: `${q.green} good · ${q.yellow} warning · ${q.red} risk` },
      { label: "Pose detected", value: detection,
        note: `${r.detection.pose_frames.toLocaleString()} of ${r.detection.total_frames.toLocaleString()} frames`,
        tone: r.detection.rate >= 0.95 ? "good" : r.detection.rate >= 0.8 ? "warning" : "critical" },
      { label: "Camera view", value: view,
        note: `${viewShare}% of tracked frames · ${VIEW_NOTES[view] || ""}`,
        tone: VIEW_TONES[view] || "warning" },
      { label: "Calibration", value: calValue, note: calNote,
        tone: !cal.done ? "critical" : cal.clean ? "good" : "warning" },
      { label: "Coaching cues", value: r.cues.length,
        note: `over ${r.video.duration}s of video` },
      { label: "Analysis speed", value: `${r.processing.fps} fps`,
        note: `${r.processing.realtime_ratio}x real time · ${r.processing.seconds}s total` },
    ];

    // What went to Module 3. Absent on results produced before that hand-off
    // existed, so this is additive rather than assumed.
    if (r.nutrition) {
      const log = r.nutrition.log || [];
      tiles.push(log.length
        ? { label: "Energy burned", value: `${r.nutrition.calories_burned} kcal`,
            tone: "good",
            note: `${log.map((e) => `${e.exercise_name.replace(/_/g, " ")} ` +
                                    `${e.duration_minutes} min`).join(" · ")} · ` +
                  `logged to your meal plan` }
        : { label: "Energy burned", value: "not logged",
            note: "no exercise was held long enough to record" });
    }

    $("summary-tiles").innerHTML = tiles.map((t) => `
      <div class="tile">
        <div class="tile-label">${t.label}</div>
        <div class="tile-value ${t.tone || ""}">${t.value}</div>
        <div class="tile-note">${t.note}</div>
      </div>`).join("");
  }

  /** 12 -> "12s", 134 -> "2m 14s". Rounded: sub-second precision means nothing here. */
  function formatDuration(seconds) {
    const total = Math.max(0, Math.round(seconds || 0));
    const minutes = Math.floor(total / 60);
    return minutes
      ? `${minutes}m ${String(total % 60).padStart(2, "0")}s`
      : `${total}s`;
  }

  /** 'bicep_curl' -> 'Bicep curl'. Module 1's labels are snake_case. */
  function prettyExercise(name) {
    const text = String(name || "").replace(/_/g, " ");
    return text.charAt(0).toUpperCase() + text.slice(1);
  }

  /**
   * Where the session's time went, split by Module 1's frame-by-frame verdict.
   *
   * Every number is `frames / fps`, so it is time IN THE VIDEO — not how long
   * the server took, and not wall-clock. The bars are shares of the analysed
   * duration, which is why they add up.
   */
  function renderBreakdown(r) {
    const panel = $("breakdown-panel");
    const n = r.nutrition;
    // Results produced before this section existed have no `nutrition` block.
    if (!n) { panel.hidden = true; return; }
    panel.hidden = false;

    const kcal = {};
    (n.log || []).forEach((e) => { kcal[e.exercise_name] = e.calories_burned; });

    const durations = n.durations || {};
    const rows = Object.keys(durations)
      .sort((a, b) => durations[b] - durations[a])
      .map((name) => ({
        cls: "work",
        label: prettyExercise(name),
        seconds: durations[name],
        note: kcal[name] != null ? `${kcal[name]} kcal` : "",
      }));

    const worked = rows.reduce((sum, row) => sum + row.seconds, 0);
    const rest = n.rest_seconds || 0;
    // `n.idle_seconds` (nobody detected) is deliberately not shown as a row —
    // it reads as an error message rather than useful information, and this
    // panel is exactly what fills the empty space under the video once a
    // session ends or is stopped, so it should read as a clean summary.

    if (rest > 0) {
      rows.push({ cls: "rest", label: "Rest between sets", seconds: rest, note: "" });
    }

    const span = Math.max(r.video.duration || 0, worked + rest, 1);

    if (!rows.length) {
      $("breakdown").innerHTML =
        `<p class="hint">Module 1 never confirmed an exercise in this clip, so
         there is no time to attribute.</p>`;
    } else {
      $("breakdown").innerHTML = rows.map((row) => `
        <div class="bd-row">
          <span class="bd-name">${row.label}</span>
          <span class="bd-track">
            <span class="bd-fill bd-${row.cls}" style="width:${
              Math.min(100, (row.seconds / span) * 100).toFixed(1)}%"></span>
          </span>
          <span class="bd-time">${formatDuration(row.seconds)}</span>
          <span class="bd-note">${row.note}</span>
        </div>`).join("");
    }

    const burned = n.calories_burned || 0;
    const parts = [
      `<strong>${formatDuration(r.video.duration)}</strong> analysed`,
      `<strong>${formatDuration(worked)}</strong> exercising`,
      `<strong>${formatDuration(rest)}</strong> resting`,
    ];
    if (burned > 0) {
      parts.push(`<strong>${burned} kcal</strong> logged to your meal plan`);
    }
    $("breakdown").insertAdjacentHTML("beforeend",
      `<div class="bd-total">${parts.join(" &middot; ")}</div>`);

    $("breakdown-hint").textContent = r.video.stopped
      ? "You stopped this run early — everything below covers the part that was analysed."
      : "Measured frame by frame by Module 1, as time in the video.";
  }

  function renderRepTable(r) {
    const table = $("rep-table");
    const hint = $("rep-hint");
    if (!r.reps.rows.length) {
      table.innerHTML = "";
      hint.textContent = r.calibration.done
        ? "No repetition completed the counter's gates in this clip."
        : "Calibration never completed, so rep counting never started.";
      return;
    }
    hint.textContent = "Click a row to jump to that repetition.";
    state.repHeader = r.reps.header;
    table.innerHTML = repTableHead(r.reps.header) +
      `<tbody>${r.reps.rows.map((rep) => repRow(rep, r.reps.header)).join("")}</tbody>`;
    wireRepRows();
  }

  function repTableHead(header) {
    return `<thead><tr>${header.map((h) =>
      `<th>${h.replace(/_/g, " ")}</th>`).join("")}</tr></thead>`;
  }

  function repRow(rep, header) {
    const startFrame = rep.start_frame !== undefined ? rep.start_frame : 0;
    const cells = header.map((key) => {
      const cls = key === "rep_quality" ? ` class="q-${rep[key]}"` : "";
      return `<td${cls}>${fmtCell(rep[key])}</td>`;
    }).join("");
    return `<tr data-frame="${startFrame}">${cells}</tr>`;
  }

  function wireRepRows() {
    $("rep-table").querySelectorAll("tbody tr").forEach((tr) => {
      tr.onclick = () => seekToFrame(Number(tr.dataset.frame));
    });
  }

  /** Append one repetition as the counter closes it, mid-analysis. */
  function appendRep(rep, header) {
    const r = state.result;
    if (!state.repHeader.length) {
      state.repHeader = header || Object.keys(rep);
      r.reps.header = state.repHeader;
      $("rep-table").innerHTML = repTableHead(state.repHeader) + "<tbody></tbody>";
      $("rep-hint").textContent = "Click a row to jump to that repetition.";
    }
    r.reps.rows.push(rep);
    r.reps.count = r.reps.rows.length;
    if (rep.rep_quality in r.reps.quality) r.reps.quality[rep.rep_quality] += 1;

    const body = $("rep-table").querySelector("tbody");
    if (body) {
      body.insertAdjacentHTML("beforeend", repRow(rep, state.repHeader));
      wireRepRows();
    }
  }

  function renderCueLog(r) {
    const log = $("cue-log");
    if (!r.cues.length) {
      log.innerHTML = `<li class="empty">No cue met its dwell and cooldown
        conditions in this clip.</li>`;
      return;
    }
    log.innerHTML = r.cues.map(cueItem).join("");
    wireCueItems();
  }

  function cueItem(cue) {
    return `
      <li>
        <button type="button" class="cue-item" data-frame="${cue.frame}">
          <span class="cue-time">${timecode(cue.time)}</span>
          <span class="cue-text">${escapeHtml(cue.caption)}</span>
          <span class="cue-tag ${cue.category}">${cue.category}</span>
        </button>
      </li>`;
  }

  function wireCueItems() {
    $("cue-log").querySelectorAll(".cue-item").forEach((btn) => {
      btn.onclick = () => seekToFrame(Number(btn.dataset.frame));
    });
  }

  function appendCue(cue) {
    const log = $("cue-log");
    const empty = log.querySelector(".empty");
    if (empty) empty.remove();
    log.insertAdjacentHTML("beforeend", cueItem(cue));
    wireCueItems();
  }

  function renderDownloads(r) {
    const a = r.artifacts || {};
    const featureFiles = a.feature_csvs && a.feature_csvs.length
      ? a.feature_csvs : [a.features_csv];
    const repFiles = a.rep_summary_csvs && a.rep_summary_csvs.length
      ? a.rep_summary_csvs : [a.rep_summary_csv];
    const files = [
      ...featureFiles.map((n) => [n, "per-frame metrics + zones"]),
      ...repFiles.map((n) => [n, "one row per repetition"]),
      [a.landmarks_csv, "raw landmark coordinates"],
      [r.video.file, "annotated video"],
    ].filter(([name]) => Boolean(name));

    $("downloads").innerHTML = files.map(([name, note]) => `
      <a class="download-link" download href="/api/jobs/${state.jobId}/files/${name}">
        ${name} <span>${note}</span>
      </a>`).join("");
  }

  // ── Player wiring ─────────────────────────────────────────────────────────
  function wirePlayer() {
    const player = $("player");
    const playBtn = $("play-btn");

    playBtn.addEventListener("click", () => {
      if (player.paused) player.play(); else player.pause();
    });
    player.addEventListener("play", () => setPlayIcon(true));
    player.addEventListener("pause", () => setPlayIcon(false));
    player.addEventListener("timeupdate", () => syncToVideo(false));
    player.addEventListener("seeking", () => {
      // Re-aim the cue cursor at the new position so scrubbing never replays
      // a stretch of coaching the user has jumped over.
      stopAudio();
      state.cueCursor = firstCueAfter(currentFrame());
    });
    player.addEventListener("seeked", () => syncToVideo(true));
    player.addEventListener("loadedmetadata", () => { resizeAll(); syncToVideo(true); });

    window.addEventListener("resize", debounce(resizeAll, 150));
    document.addEventListener("keydown", (e) => {
      if (state.mode !== "review") return;
      const tag = (e.target.tagName || "").toLowerCase();
      if (tag === "input" || tag === "textarea") return;
      if (e.code === "Space") { e.preventDefault(); playBtn.click(); }
      if (e.code === "ArrowRight") { e.preventDefault(); stepFrames(e.shiftKey ? 10 : 1); }
      if (e.code === "ArrowLeft") { e.preventDefault(); stepFrames(e.shiftKey ? -10 : -1); }
    });
  }

  function setPlayIcon(playing) {
    $("play-btn").innerHTML = playing
      ? `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M7 5h3.5v14H7zM13.5 5H17v14h-3.5z"/></svg>`
      : `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 5.5v13l11-6.5z"/></svg>`;
    $("play-btn").setAttribute("aria-label", playing ? "Pause" : "Play");
  }

  function currentFrame() {
    if (state.mode !== "review") return state.frame < 0 ? 0 : state.frame;
    const t = $("player").currentTime || 0;
    const total = state.result ? state.result.timeline.length : 0;
    return clamp(Math.round(t * state.fps), 0, Math.max(total - 1, 0));
  }

  function stepFrames(delta) {
    const player = $("player");
    player.pause();
    player.currentTime = clamp(player.currentTime + delta / state.fps, 0,
                               player.duration || 0);
  }

  function seekToFrame(frame) {
    if (state.mode !== "review") return;   // nothing to seek in yet
    const player = $("player");
    player.currentTime = clamp(frame / state.fps, 0, player.duration || 0);
    syncToVideo(true);
  }

  /**
   * The one animation loop.
   *
   * In review it advances the playhead (`timeupdate` fires only ~4x/s, too
   * coarse); in live it is the redraw throttle, so the charts and the zone strip
   * are rebuilt on a schedule rather than once per arriving frame.
   */
  function tick() {
    requestAnimationFrame(tick);
    if (!state.result) return;
    if (state.mode === "review") {
      if (!$("player").paused) syncToVideo(false);
      return;
    }
    if (state.mode === "live" && state.dirty) {
      const now = performance.now();
      if (now - state.lastRedraw >= REDRAW_INTERVAL_MS) {
        state.lastRedraw = now;
        state.dirty = false;
        redrawLive();
      }
    }
  }

  function redrawLive() {
    buildZoneStrip(state.result);
    state.charts.forEach(renderChartStatic);
    drawPlayheads(state.frame < 0 ? 0 : state.frame);
    renderTiles(state.result);
  }

  function syncToVideo(force) {
    if (!state.result) return;
    const frame = currentFrame();
    if (frame === state.frame && !force) return;
    const previous = state.frame;
    state.frame = frame;

    updateLivePanel(state.result.timeline[frame]);
    updateTimeReadout();
    drawPlayheads(frame);
    if (!force) fireCues(previous, frame);
  }

  function updateTimeReadout() {
    const player = $("player");
    $("time-readout").textContent =
      `${timecode(player.currentTime)} / ${timecode(player.duration || state.result.video.duration)}`;
  }

  function updateLivePanel(entry) {
    if (!entry) return;
    const info = entry.info || {};

    const banner = $("overall-banner");
    if (info.overall) {
      banner.textContent = info.overall.text;
      banner.style.color = info.overall.color;
    } else if (entry.banner) {
      // Null / Rest: no exercise is active, so the controller's prompt is the
      // whole message.
      banner.textContent = entry.banner;
      banner.style.removeProperty("color");
    } else {
      banner.textContent = entry.pose ? "—" : "No pose detected";
      banner.style.removeProperty("color");
    }
    banner.dataset.zone = (entry.zones && entry.zones.overall) || "";

    const rep = info.rep_row;
    $("live-reps").textContent = entry.reps || 0;
    const phase = $("live-phase");
    phase.textContent = (rep && rep.phase) || entry.phase || "—";
    phase.style.color = (rep && rep.phase_color) || COLORS.muted;
    const last = $("live-last");
    last.textContent = (rep && rep.last_text) || "";
    last.style.color = (rep && rep.last_color) || COLORS.text;

    $("info-bar").innerHTML = (info.columns || [])
      .filter((col) => col.length)
      .map((col) => `<div class="info-col">${col.map((cell) =>
        `<span class="info-cell" style="color:${cell.color}">${escapeHtml(cell.text)}</span>`
      ).join("")}</div>`).join("");
  }

  // ── Coaching audio ────────────────────────────────────────────────────────
  function firstCueAfter(frame) {
    const cues = state.result.cues;
    let i = 0;
    while (i < cues.length && cues[i].frame <= frame) i++;
    return i;
  }

  /**
   * Play any cue decided between the previous frame and this one (review only).
   *
   * Only the LAST cue in the interval is played when several fall in one step,
   * mirroring the desktop player, where starting a clip cuts off the previous
   * one so exactly one is ever audible.
   */
  function fireCues(previous, frame) {
    const cues = state.result.cues;
    if (!cues.length || frame <= previous) return;

    let fired = null;
    while (state.cueCursor < cues.length && cues[state.cueCursor].frame <= frame) {
      if (cues[state.cueCursor].frame > previous) fired = cues[state.cueCursor];
      state.cueCursor++;
    }
    if (!fired) return;
    showCueBanner(fired);
    playCue(fired);
  }

  function playCue(cue) {
    if (!$("audio-toggle").checked) return;
    audioEl.pause();
    audioEl.src = `/api/audio/${encodeURIComponent(cue.clip)}`;
    audioEl.currentTime = 0;
    audioEl.play().catch(() => { /* autoplay blocked until the user interacts */ });
  }

  function stopAudio() {
    audioEl.pause();
    if (state.cueTimer) clearTimeout(state.cueTimer);
    $("cue-now").hidden = true;
  }

  function showCueBanner(cue) {
    const box = $("cue-now");
    $("cue-now-text").textContent = cue.caption;
    box.hidden = false;
    if (state.cueTimer) clearTimeout(state.cueTimer);
    state.cueTimer = setTimeout(() => { box.hidden = true; }, 2600);
  }

  // ── Canvas plumbing ───────────────────────────────────────────────────────
  /**
   * Every canvas keeps a STATIC buffer that is redrawn only when the data
   * changes.  The per-frame work is then a blit plus one playhead line, which
   * keeps four charts and the zone strip at 60 fps instead of re-tracing
   * thousands of points each time the playhead moves.
   */
  function setupCanvas(canvas, cssHeight) {
    const dpr = window.devicePixelRatio || 1;
    const width = canvas.clientWidth || canvas.parentElement.clientWidth || 600;
    canvas.width = Math.max(1, Math.round(width * dpr));
    canvas.height = Math.max(1, Math.round(cssHeight * dpr));
    canvas.style.height = `${cssHeight}px`;
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, width, height: cssHeight, dpr };
  }

  function makeBuffer(width, height, dpr) {
    const buffer = document.createElement("canvas");
    buffer.width = Math.max(1, Math.round(width * dpr));
    buffer.height = Math.max(1, Math.round(height * dpr));
    const ctx = buffer.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { buffer, ctx };
  }

  /**
   * How many frames the x-axis spans.
   *
   * While analysing, the DECLARED length of the video, so the traces sweep left
   * to right against a fixed axis instead of the whole plot rescaling on every
   * frame.  Afterwards (and for a container that declared no length) the axis is
   * what was actually analysed.
   */
  function axisFrames() {
    const analysed = state.result ? state.result.timeline.length : 0;
    const expected = state.mode === "live" ? state.expectedFrames : 0;
    return Math.max(analysed, expected, 1);
  }

  function resizeAll() {
    if (!state.result) return;
    buildZoneStrip(state.result);
    state.charts.forEach(renderChartStatic);
    drawPlayheads(state.frame < 0 ? 0 : state.frame);
  }

  function drawPlayheads(frame) {
    if (state.strip) blitWithPlayhead(state.strip, frame);
    state.charts.forEach((chart) => blitWithPlayhead(chart, frame));
  }

  function blitWithPlayhead(view, frame) {
    if (!view.buffer) return;
    const { ctx, width, height, dpr } = view;
    ctx.clearRect(0, 0, width, height);
    ctx.drawImage(view.buffer, 0, 0, view.buffer.width / dpr, view.buffer.height / dpr);
    const x = view.plotX + view.plotW * (frame / Math.max(view.frames - 1, 1));
    ctx.save();
    ctx.strokeStyle = COLORS.playhead;
    ctx.globalAlpha = 0.9;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(Math.round(x) + 0.5, view.plotY);
    ctx.lineTo(Math.round(x) + 0.5, view.plotY + view.plotH);
    ctx.stroke();
    ctx.restore();
  }

  // ── Zone strip ────────────────────────────────────────────────────────────
  /**
   * A state-over-time band per risk channel: "which zone was this channel in at
   * this moment".  Bands (not lines) because the value is categorical, and the
   * three fills are the pipeline's own zone colours so the strip reads as the
   * same signal the aura in the video shows.
   */
  function buildZoneStrip(r) {
    const canvas = $("zone-strip");
    const channels = Object.keys(r.zones.counts);
    // Hide the whole figure, caption included: a heading over nothing reads as
    // a strip that failed to draw rather than one with nothing to draw yet.
    const figure = canvas.closest(".strip-figure");
    if (!channels.length) {
      figure.hidden = true;
      state.strip = null;
      return;
    }
    figure.hidden = false;
    canvas.hidden = false;

    // `overall` is the aggregate the border and audio act on — it leads.
    channels.sort((a, b) => (a === "overall" ? -1 : b === "overall" ? 1 : a.localeCompare(b)));

    const rowH = 16, gap = 4, padTop = 14, padBottom = 18, labelW = 118;
    const height = padTop + channels.length * (rowH + gap) + padBottom;
    const view = setupCanvas(canvas, height);
    const { buffer, ctx } = makeBuffer(view.width, height, view.dpr);

    const plotX = labelW, plotW = Math.max(view.width - labelW - 8, 10);
    const frames = axisFrames();

    ctx.fillStyle = COLORS.surface;
    ctx.fillRect(0, 0, view.width, height);

    ctx.font = "11px " + getComputedStyle(document.body).fontFamily;
    ctx.textBaseline = "middle";

    channels.forEach((channel, row) => {
      const y = padTop + row * (rowH + gap);
      ctx.fillStyle = COLORS.muted;
      ctx.textAlign = "right";
      ctx.fillText(channel.replace(/_/g, " "), labelW - 10, y + rowH / 2);

      ctx.fillStyle = "#1e2024";
      ctx.fillRect(plotX, y, plotW, rowH);

      // Collapse equal-zone runs into single rects: thousands of 1px fills is
      // both slow and visibly seamed at fractional pixel widths.
      let runStart = 0, runZone = zoneAt(r, 0, channel);
      for (let f = 1; f <= frames; f++) {
        const zone = f < frames ? zoneAt(r, f, channel) : null;
        if (zone !== runZone || f === frames) {
          if (runZone) {
            const x0 = plotX + plotW * (runStart / frames);
            const x1 = plotX + plotW * (f / frames);
            ctx.fillStyle = COLORS.zone[runZone];
            ctx.fillRect(x0, y, Math.max(x1 - x0, 1), rowH);
          }
          runStart = f;
          runZone = zone;
        }
      }
    });

    // Cue ticks along the bottom rule, so cause and coaching line up visually.
    const cueY = padTop + channels.length * (rowH + gap) + 2;
    ctx.strokeStyle = COLORS.axis;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(plotX, cueY + 0.5);
    ctx.lineTo(plotX + plotW, cueY + 0.5);
    ctx.stroke();
    r.cues.forEach((cue) => {
      const x = plotX + plotW * (cue.frame / Math.max(frames - 1, 1));
      ctx.strokeStyle = cue.category === "encouragement"
        ? COLORS.band.gate : COLORS.band.yellow;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(x, cueY);
      ctx.lineTo(x, cueY + 8);
      ctx.stroke();
    });
    ctx.fillStyle = COLORS.muted;
    ctx.textAlign = "right";
    ctx.fillText("cues", labelW - 10, cueY + 5);

    state.strip = {
      ctx: view.ctx, width: view.width, height, dpr: view.dpr, buffer,
      plotX, plotW, plotY: 0, plotH: height, frames,
    };

    canvas.onclick = (event) => {
      const rect = canvas.getBoundingClientRect();
      const ratio = (event.clientX - rect.left - plotX) / plotW;
      seekToFrame(Math.round(clamp(ratio, 0, 1) * (frames - 1)));
    };

    $("zone-legend").innerHTML = [
      ["green", "good"], ["yellow", "warning"], ["red", "high risk"],
    ].map(([zone, label]) =>
      `<span class="legend-item"><span class="legend-swatch"
        style="background:${COLORS.zone[zone]}"></span>${label}</span>`
    ).join("") +
      `<span class="legend-item"><span class="legend-line"
        style="background:${COLORS.band.yellow}"></span>audio cue</span>`;

    blitWithPlayhead(state.strip, state.frame < 0 ? 0 : state.frame);
  }

  function zoneAt(r, frame, channel) {
    const zones = r.timeline[frame] && r.timeline[frame].zones;
    return (zones && zones[channel]) || null;
  }

  // ── Line charts ───────────────────────────────────────────────────────────
  function buildCharts(specs) {
    const host = $("charts");
    host.innerHTML = "";
    state.charts = [];

    if (!specs || !specs.length) {
      $("charts-panel").hidden = true;
      return;
    }
    $("charts-panel").hidden = false;

    specs.forEach((spec) => {
      const block = document.createElement("div");
      block.className = "chart-block";
      block.innerHTML = `
        <div class="chart-head">
          <span class="chart-title"></span>
          <span class="chart-unit"></span>
          <span class="legend"></span>
          <span class="chart-note"></span>
        </div>
        <canvas class="chart-canvas"></canvas>
        <div class="chart-tooltip" hidden></div>`;
      block.querySelector(".chart-title").textContent = spec.title;
      block.querySelector(".chart-unit").textContent = spec.unit;
      block.querySelector(".legend").innerHTML = spec.series.map((s, i) =>
        `<span class="legend-item"><span class="legend-line"
          style="background:${COLORS.series[i % COLORS.series.length]}"></span>${s.label}</span>`
      ).join("");
      host.appendChild(block);

      const chart = { spec, block, canvas: block.querySelector(".chart-canvas"),
                      tooltip: block.querySelector(".chart-tooltip") };
      state.charts.push(chart);
      renderChartStatic(chart);
      wireChartHover(chart);
    });
  }

  function renderChartStatic(chart) {
    const r = state.result;
    if (!r) return;
    const spec = chart.spec;
    const height = 168;
    const view = setupCanvas(chart.canvas, height);
    const { buffer, ctx } = makeBuffer(view.width, height, view.dpr);

    // padR is the right gutter the threshold labels are written into; it fits a
    // merged two-rule label ("stand / desc") at the 10px label size.
    const padL = 46, padR = 88, padT = 10, padB = 20;
    const plotX = padL, plotY = padT;
    const plotW = Math.max(view.width - padL - padR, 10);
    const plotH = Math.max(height - padT - padB, 10);
    const frames = axisFrames();

    // Series values, with gaps where no pose was detected.
    const series = spec.series.map((s) =>
      r.timeline.map((entry) => {
        const v = entry.metrics ? entry.metrics[s.key] : undefined;
        return typeof v === "number" ? v : null;
      }));

    let { min, max } = robustExtent(series);
    // Threshold rules are always inside the extent — a boundary the data never
    // approaches is exactly the fact the chart should show.
    spec.bands.forEach((band) => {
      if (band.value < min) min = band.value;
      if (band.value > max) max = band.value;
    });
    if (max - min < 1e-6) { max += 0.5; min -= 0.5; }
    const pad = (max - min) * 0.08;
    min -= pad; max += pad;

    let clipped = 0;
    series.forEach((values) => values.forEach((v) => {
      if (v !== null && (v < min || v > max)) clipped++;
    }));

    const xOf = (i) => plotX + plotW * (i / Math.max(frames - 1, 1));
    const yOf = (v) => plotY + plotH * (1 - (v - min) / (max - min));

    ctx.fillStyle = COLORS.surface;
    ctx.fillRect(0, 0, view.width, height);

    // Recessive hairline grid + y labels.
    ctx.font = "10px " + getComputedStyle(document.body).fontFamily;
    ctx.textBaseline = "middle";
    ctx.textAlign = "right";
    for (let i = 0; i <= 3; i++) {
      const value = min + (max - min) * (i / 3);
      const y = Math.round(yOf(value)) + 0.5;
      ctx.strokeStyle = COLORS.grid;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(plotX, y);
      ctx.lineTo(plotX + plotW, y);
      ctx.stroke();
      ctx.fillStyle = COLORS.muted;
      ctx.fillText(formatTick(value), plotX - 8, y);
    }

    // Threshold rules, labelled at the right margin.  Rules closer together
    // than a line height share ONE merged label ("standing / descending") — the
    // squat's rep-counter gates sit 15-20 deg apart and would otherwise
    // overprint.  Dropping the second label instead would leave whichever
    // survived looking like the only rule there.
    const inRange = [...spec.bands]
      .filter((band) => band.value >= min && band.value <= max)
      .sort((a, b) => b.value - a.value);
    const clusters = [];
    inRange.forEach((band) => {
      const y = Math.round(yOf(band.value)) + 0.5;
      const color = COLORS.band[band.severity] || COLORS.muted;
      ctx.strokeStyle = color;
      ctx.globalAlpha = band.severity === "gate" ? 0.5 : 0.85;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(plotX, y);
      ctx.lineTo(plotX + plotW, y);
      ctx.stroke();
      ctx.globalAlpha = 1;

      const last = clusters[clusters.length - 1];
      if (last && y - last.y < 11) last.labels.push(band.label);
      else clusters.push({ y, color, labels: [band.label] });
    });
    ctx.textAlign = "left";
    clusters.forEach((cluster) => {
      ctx.fillStyle = cluster.color;
      ctx.fillText(cluster.labels.join(" / "), plotX + plotW + 6, cluster.y);
    });
    ctx.textAlign = "right";

    // Traces, clipped to the plot rectangle so an off-scale excursion runs to
    // the edge instead of scribbling over the axes and labels.
    ctx.save();
    ctx.beginPath();
    ctx.rect(plotX, plotY, plotW, plotH);
    ctx.clip();
    series.forEach((values, i) => {
      ctx.strokeStyle = COLORS.series[i % COLORS.series.length];
      ctx.lineWidth = 2;
      ctx.lineJoin = "round";
      ctx.beginPath();
      let drawing = false;
      values.forEach((v, idx) => {
        if (v === null) { drawing = false; return; }
        const x = xOf(idx), y = yOf(v);
        if (drawing) ctx.lineTo(x, y);
        else { ctx.moveTo(x, y); drawing = true; }
      });
      ctx.stroke();
    });
    ctx.restore();

    // X axis: time ticks.
    ctx.strokeStyle = COLORS.axis;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(plotX, plotY + plotH + 0.5);
    ctx.lineTo(plotX + plotW, plotY + plotH + 0.5);
    ctx.stroke();
    ctx.fillStyle = COLORS.muted;
    ctx.textAlign = "center";
    for (let i = 0; i <= 4; i++) {
      const frame = Math.round((frames - 1) * (i / 4));
      ctx.fillText(timecode(frame / state.fps), xOf(frame), plotY + plotH + 11);
    }

    const note = chart.block.querySelector(".chart-note");
    note.textContent = clipped
      ? `${clipped} off-scale point${clipped > 1 ? "s" : ""} clipped`
      : "";

    Object.assign(chart, {
      ctx: view.ctx, width: view.width, height, dpr: view.dpr, buffer,
      plotX, plotY, plotW, plotH, frames, series, min, max,
    });
  }

  /**
   * A y-range that survives tracking glitches.
   *
   * A handful of frames where MediaPipe loses a limb send a metric to values
   * hundreds of times its normal magnitude — a trunk deviation of 200 deg on a
   * trace that otherwise lives inside +/-10.  Scaling to the raw min/max lets
   * those few frames flatten the entire signal into a horizontal line, and the
   * thresholds the chart exists to compare against collapse onto it.
   *
   * So the extent is Tukey whiskers at k=3 — far outside the normal spread, so
   * genuine extremes (the bottom of a deep squat) are kept — intersected with
   * the real min/max so the range is never wider than the data.  Anything still
   * outside is drawn clipped and counted in the caption, never silently hidden.
   */
  function robustExtent(series) {
    const values = [];
    series.forEach((arr) => arr.forEach((v) => {
      if (v !== null && isFinite(v)) values.push(v);
    }));
    if (!values.length) return { min: 0, max: 1 };
    values.sort((a, b) => a - b);
    const at = (q) => values[clamp(Math.floor(q * (values.length - 1)), 0, values.length - 1)];
    const q1 = at(0.25), q3 = at(0.75), iqr = q3 - q1;
    return {
      min: Math.max(values[0], q1 - 3 * iqr),
      max: Math.min(values[values.length - 1], q3 + 3 * iqr),
    };
  }

  function formatTick(value) {
    const abs = Math.abs(value);
    if (abs >= 100) return value.toFixed(0);
    if (abs >= 10) return value.toFixed(1);
    return value.toFixed(2);
  }

  function wireChartHover(chart) {
    const canvas = chart.canvas;

    canvas.addEventListener("mousemove", (event) => {
      const rect = canvas.getBoundingClientRect();
      const ratio = (event.clientX - rect.left - chart.plotX) / chart.plotW;
      const frame = Math.round(clamp(ratio, 0, 1) * (chart.frames - 1));
      const values = chart.spec.series.map((s, i) => {
        const v = chart.series[i] ? chart.series[i][frame] : undefined;
        return `${s.label}: ${v === null || v === undefined ? "—" : v.toFixed(2)}`;
      });
      const tip = chart.tooltip;
      tip.innerHTML = `<strong>${timecode(frame / state.fps)}</strong><br>${values.join("<br>")}`;
      tip.hidden = false;
      const left = clamp(event.clientX - rect.left + 12, 0,
                         Math.max(rect.width - tip.offsetWidth - 4, 0));
      tip.style.left = `${left}px`;
      tip.style.top = `${canvas.offsetTop + 8}px`;
    });

    canvas.addEventListener("mouseleave", () => { chart.tooltip.hidden = true; });
    canvas.addEventListener("click", (event) => {
      const rect = canvas.getBoundingClientRect();
      const ratio = (event.clientX - rect.left - chart.plotX) / chart.plotW;
      seekToFrame(Math.round(clamp(ratio, 0, 1) * (chart.frames - 1)));
    });
  }

  function debounce(fn, ms) {
    let timer = null;
    return (...args) => {
      clearTimeout(timer);
      timer = setTimeout(() => fn(...args), ms);
    };
  }

  boot();
})();
