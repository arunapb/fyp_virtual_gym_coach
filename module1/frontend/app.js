const WS_URL = `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/session`;

const els = {
  stop: document.getElementById("btn-stop"),
  fileInput: document.getElementById("file-video"),
  connStatus: document.getElementById("conn-status"),
  previewImg: document.getElementById("preview-img"),
  stageLoading: document.getElementById("stage-loading"),
  stageLoadingText: document.getElementById("stage-loading-text"),
  stagePlaceholder: document.getElementById("stage-placeholder"),
  state: document.getElementById("state-value"),
  exercise: document.getElementById("exercise-value"),
  confidence: document.getElementById("confidence-value"),
  elapsed: document.getElementById("elapsed-value"),
  rawLabel: document.getElementById("raw-label-value"),
  sessionTime: document.getElementById("session-time-value"),
  frameCount: document.getElementById("frame-count-value"),
  probActiveBar: document.getElementById("prob-active-bar"),
  probActiveValue: document.getElementById("prob-active-value"),
  probRestBar: document.getElementById("prob-rest-bar"),
  probRestValue: document.getElementById("prob-rest-value"),
  probNullBar: document.getElementById("prob-null-bar"),
  probNullValue: document.getElementById("prob-null-value"),
  summarySection: document.getElementById("summary-section"),
  summaryTable: document.getElementById("summary-table"),
  summaryJson: document.getElementById("summary-json"),
};

let ws = null;
let captureTimer = null;
let sessionTimer = null;
let sessionStartTime = null;

function setConnStatus(text, cls) {
  els.connStatus.textContent = text;
  els.connStatus.className = `badge badge-${cls}`;
}

function resetReadout() {
  els.state.textContent = "—";
  els.state.className = "tile-value";
  els.exercise.textContent = "—";
  els.confidence.textContent = "—";
  els.elapsed.textContent = "—";
  els.rawLabel.textContent = "—";
  els.sessionTime.textContent = "—";
  els.frameCount.textContent = "—";
  setProbBar(els.probActiveBar, els.probActiveValue, null);
  setProbBar(els.probRestBar, els.probRestValue, null);
  setProbBar(els.probNullBar, els.probNullValue, null);
  els.summarySection.hidden = true;
  hideStageLoading();
}

function showStageLoading(text) {
  els.stagePlaceholder.hidden = true;
  els.previewImg.hidden = true;
  els.stageLoadingText.textContent = text;
  els.stageLoading.hidden = false;
}

function hideStageLoading() {
  els.stageLoading.hidden = true;
}

function setProbBar(barEl, valueEl, prob) {
  if (prob === null || prob === undefined) {
    barEl.style.width = "0%";
    valueEl.textContent = "—";
    return;
  }
  const pct = prob * 100;
  barEl.style.width = `${pct.toFixed(0)}%`;
  valueEl.textContent = `${pct.toFixed(0)}%`;
}

function startSessionTimer() {
  sessionStartTime = Date.now();
  stopSessionTimer();
  sessionTimer = setInterval(() => {
    const secs = (Date.now() - sessionStartTime) / 1000;
    els.sessionTime.textContent = `${secs.toFixed(0)}s`;
  }, 1000);
}

function stopSessionTimer() {
  if (sessionTimer) {
    clearInterval(sessionTimer);
    sessionTimer = null;
  }
}

function applyStateUpdate(update) {
  els.state.textContent = update.state.toUpperCase();
  els.state.className = `tile-value state-${update.state}`;
  els.exercise.textContent = update.exercise
    ? update.exercise.replace("_", " ")
    : (update.label === "VERIFYING..." ? "verifying…" : "—");
  els.confidence.textContent = update.confidence ? `${update.confidence.toFixed(1)}%` : "—";
  els.elapsed.textContent = update.exercise ? `${update.elapsed_active_sec.toFixed(1)}s` : "—";
  els.rawLabel.textContent = update.label || "—";
  els.frameCount.textContent = update.frame_index ?? "—";

  if (update.state_probs) {
    setProbBar(els.probActiveBar, els.probActiveValue, update.state_probs.active);
    setProbBar(els.probRestBar, els.probRestValue, update.state_probs.rest);
    setProbBar(els.probNullBar, els.probNullValue, update.state_probs.null);
  }

  // Video mode: the server echoes back the frame it just analyzed (skeleton
  // overlay drawn in), since the browser can't reliably play an arbitrary
  // uploaded file's native codec/container.
  if (update.frame) {
    hideStageLoading();
    els.previewImg.src = `data:image/jpeg;base64,${update.frame}`;
    els.previewImg.hidden = false;
  }
}

function showSummary(summary) {
  els.summarySection.hidden = false;
  const rows = Object.entries(summary.durations)
    .map(([name, secs]) => `<tr><td>${name.replace("_", " ")}</td><td>${secs.toFixed(2)}s</td></tr>`)
    .join("");
  els.summaryTable.innerHTML = `
    <tr><th>Exercise</th><th>Duration</th></tr>
    ${rows || `<tr><td colspan="2">No exercise detected this session.</td></tr>`}
    <tr><td><strong>Rest time</strong></td><td>${summary.rest_time.toFixed(2)}s</td></tr>
    <tr><td><strong>Total session time</strong></td><td>${summary.total_time.toFixed(2)}s</td></tr>
  `;
  els.summaryJson.textContent = JSON.stringify(summary, null, 2);
}

function stopCaptureLoop() {
  stopSessionTimer();
  if (captureTimer) {
    clearInterval(captureTimer);
    captureTimer = null;
  }
}

function closeSocket() {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.close();
  }
  ws = null;
}

function onSessionEnded() {
  stopCaptureLoop();
  hideStageLoading();
  els.stop.disabled = true;
  els.fileInput.disabled = false;
  setConnStatus("idle", "idle");
}

function openSocket(onOpen) {
  ws = new WebSocket(WS_URL);
  ws.onopen = () => {
    setConnStatus("connected", "connected");
    onOpen();
  };
  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    if (msg.type === "state_update") {
      applyStateUpdate(msg);
    } else if (msg.type === "summary") {
      showSummary(msg.summary);
      onSessionEnded();
    } else if (msg.type === "error") {
      console.error("Server error:", msg.message);
      setConnStatus(`error: ${msg.message}`, "error");
      onSessionEnded();
    }
  };
  ws.onerror = () => setConnStatus("connection error", "error");
  ws.onclose = () => onSessionEnded();
}

// ---- Video upload mode ------------------------------------------------------

async function startVideoSession(file) {
  resetReadout();
  els.fileInput.disabled = true;
  setConnStatus("uploading…", "idle");

  // Don't try to play the raw upload in a native <video> tag — the browser's
  // demuxer often can't handle it even when OpenCV can decode it fine
  // server-side. Instead we show the analyzed frames the server sends back
  // (see applyStateUpdate's handling of update.frame).
  els.previewImg.hidden = true;
  els.previewImg.src = "";
  showStageLoading("Uploading");

  const formData = new FormData();
  formData.append("file", file);

  let videoId;
  try {
    const res = await fetch("/api/videos", { method: "POST", body: formData });
    if (!res.ok) throw new Error(await res.text());
    ({ video_id: videoId } = await res.json());
  } catch (err) {
    setConnStatus(`upload failed: ${err.message}`, "error");
    els.fileInput.disabled = false;
    hideStageLoading();
    return;
  }

  showStageLoading("Processing");

  openSocket(() => {
    ws.send(JSON.stringify({ type: "start", mode: "video", video_id: videoId }));
    els.stop.disabled = false;
    startSessionTimer();
  });
}

// ---- Controls ---------------------------------------------------------------

els.fileInput.addEventListener("change", (e) => {
  const file = e.target.files[0];
  e.target.value = "";
  if (file) startVideoSession(file);
});

els.stop.addEventListener("click", () => {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: "stop" }));
  }
  els.stop.disabled = true;
});
