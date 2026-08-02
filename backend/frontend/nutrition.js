/* =============================================================================
   Nutrition page (Module 4) — NutriIngredientNet client.

   Module 4's original demo.html carried this inline. Behaviour is unchanged:
   drop/choose a photo -> preview -> stepper overlay -> POST -> macro cards
   count up and the ingredient rows fade in staggered.

   Two changes for the shared server:
     * the endpoint is namespaced: /api/nutrition/predict (was /predict, which
       would have collided with the workout dashboard's own routes);
     * results render into the shared theme's .tile / table markup instead of
       Module 4's own .mc / .tbl-wrap classes.
   ============================================================================= */

(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const PREDICT_URL = "/api/nutrition/predict";

  const drop = $("drop");
  const fileInput = $("file-input");
  const prevWrap = $("preview-wrap");
  const preview = $("preview");
  const changeBtn = $("change-btn");
  const statusEl = $("status");
  const resultsEl = $("results");
  const overlay = $("overlay");

  let timer = null;

  // ── Input ───────────────────────────────────────────────────────────────
  drop.addEventListener("click", () => fileInput.click());
  drop.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); }
  });
  changeBtn.addEventListener("click", () => fileInput.click());

  ["dragover", "dragenter"].forEach((evt) =>
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
    const f = e.dataTransfer.files[0];
    if (f) run(f);
  });
  fileInput.addEventListener("change", () => {
    if (fileInput.files[0]) run(fileInput.files[0]);
  });

  // ── Overlay ─────────────────────────────────────────────────────────────
  function stepSet(id, state) {
    const el = $(id);
    el.classList.remove("active", "done");
    if (state) el.classList.add(state);
    if (state === "done") el.querySelector(".si").textContent = "✓";
  }

  function resetSteps() {
    ["s1", "s2", "s3"].forEach((id, i) => {
      const el = $(id);
      el.classList.remove("active", "done");
      el.querySelector(".si").textContent = String(i + 1);
    });
  }

  function showOverlay(msg) {
    $("ov-sub").textContent = msg;
    overlay.classList.add("on");
    document.body.style.overflow = "hidden";
    const t0 = Date.now();
    $("elapsed").textContent = "0.0 s";
    timer = setInterval(() => {
      $("elapsed").textContent = ((Date.now() - t0) / 1000).toFixed(1) + " s";
    }, 100);
  }

  function hideOverlay() {
    overlay.classList.remove("on");
    document.body.style.overflow = "";
    clearInterval(timer);
  }

  function setStatus(html) { statusEl.innerHTML = html; }

  function setError(msg) {
    hideOverlay();
    setStatus(`<span class="err">⚠ ${msg}</span>`);
  }

  function escapeHtml(text) {
    return String(text).replace(/[&<>"]/g, (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  }

  // ── Count-up ────────────────────────────────────────────────────────────
  function countUp(el, target, ms = 700) {
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      el.textContent = Number.isInteger(target) ? target : target.toFixed(1);
      return;
    }
    const t0 = performance.now();
    const step = (now) => {
      const p = Math.min((now - t0) / ms, 1);
      const ease = 1 - Math.pow(1 - p, 3);
      const v = target * ease;
      el.textContent = Number.isInteger(target) ? Math.round(v) : v.toFixed(1);
      if (p < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }

  // ── Flow ────────────────────────────────────────────────────────────────
  async function run(file) {
    preview.src = URL.createObjectURL(file);
    prevWrap.hidden = false;
    drop.hidden = true;
    resultsEl.hidden = true;
    setStatus("");

    resetSteps();
    showOverlay("Sending image to the analyser…");
    stepSet("s1", "active");

    const fd = new FormData();
    fd.append("file", file);

    let response;
    try {
      response = await fetch(PREDICT_URL, { method: "POST", body: fd });
    } catch (err) {
      setError("Cannot reach the analyser — is the server still running?");
      drop.hidden = false;
      return;
    }

    stepSet("s1", "done");

    if (!response.ok) {
      let detail = response.statusText;
      try {
        const body = await response.json();
        detail = body.detail || detail;
      } catch (e) { /* non-JSON error body; keep the status text */ }
      setError(`Analyser error ${response.status}: ${escapeHtml(detail)}`);
      drop.hidden = false;
      return;
    }

    stepSet("s2", "active");
    $("ov-sub").textContent = "Parsing model output…";
    let data;
    try {
      data = await response.json();
    } catch (err) {
      setError("Failed to parse the analyser response: " + escapeHtml(err.message));
      drop.hidden = false;
      return;
    }
    stepSet("s2", "done");

    stepSet("s3", "active");
    $("ov-sub").textContent = "Building results…";
    render(data);
    stepSet("s3", "done");

    await new Promise((r) => setTimeout(r, 300));
    hideOverlay();
  }

  // ── Render ──────────────────────────────────────────────────────────────
  function render(d) {
    const t = d.totals;

    const macros = [
      { cls: "cal", val: t.calories, unit: "kcal", label: "Calories" },
      { cls: "fat", val: t.fat_g, unit: "g", label: "Fat" },
      { cls: "carb", val: t.carbs_g, unit: "g", label: "Carbs" },
      { cls: "prot", val: t.protein_g, unit: "g", label: "Protein" },
    ];
    $("macros-grid").innerHTML = macros.map((m) => `
      <div class="tile fade-up">
        <div class="tile-label">${m.label}</div>
        <div class="tile-value ${m.cls}"><span id="mv-${m.cls}">0</span><span
             class="tile-unit">${m.unit}</span></div>
      </div>`).join("");
    macros.forEach((m) => countUp($("mv-" + m.cls), m.val));

    const atw = d.atwater_check;
    $("meta-row").innerHTML =
      `<span>Total mass <strong>${t.mass_g} g</strong></span>
       <span class="meta-sep">|</span>
       <span>Calorie check: <strong>${atw.from_macros}</strong> kcal</span>
       <span class="atwater-badge">${atw.consistent ? "✓ Atwater exact" : "⚠ Mismatch"}</span>`;

    $("ing-body").innerHTML = d.ingredients.map((i, idx) => `
      <tr class="${i.status === "possible" ? "possible" : ""}"
          style="animation: nutri-fade-up .35s ease ${idx * 40}ms both">
        <td><span class="iname">${escapeHtml(i.name)}<span
            class="pill ${i.status === "certain" ? "cert" : "poss"}">${i.status}</span></span></td>
        <td>${i.grams}</td>
        <td>${i.kcal}</td>
        <td>${i.fat}</td>
        <td>${i.carb}</td>
        <td>${i.protein}</td>
      </tr>`).join("");

    // `d.warning` (the fixed-camera-rig / not-medical-advice notice) is still
    // returned by the API; it is deliberately not rendered.
    $("acc-box").textContent =
      `Typical error on the Nutrition5k test split: ±${d.accuracy.test_kcal_mae} kcal `
      + `(${d.accuracy.test_kcal_mae_pct}%). ${d.accuracy.note}`;

    resultsEl.hidden = false;
  }
})();
