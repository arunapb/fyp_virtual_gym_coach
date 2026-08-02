/* =============================================================================
   Meal plan page (Module 3) — client for the BM25 + FAISS recommender, the
   NL-feedback pipeline and the goal/pace calculator.

   Module 3's test_frontend/index.html carried this inline. Behaviour is
   unchanged: load the profile, fetch a plan, edit profile/goal in a modal,
   submit feedback, watch preferences move. Differences, all forced by the
   shared server:

     * endpoints are namespaced under /api/meals (its bare /profile, /health
       and /recommend would have collided with the workout dashboard's routes)
       and same-origin, so the hardcoded http://localhost:8000 is gone;
     * inline `onclick=` handlers are replaced with listeners registered here,
       so nothing on this page depends on globals hanging off `window`;
     * the "Logged workouts" panel is new — it shows the exercise log that
       every finished workout analysis now writes (backend/module3/
       exercise_log.py), which is what moves the "Exercise avg (7d)" number.

   One deliberate addition: the first /recommend of a session can take tens of
   seconds while Module 3 loads its FAISS index and SentenceTransformer (they
   load on first use, not at server boot — see backend/module3/bridge.py), so
   requests show a busy state instead of an empty panel.
   ============================================================================= */

(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const API = "/api/meals";

  const ACTIVITY_LABELS = {
    sedentary: "Sedentary",
    lightly_active: "Lightly active",
    moderately_active: "Moderately active",
    very_active: "Very active",
  };
  const GOAL_LABELS = {
    weight_loss: "Lose weight",
    maintenance: "Maintain weight",
    weight_gain: "Gain weight",
  };
  const PACE_LABELS = { gentle: "Gentle", balanced: "Balanced", faster: "Faster" };
  const SLOT_ORDER = ["Breakfast", "Lunch", "Dinner"];
  const MAX_TAGS = 6;          // ingredient chips shown before "+N more"

  let cachedProfile = null;
  let toastTimer = null;

  // ── Utilities ────────────────────────────────────────────────────────────
  function esc(text) {
    return String(text ?? "").replace(/[&<>"]/g, (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  }

  function toast(message, bad = false) {
    const el = $("toast");
    el.textContent = message;
    el.classList.toggle("bad", bad);
    el.classList.add("show");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => el.classList.remove("show"), 3500);
  }

  /** fetch + JSON + a usable error message. FastAPI puts its reason in `detail`. */
  async function api(path, options) {
    const response = await fetch(API + path, options);
    if (!response.ok) {
      let detail = `${response.status} ${response.statusText}`;
      try {
        const body = await response.json();
        if (body.detail) detail = body.detail;
      } catch (e) { /* non-JSON error body; keep the status line */ }
      throw new Error(detail);
    }
    return response.status === 204 ? null : response.json();
  }

  function jsonBody(payload) {
    return {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    };
  }

  function postBody(payload) {
    return { ...jsonBody(payload), method: "POST" };
  }

  function busy(container, message) {
    container.innerHTML =
      `<div class="busy-note"><span class="busy-spinner"></span>${esc(message)}</div>`;
  }

  // ── Health ───────────────────────────────────────────────────────────────
  async function checkHealth() {
    const badge = $("api-status");
    try {
      const data = await api("/health");
      badge.className = "conn ok";
      $("status-text").textContent = "Connected";
      badge.title = data.message || "";
    } catch (err) {
      badge.className = "conn down";
      $("status-text").textContent = "Server offline";
      badge.title = err.message;
    }
  }

  // ── Profile ──────────────────────────────────────────────────────────────
  async function loadProfile() {
    try {
      cachedProfile = await api("/profile");
      renderProfileSummary(cachedProfile);
    } catch (err) {
      toast("Could not load your profile: " + err.message, true);
    }
  }

  function renderProfileSummary(p) {
    $("profile-summary-weight").textContent =
      p.weight_kg != null ? `${p.weight_kg} kg` : "—";
    $("profile-summary-height").textContent =
      p.height_cm != null ? `${p.height_cm} cm` : "—";
    $("profile-summary-activity").textContent =
      ACTIVITY_LABELS[p.activity_level] || p.activity_level || "—";

    const goalEl = $("profile-summary-goal");
    if (p.goal_weight_kg == null || p.weight_kg == null) {
      goalEl.textContent = GOAL_LABELS[p.goal] || p.goal || "—";
      return;
    }
    if (p.already_at_goal || Math.abs(p.kg_to_change ?? 0) < 0.01) {
      goalEl.textContent = `At goal (${p.goal_weight_kg} kg)`;
      return;
    }
    // is_loss / kg_to_change / estimated_weeks are computed live server-side
    // by goal_service, and work identically for gain and loss.
    const verb = p.is_loss ? "Lose" : "Gain";
    const weeks = p.estimated_weeks != null
      ? `<span class="tile-sub">~${p.estimated_weeks} wks · ${
          esc(PACE_LABELS[p.selected_pace] || p.selected_pace || "")}</span>`
      : "";
    goalEl.innerHTML =
      `${verb} ${esc(p.kg_to_change)} kg → ${esc(p.goal_weight_kg)} kg${weeks}`;
  }

  // ── Profile modal ────────────────────────────────────────────────────────
  function openModal(id) {
    $(id).hidden = false;
    document.body.style.overflow = "hidden";
  }

  function closeModal(id) {
    $(id).hidden = true;
    document.body.style.overflow = "";
  }

  function openProfileModal() {
    const p = cachedProfile || {};
    $("modal-weight").value = p.weight_kg ?? "";
    $("modal-height").value = p.height_cm ?? "";
    $("modal-activity").value = p.activity_level || "sedentary";
    $("modal-goal-weight").value = p.goal_weight_kg ?? "";
    openModal("profile-modal");
    weightsChanged();
  }

  /** Shows the pace picker once both weights are usable. */
  function weightsChanged() {
    const current = parseFloat($("modal-weight").value);
    const goal = parseFloat($("modal-goal-weight").value);
    if (!current || current <= 0 || !goal || goal <= 0) {
      $("modal-goal-section").hidden = true;
      return;
    }
    fetchGoalOptions(current, goal);
  }

  async function fetchGoalOptions(currentWeight, goalWeight) {
    try {
      // Push the typed weight first, so kg_to_change is computed against what
      // is on screen rather than the last saved value.
      await api("/profile", jsonBody({ weight_kg: currentWeight }));
      renderGoalPanel(await api("/goal/options", postBody({ goal_weight_kg: goalWeight })));
    } catch (err) {
      toast("Could not work out your goal: " + err.message, true);
    }
  }

  function renderGoalPanel(data) {
    const section = $("modal-goal-section");
    section.hidden = false;

    if (data.already_at_goal) {
      $("goal-summary").innerHTML = "<strong>You're already at your goal weight.</strong>";
      $("goal-pace-selector").innerHTML = "";
      $("goal-safe-note").hidden = true;
      return;
    }

    $("goal-safe-note").hidden = false;
    $("goal-summary").innerHTML =
      `You want to <strong>${data.is_loss ? "lose" : "gain"} ${esc(data.kg_to_change)} kg</strong>.`;

    $("goal-pace-selector").innerHTML = data.pace_options.map((opt) => `
      <label class="pace-option ${opt.is_default ? "selected" : ""}" data-pace-key="${esc(opt.key)}">
        <input type="radio" name="pace-option" value="${esc(opt.key)}"
               ${opt.is_default ? "checked" : ""}>
        <span>
          <span class="pace-label">${esc(opt.label)}</span>
          <span class="pace-meta">${opt.kcal_per_day} kcal/day &middot;
            ~${opt.estimated_weeks} weeks (${opt.estimated_months} months)</span>
        </span>
      </label>`).join("");

    $("goal-pace-selector").querySelectorAll("input[name='pace-option']")
      .forEach((input) => input.addEventListener("change", () => selectPace(input.value)));
  }

  async function selectPace(pace) {
    document.querySelectorAll(".pace-option").forEach((el) => {
      el.classList.toggle("selected", el.dataset.paceKey === pace);
    });
    try {
      const data = await api("/goal/select-pace", postBody({ pace }));
      toast(`Pace set to ${PACE_LABELS[pace] || pace} — ${data.new_daily_target} kcal/day`);
      loadProfile();
      fetchRecommendations();
    } catch (err) {
      toast("Could not set that pace: " + err.message, true);
    }
  }

  async function saveProfile(event) {
    event.preventDefault();
    const weight_kg = parseFloat($("modal-weight").value);
    const height_cm = parseFloat($("modal-height").value);
    const activity_level = $("modal-activity").value;
    const goalRaw = $("modal-goal-weight").value;
    const goal_weight_kg = goalRaw ? parseFloat(goalRaw) : null;

    if (!weight_kg || weight_kg <= 0 || !height_cm || height_cm <= 0) {
      toast("Enter a valid weight and height.", true);
      return;
    }

    try {
      await api("/profile", jsonBody({ weight_kg, height_cm, activity_level }));
      if (goal_weight_kg && goal_weight_kg > 0) {
        await api("/goal/options", postBody({ goal_weight_kg }));
      }
      toast("Profile updated.");
      closeModal("profile-modal");
      await loadProfile();
      fetchRecommendations();
    } catch (err) {
      toast("Could not save your profile: " + err.message, true);
    }
  }

  // ── Recommendations ──────────────────────────────────────────────────────
  async function fetchRecommendations() {
    const container = $("meal-plan-container");
    busy(container, "Building your meal plan… the first one also loads the "
                  + "retrieval index, so give it a moment.");
    try {
      const topN = $("top-n-select").value || 3;
      const data = await api(`/recommend?top_n=${encodeURIComponent(topN)}`);
      renderTargets(data.nutrition_targets);
      renderMealPlan(data.meal_plan, container);
      fetchPreferences();
    } catch (err) {
      container.innerHTML =
        `<p class="error">Could not load recommendations: ${esc(err.message)}</p>`;
      toast("Could not load recommendations.", true);
    }
  }

  function renderTargets(nt) {
    $("stat-cal").textContent = `${nt.total_calories} kcal`;
    $("stat-prot").textContent = `${nt.total_protein} g`;
    $("stat-bmr").textContent = `${Math.round(nt.bmr)} kcal`;
    $("stat-ex").textContent = `+${nt.exercise_avg_7} kcal`;

    const targets = nt.meal_targets || {};
    const ids = { Breakfast: "target-breakfast", Lunch: "target-lunch", Dinner: "target-dinner" };
    SLOT_ORDER.forEach((slot) => {
      const el = $(ids[slot]);
      const t = targets[slot];
      el.innerHTML = t
        ? `${t.calories} kcal <small>· ${t.protein} g protein</small>`
        : "—";
    });
  }

  function renderMealPlan(mealPlan, container) {
    // Object key order is insertion order from the server, but the slots read
    // better in day order regardless of what came back.
    const slots = Object.keys(mealPlan).sort(
      (a, b) => SLOT_ORDER.indexOf(a) - SLOT_ORDER.indexOf(b));

    if (!slots.length) {
      container.innerHTML = `<p class="hint loading-note">No meals came back.</p>`;
      return;
    }

    container.innerHTML = slots.map((slot) => {
      const s = mealPlan[slot];
      return `
        <section class="meal-section">
          <div class="meal-head">
            <div class="meal-name">${esc(slot)}</div>
            <div class="meal-target">
              Target <strong>${s.target_calories} kcal</strong> ·
              <strong>${s.target_protein} g</strong> protein
            </div>
          </div>
          <div class="recipe-grid">${s.recipes.map(recipeCard).join("")}</div>
        </section>`;
    }).join("");
  }

  function recipeCard(r) {
    const ingredients = r.ingredients || [];
    const shown = ingredients.slice(0, MAX_TAGS)
      .map((i) => `<span class="ing-tag">${esc(i)}</span>`).join("");
    const rest = ingredients.length > MAX_TAGS
      ? `<span class="ing-tag more">+${ingredients.length - MAX_TAGS} more</span>` : "";

    const steps = (r.steps && r.steps.length)
      ? `<details class="recipe-steps">
           <summary>${r.steps.length} step${r.steps.length === 1 ? "" : "s"}</summary>
           <ol>${r.steps.map((s) => `<li>${esc(s)}</li>`).join("")}</ol>
         </details>`
      : "";

    // The recommender explains itself; the first two reasons are the ones
    // about preferences and calories, which is what a reader wants to see.
    const why = (r.explanation && r.explanation.length)
      ? `<div class="recipe-why">${esc(r.explanation.slice(0, 2).join(" · "))}</div>` : "";

    const rating = r.rating != null
      ? `<span class="recipe-rating">★ ${r.rating.toFixed(1)}</span>` : "";

    return `
      <article class="recipe-card">
        <div class="recipe-top">
          <div class="recipe-title">${esc(r.title)}</div>
          ${rating}
        </div>
        <div class="recipe-meta">
          <span class="kcal">${Math.round(r.calories)} kcal</span>
          <span class="prot">${Math.round(r.protein)} g protein</span>
        </div>
        <div class="ing-tags">${shown}${rest}</div>
        ${steps}
        ${why}
      </article>`;
  }

  async function clearHistoryAndFetch() {
    try {
      await api("/history", { method: "DELETE" });
      toast("Today's history cleared — picking fresh recipes.");
      fetchRecommendations();
    } catch (err) {
      toast("Could not clear history: " + err.message, true);
    }
  }

  // ── Preferences ──────────────────────────────────────────────────────────
  async function fetchPreferences() {
    const container = $("pref-container");
    try {
      const data = await api("/preferences");
      const prefs = data.preferences || {};
      const names = Object.keys(prefs).sort((a, b) => prefs[b] - prefs[a]);

      if (!names.length) {
        container.innerHTML =
          `<p class="hint">No preferences yet — leave some feedback on a meal.</p>`;
        return;
      }
      // Names only — the raw score is deliberately not shown. It still drives
      // the sort above (likes first, dislikes last) and the like/dislike split
      // still reads from the thumb and the chip's edge colour.
      container.innerHTML = names.map((name) => {
        const positive = prefs[name] >= 0;
        return `
          <div class="pref-chip ${positive ? "positive" : "negative"}">
            <span>${positive ? "👍" : "👎"} ${esc(name)}</span>
          </div>`;
      }).join("");
    } catch (err) {
      container.innerHTML = `<p class="error">${esc(err.message)}</p>`;
    }
  }

  async function resetPreferences() {
    if (!window.confirm("Reset every learned preference back to zero?")) return;
    try {
      await api("/preferences", { method: "DELETE" });
      toast("Preferences reset.");
      fetchRecommendations();
    } catch (err) {
      toast("Could not reset preferences: " + err.message, true);
    }
  }

  // ── Feedback ─────────────────────────────────────────────────────────────
  async function submitFeedback(event) {
    event.preventDefault();
    const input = $("modal-feedback-input");
    const text = input.value.trim();
    if (!text) {
      toast("Write some feedback first.", true);
      return;
    }

    const button = $("feedback-submit");
    button.disabled = true;
    button.textContent = "Analysing…";
    try {
      // The response carries `causal` and a per-ingredient `aspects` breakdown.
      // Neither is rendered — the effect shows up where it matters, in the
      // Active preferences panel and the regenerated plan behind the modal.
      await api("/feedback", postBody({ feedback_text: text }));
      input.value = "";
      toast("Feedback applied — regenerating your plan.");
      fetchRecommendations();
      closeModal("feedback-modal");
    } catch (err) {
      toast("Could not process that feedback: " + err.message, true);
    } finally {
      button.disabled = false;
      button.textContent = "Submit & update plan";
    }
  }

  // ── Logged workouts ──────────────────────────────────────────────────────
  async function fetchExerciseLog() {
    const container = $("log-container");
    try {
      const data = await api("/exercise");
      $("log-average").textContent = `${data.seven_day_average_calories} kcal`;

      const logs = (data.logs || []).slice().reverse();   // newest first
      if (!logs.length) {
        container.innerHTML = `<p class="hint">No workouts logged yet.</p>`;
        return;
      }
      const today = new Date().toISOString().slice(0, 10);
      container.innerHTML = logs.map((entry) => `
        <div class="log-row ${entry.date === today ? "today" : ""}">
          <span class="log-ex">${esc(prettyExercise(entry.exercise_name))}</span>
          <span class="log-kcal">${entry.calories_burned} kcal</span>
          <span class="log-when">${entry.date === today ? "today" : esc(entry.date)}</span>
          <span class="log-dur">${entry.duration_minutes} min</span>
        </div>`).join("");
    } catch (err) {
      container.innerHTML = `<p class="error">${esc(err.message)}</p>`;
    }
  }

  /** 'bicep_curl' -> 'Bicep curl'. Module 1's label vocabulary is snake_case. */
  function prettyExercise(name) {
    const text = String(name || "").replace(/_/g, " ");
    return text.charAt(0).toUpperCase() + text.slice(1);
  }

  // ── Wiring ───────────────────────────────────────────────────────────────
  $("edit-profile-btn").addEventListener("click", openProfileModal);
  $("refresh-btn").addEventListener("click", () => { fetchRecommendations(); fetchExerciseLog(); });
  $("top-n-select").addEventListener("change", fetchRecommendations);
  $("feedback-btn").addEventListener("click", () => {
    openModal("feedback-modal");
    $("modal-feedback-input").focus();
  });
  $("clear-history-btn").addEventListener("click", clearHistoryAndFetch);
  $("reset-prefs-btn").addEventListener("click", resetPreferences);

  $("profile-form").addEventListener("submit", saveProfile);
  $("feedback-form").addEventListener("submit", submitFeedback);
  $("modal-weight").addEventListener("change", weightsChanged);
  $("modal-goal-weight").addEventListener("change", weightsChanged);

  document.querySelectorAll(".presets .chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      $("modal-feedback-input").value = chip.dataset.preset;
      $("modal-feedback-input").focus();
    });
  });

  // Close on the X, on Cancel, on a backdrop click, and on Escape.
  document.querySelectorAll("[data-close]").forEach((el) => {
    el.addEventListener("click", () => closeModal(el.dataset.close));
  });
  document.querySelectorAll(".modal-backdrop").forEach((backdrop) => {
    backdrop.addEventListener("click", (e) => {
      if (e.target === backdrop) closeModal(backdrop.id);
    });
  });
  document.addEventListener("keydown", (e) => {
    if (e.key !== "Escape") return;
    document.querySelectorAll(".modal-backdrop:not([hidden])")
      .forEach((backdrop) => closeModal(backdrop.id));
  });

  // ── Boot ─────────────────────────────────────────────────────────────────
  checkHealth();
  loadProfile();
  fetchExerciseLog();
  fetchRecommendations();
})();
