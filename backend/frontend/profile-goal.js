/* =============================================================================
   Shared "Profile & goal" popup — one implementation, two callers.

     * the meal-plan page's optional "Edit profile & goal" modal (meals.js)
     * the login page's mandatory onboarding modal (login.js)

   The user asked for the second to be *exactly* the first, so it is literally
   the same script over the same markup and the same stylesheet (forms.css).
   Both pages carry the same element ids inside the dialog, listed in IDS
   below; only the surrounding chrome and what happens after saving differ.

   Where the data goes
   -------------------
   Both writes land in Module 3's own user_profile.json, through its own
   services — nothing here touches that file directly:

       PUT  /api/meals/profile      weight_kg, height_cm, activity_level,
                                    selected_pace     -> profile_service.update_profile
       POST /api/meals/goal/options goal_weight_kg    -> also syncs the profile's
                                    `goal` direction (loss/gain/maintenance)

   Why the pace is saved rather than applied live
   ----------------------------------------------
   Module 3's `POST /goal/select-pace` recalculates the daily target, and to do
   that it calls `recommender_service.calculate_targets()` — which imports the
   recommender, which imports the FAISS index and the SentenceTransformer. That
   is a ~60-second first load (see backend/module3/bridge.py). Firing it from a
   radio button would hang the login page on a click that should be instant.

   `selected_pace` is a plain field on Module 3's own ProfileUpdateRequest, so
   it is persisted with the rest of the form via the light `PUT /profile`
   instead, and the recalculated target simply shows up the next time the meal
   plan is fetched. Same stored result, no ML import on the critical path.
   ============================================================================= */

window.ProfileGoal = (() => {
  "use strict";

  const API = "/api/meals";
  const $ = (id) => document.getElementById(id);

  // Every id this module reads or writes. Both pages must provide all of them.
  const IDS = ["profile-modal", "profile-form", "modal-weight", "modal-height",
               "modal-activity", "modal-goal-weight", "modal-goal-section",
               "goal-summary", "goal-pace-selector", "goal-safe-note"];

  let notify = () => {};
  let onSaved = () => {};
  let requireGoalWeight = false;
  let selectedPace = null;

  // ── HTTP ─────────────────────────────────────────────────────────────────
  async function request(path, options) {
    // X-Demo-User decides whose profile is read and written; without it the
    // server falls back to Module 3's shared files. See backend/user_store.py.
    const opts = Object.assign({}, options);
    opts.headers = window.Auth ? Auth.headers(opts.headers) : opts.headers;
    const response = await fetch(API + path, opts);
    if (!response.ok) {
      let detail = `${response.status} ${response.statusText}`;
      try {
        const body = await response.json();
        if (body.detail) detail = body.detail;
      } catch (e) { /* non-JSON error body; keep the status line */ }
      throw new Error(detail);
    }
    return response.json();
  }

  const body = (method, payload) => ({
    method,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  function esc(text) {
    return String(text ?? "").replace(/[&<>"]/g, (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  }

  // ── Fill ─────────────────────────────────────────────────────────────────
  /** Prime the form from a profile object (from GET /api/meals/profile). */
  function fill(profile) {
    const p = profile || {};
    $("modal-weight").value = p.weight_kg ?? "";
    $("modal-height").value = p.height_cm ?? "";
    $("modal-activity").value = p.activity_level || "sedentary";
    $("modal-goal-weight").value = p.goal_weight_kg ?? "";
    selectedPace = p.selected_pace || null;
    weightsChanged();
  }

  // ── Pace picker ──────────────────────────────────────────────────────────
  /** Shown only once both weights are usable — there is nothing to pace until then. */
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
      // Push the typed weight first so kg_to_change is computed against what is
      // on screen, not against the last saved value.
      await request("/profile", body("PUT", { weight_kg: currentWeight }));
      renderGoalPanel(await request("/goal/options",
                                    body("POST", { goal_weight_kg: goalWeight })));
    } catch (err) {
      notify("Could not work out your goal: " + err.message, true);
    }
  }

  function renderGoalPanel(data) {
    $("modal-goal-section").hidden = false;

    if (data.already_at_goal) {
      $("goal-summary").innerHTML = "<strong>You're already at your goal weight.</strong>";
      $("goal-pace-selector").innerHTML = "";
      $("goal-safe-note").hidden = true;
      selectedPace = null;
      return;
    }

    $("goal-safe-note").hidden = false;
    $("goal-summary").innerHTML =
      `You want to <strong>${data.is_loss ? "lose" : "gain"} ${esc(data.kg_to_change)} kg</strong>.`;

    // Keep the user's existing pace checked if they have one; otherwise take
    // the server's default (is_default), so a pace is always selected and the
    // form can never be submitted with none.
    const known = data.pace_options.some((o) => o.key === selectedPace);
    if (!known) selectedPace = (data.pace_options.find((o) => o.is_default) || {}).key || null;

    $("goal-pace-selector").innerHTML = data.pace_options.map((opt) => `
      <label class="pace-option ${opt.key === selectedPace ? "selected" : ""}"
             data-pace-key="${esc(opt.key)}">
        <input type="radio" name="pace-option" value="${esc(opt.key)}"
               ${opt.key === selectedPace ? "checked" : ""}>
        <span>
          <span class="pace-label">${esc(opt.label)}</span>
          <span class="pace-meta">${opt.kcal_per_day} kcal/day &middot;
            ~${opt.estimated_weeks} weeks (${opt.estimated_months} months)</span>
        </span>
      </label>`).join("");

    $("goal-pace-selector").querySelectorAll("input[name='pace-option']")
      .forEach((input) => input.addEventListener("change", () => {
        selectedPace = input.value;
        document.querySelectorAll(".pace-option").forEach((el) => {
          el.classList.toggle("selected", el.dataset.paceKey === selectedPace);
        });
      }));
  }

  // ── Save ─────────────────────────────────────────────────────────────────
  async function submit(event) {
    event.preventDefault();
    const weight_kg = parseFloat($("modal-weight").value);
    const height_cm = parseFloat($("modal-height").value);
    const activity_level = $("modal-activity").value;
    const goalRaw = $("modal-goal-weight").value;
    const goal_weight_kg = goalRaw ? parseFloat(goalRaw) : null;

    if (!weight_kg || weight_kg <= 0 || !height_cm || height_cm <= 0) {
      notify("Enter a valid weight and height.", true);
      return false;
    }
    if (requireGoalWeight && (!goal_weight_kg || goal_weight_kg <= 0)) {
      notify("Enter the goal weight you're training towards.", true);
      $("modal-goal-weight").focus();
      return false;
    }

    const button = event.target.querySelector("button[type='submit']");
    const label = button ? button.textContent : null;
    if (button) { button.disabled = true; button.textContent = "Saving…"; }

    try {
      const update = { weight_kg, height_cm, activity_level };
      if (selectedPace) update.selected_pace = selectedPace;
      await request("/profile", body("PUT", update));

      // Persists goal_weight_kg AND syncs the profile's goal direction, which
      // is why it runs even though the pace panel already called it.
      if (goal_weight_kg && goal_weight_kg > 0) {
        await request("/goal/options", body("POST", { goal_weight_kg }));
      }
      await onSaved();
      return true;
    } catch (err) {
      notify("Could not save your profile: " + err.message, true);
      return false;
    } finally {
      if (button) { button.disabled = false; button.textContent = label; }
    }
  }

  // ── Mount ────────────────────────────────────────────────────────────────
  /**
   * options.notify(message, isError)  — how this page surfaces messages
   * options.onSaved()                 — after user_profile.json is updated
   * options.requireGoalWeight         — true for the mandatory login popup
   */
  function mount(options) {
    const missing = IDS.filter((id) => !$(id));
    if (missing.length) {
      // Loud on purpose: a silently half-wired popup would look like it saved.
      throw new Error("ProfileGoal: missing element ids: " + missing.join(", "));
    }
    notify = options.notify || (() => {});
    onSaved = options.onSaved || (() => {});
    requireGoalWeight = !!options.requireGoalWeight;

    $("profile-form").addEventListener("submit", submit);
    $("modal-weight").addEventListener("change", weightsChanged);
    $("modal-goal-weight").addEventListener("change", weightsChanged);
  }

  /** GET the stored profile and prime the form with it. */
  async function load() {
    try {
      const profile = await request("/profile", undefined);
      fill(profile);
      return profile;
    } catch (err) {
      fill({});
      notify("Could not load your saved profile: " + err.message, true);
      return null;
    }
  }

  return { mount, fill, load };
})();
