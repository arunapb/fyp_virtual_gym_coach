/* =============================================================================
   Login page.

   Two steps, one page:
     1. credentials — accepted whatever they are (demo build, no backend auth;
        see auth.js for exactly how little this guarantees). Both fields must
        be non-empty so the button is a real action rather than a no-op.
     2. the mandatory "Set your goal" popup — the meal-plan page's profile
        dialog, same markup and same driver (profile-goal.js). It cannot be
        dismissed: no close button, no Cancel, Escape and backdrop clicks
        suppressed below. The redirect into the app happens only after
        user_profile.json has actually been written.

   Doing the goal setup here rather than on the landing page keeps it out of
   the app pages entirely — no "have they onboarded yet?" check on every page,
   and no flash of the dashboard behind a modal the user must deal with first.
   ============================================================================= */

(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  let toastTimer = null;

  function toast(message, bad = false) {
    const el = $("toast");
    el.textContent = message;
    el.classList.toggle("bad", bad);
    el.classList.add("show");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => el.classList.remove("show"), 3500);
  }

  // ── Step 2: the goal popup ───────────────────────────────────────────────
  ProfileGoal.mount({
    notify: toast,
    requireGoalWeight: true,
    onSaved: () => {
      // Straight in, no confirmation step — the app itself is the confirmation.
      // replace() so Back cannot return to a login screen that is now moot.
      location.replace("/");
    },
  });

  async function openGoalSetup() {
    $("profile-modal").hidden = false;
    document.body.style.overflow = "hidden";
    // Prefill from whatever is already stored, so a returning demo user is
    // confirming their numbers rather than retyping them.
    await ProfileGoal.load();
    $("modal-weight").focus();
  }

  // Required means required: the two normal escape hatches are removed. (The
  // dialog has no close button or Cancel in the markup, so there is nothing
  // else to disable.)
  $("profile-modal").addEventListener("click", (e) => e.stopPropagation());
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !$("profile-modal").hidden) e.preventDefault();
  });

  // ── Step 1: credentials ──────────────────────────────────────────────────
  $("login-form").addEventListener("submit", (event) => {
    event.preventDefault();
    const username = $("username").value.trim();
    const password = $("password").value;
    const error = $("login-error");

    if (!username || !password) {
      error.textContent = "Enter a username and password — any will do.";
      error.hidden = false;
      (username ? $("password") : $("username")).focus();
      return;
    }

    error.hidden = true;
    Auth.signIn(username);
    openGoalSetup();
  });

  // Already signed in and back on this page (Back button, bookmark): don't make
  // them retype anything, just resume at the step they were on.
  if (Auth.signedIn()) {
    const current = Auth.user();
    if (current && current.username) $("username").value = current.username;
  }
})();
