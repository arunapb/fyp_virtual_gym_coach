/* =============================================================================
   Demo sign-in gate.

   THIS IS NOT AUTHENTICATION. There is no account, no password check, no
   server-side session and no protection of any kind: every API route stays
   open, and anyone can walk straight past this by setting one sessionStorage
   key from the console. It exists so the demo opens on a sign-in screen and so
   the goal-setup popup has a natural moment to appear — nothing more. Building
   it in the backend was explicitly out of scope.

   Loaded from <head> on the three app pages, deliberately render-blocking:
   redirecting after the page has painted would flash the dashboard at a
   signed-out visitor before whisking it away.

   sessionStorage, not localStorage, so the flag dies with the browser tab and
   the flow can be demonstrated again by reopening it — rather than being
   sticky forever with no obvious way back to the login screen.
   ============================================================================= */

window.Auth = (() => {
  "use strict";

  const KEY = "vgc-demo-session";
  const LOGIN_URL = "/login";

  const signedIn = () => sessionStorage.getItem(KEY) !== null;

  function signIn(username) {
    sessionStorage.setItem(KEY, JSON.stringify({
      username: username || "demo",
      at: new Date().toISOString(),
    }));
  }

  function user() {
    try {
      return JSON.parse(sessionStorage.getItem(KEY) || "null");
    } catch (e) {
      return null;
    }
  }

  function signOut() {
    sessionStorage.removeItem(KEY);
    location.replace(LOGIN_URL);
  }

  /** Bounce to the login page unless signed in. Call before the body renders. */
  function guard() {
    if (!signedIn()) {
      // replace(), not assign(), so Back does not bounce between the two.
      location.replace(LOGIN_URL);
      return false;
    }
    return true;
  }

  /** Wire the topbar's sign-out button and greeting, once the DOM exists. */
  function wireControls() {
    const button = document.getElementById("sign-out");
    if (button) button.addEventListener("click", signOut);

    const label = document.getElementById("who");
    const current = user();
    if (label && current && current.username) label.textContent = current.username;
  }

  return { guard, signIn, signOut, signedIn, user, wireControls };
})();

// The three app pages load this in <head> to gate themselves. The login page
// loads it too but must NOT be gated, so it opts out with data-no-guard.
if (!document.currentScript || !document.currentScript.hasAttribute("data-no-guard")) {
  if (window.Auth.guard()) {
    document.addEventListener("DOMContentLoaded", window.Auth.wireControls);
  }
}
