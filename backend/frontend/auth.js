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

  /**
   * Claim a username and make sure the server has a data folder for it.
   *
   * Resolves to `{ new_user }` — true the first time a name is used, which is
   * when the workout log and preference vector start empty (see
   * backend/user_store.py).
   */
  async function signIn(username) {
    const name = (username || "demo").trim() || "demo";
    let record = { username: name, new_user: false };
    try {
      const response = await fetch("/api/users/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username: name }),
      });
      if (response.ok) record = await response.json();
    } catch (e) {
      // Offline or the route is missing: still let them in — this is a demo
      // gate, and the server falls back to its shared data files anyway.
    }
    sessionStorage.setItem(KEY, JSON.stringify({
      username: record.username || name,
      newUser: !!record.new_user,
      at: new Date().toISOString(),
    }));
    return record;
  }

  /** The username to send with API calls, or "" when signed out. */
  function username() {
    const current = user();
    return (current && current.username) || "";
  }

  /**
   * Every Module 3 request must say who it is for, or it reads the shared
   * files instead of this user's. A WebSocket handshake cannot carry custom
   * headers, so app.js puts the same value in the query string instead.
   */
  function headers(extra) {
    const name = username();
    return Object.assign({}, extra, name ? { "X-Demo-User": name } : {});
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

  return { guard, signIn, signOut, signedIn, user, username, headers, wireControls };
})();

// The three app pages load this in <head> to gate themselves. The login page
// loads it too but must NOT be gated, so it opts out with data-no-guard.
if (!document.currentScript || !document.currentScript.hasAttribute("data-no-guard")) {
  if (window.Auth.guard()) {
    document.addEventListener("DOMContentLoaded", window.Auth.wireControls);
  }
}
