/* Role-aware login and navigation controls. Access checks also run in the API. */
(() => {
  const $ = (selector) => document.querySelector(selector);
  const overlay = $("#loginOverlay");
  const form = $("#loginForm");
  const errorBox = $("#loginError");
  let activeRole = null;

  function showLogin(message = "") {
    if (!overlay) return;
    overlay.classList.remove("hidden");
    errorBox.textContent = message;
    document.body.classList.add("login-active");
    if (message) $("#loginPassword")?.focus();
  }

  function setSession(user) {
    activeRole = user.role;
    document.body.dataset.role = user.role;
    document.body.classList.remove("login-active");
    overlay?.classList.add("hidden");
    const badge = $("#accountBadge");
    badge?.classList.remove("hidden");
    $("#accountName").textContent = user.username;
    $("#accountRole").textContent = user.role.toUpperCase();
    document.querySelectorAll("[data-admin-only]").forEach((item) => {
      item.classList.toggle("hidden", user.role !== "admin");
    });
  }

  const originalFetch = window.fetch.bind(window);
  window.fetch = async (...args) => {
    const response = await originalFetch(...args);
    const url = typeof args[0] === "string" ? args[0] : args[0]?.url || "";
    if (response.status === 401 && url.startsWith("/api/") && !url.startsWith("/api/auth/")) {
      activeRole = null;
      showLogin("Your session ended. Sign in again to continue.");
    }
    return response;
  };

  form?.addEventListener("submit", async (event) => {
    event.preventDefault();
    const button = $("#loginSubmit");
    button.disabled = true;
    errorBox.textContent = "";
    try {
      const response = await originalFetch("/api/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          role: $("#loginRole").value,
          username: $("#loginUsername").value,
          password: $("#loginPassword").value,
        }),
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.detail || "Sign in failed.");
      setSession(result.user);
      window.location.reload();
    } catch (error) {
      showLogin(error.message || "Unable to sign in.");
    } finally {
      button.disabled = false;
    }
  });

  $("#logoutButton")?.addEventListener("click", async () => {
    await originalFetch("/api/auth/logout", { method: "POST" });
    window.location.reload();
  });

  showLogin();
  originalFetch("/api/auth/me", { cache: "no-store" })
    .then((response) => response.json())
    .then((result) => result.authenticated && result.user ? setSession(result.user) : showLogin())
    .catch(() => showLogin("Cannot reach the authentication service. Check that the API is running."));
})();
