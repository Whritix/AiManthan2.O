"""Role-based cookie authentication for the Indore QA Cell demo."""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from http.cookies import SimpleCookie


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _unb64(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


class DemoAuthMiddleware:
    """Protect APIs and live streams while leaving the login shell loadable."""

    COOKIE = "indore_session"
    TTL_SECONDS = 8 * 60 * 60
    ADMIN_ONLY = {
        ("GET", "/api/audit"),
        ("GET", "/api/models"),
        ("GET", "/api/config"),
        ("PATCH", "/api/thresholds"),
        ("POST", "/api/control"),
        ("POST", "/api/outbox/flush"),
        ("POST", "/api/mes/events"),
    }
    ADMIN_PREFIXES = ("/api/datasets/", "/api/eval/", "/api/m2-body/", "/api/outbox/", "/api/mes/")

    def __init__(self, app):
        self.app = app
        render = os.environ.get("INDORE_RENDER") == "1"
        if render:
            required = (
                "INDORE_ADMIN_USERNAME", "INDORE_ADMIN_PASSWORD",
                "INDORE_EMPLOYEE_USERNAME", "INDORE_EMPLOYEE_PASSWORD",
                "INDORE_AUTH_SECRET",
            )
            missing = [name for name in required if not os.environ.get(name)]
            if missing:
                raise RuntimeError("Missing required Render auth variables: " + ", ".join(missing))
        self.users = {
            "admin": (
                os.environ.get("INDORE_ADMIN_USERNAME", "admin").strip().casefold(),
                os.environ.get("INDORE_ADMIN_PASSWORD", "admin123"),
            ),
            "employee": (
                os.environ.get("INDORE_EMPLOYEE_USERNAME", "employee").strip().casefold(),
                os.environ.get("INDORE_EMPLOYEE_PASSWORD", "employee123"),
            ),
        }
        self.secret = os.environ.get("INDORE_AUTH_SECRET", "")
        if not self.secret:
            # Local-only key: sessions expire on server restart. Set an env key for persistence.
            self.secret = secrets.token_urlsafe(48)
            print("[auth] Local demo accounts enabled (admin/admin123, employee/employee123). Set INDORE_* credentials before exposing the server.", flush=True)
        self.secure_cookie = render

    def _sign(self, payload: bytes) -> str:
        return _b64(hmac.new(self.secret.encode("utf-8"), payload, hashlib.sha256).digest())

    def _make_token(self, role: str, username: str) -> str:
        payload = json.dumps(
            {"role": role, "username": username, "exp": int(time.time()) + self.TTL_SECONDS},
            separators=(",", ":"),
        ).encode("utf-8")
        encoded = _b64(payload)
        return encoded + "." + self._sign(encoded.encode("ascii"))

    def _read_user(self, headers: dict[bytes, bytes]) -> dict | None:
        cookies = SimpleCookie()
        try:
            cookies.load(headers.get(b"cookie", b"").decode("latin-1"))
            morsel = cookies.get(self.COOKIE)
            if not morsel:
                return None
            encoded, signature = morsel.value.rsplit(".", 1)
            if not hmac.compare_digest(signature, self._sign(encoded.encode("ascii"))):
                return None
            user = json.loads(_unb64(encoded))
            if user.get("role") not in self.users or int(user.get("exp", 0)) < int(time.time()):
                return None
            if self.users[user["role"]][0] != user.get("username"):
                return None
            return {"username": user["username"], "role": user["role"]}
        except (ValueError, TypeError, KeyError, json.JSONDecodeError, UnicodeError):
            return None

    @staticmethod
    async def _respond(send, status: int, payload: dict, headers: list[tuple[bytes, bytes]] | None = None):
        body = json.dumps(payload).encode("utf-8")
        response_headers = [(b"content-type", b"application/json; charset=utf-8"),
                            (b"content-length", str(len(body)).encode("ascii"))]
        response_headers.extend(headers or [])
        await send({"type": "http.response.start", "status": status, "headers": response_headers})
        await send({"type": "http.response.body", "body": body})

    async def __call__(self, scope, receive, send):
        if scope["type"] not in {"http", "websocket"}:
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        method = scope.get("method", "GET").upper()
        headers = dict(scope.get("headers") or [])
        if scope["type"] == "http" and path == "/healthz":
            await send({"type": "http.response.start", "status": 200,
                        "headers": [(b"content-type", b"text/plain"), (b"content-length", b"2")]})
            await send({"type": "http.response.body", "body": b"ok"})
            return

        if scope["type"] == "http" and path == "/api/auth/login" and method == "POST":
            chunks = []
            while True:
                message = await receive()
                if message["type"] == "http.request":
                    chunks.append(message.get("body", b""))
                    if not message.get("more_body", False):
                        break
                elif message["type"] == "http.disconnect":
                    return
            try:
                body = json.loads(b"".join(chunks) or b"{}")
                role = str(body.get("role", "")).strip().casefold()
                username = str(body.get("username", "")).strip().casefold()
                password = str(body.get("password", ""))
            except (ValueError, AttributeError):
                await self._respond(send, 400, {"detail": "Enter a role, username, and password."})
                return
            expected_user, expected_password = self.users.get(role, ("", ""))
            valid = bool(expected_user and expected_password and
                         hmac.compare_digest(username.encode("utf-8"), expected_user.encode("utf-8")) and
                         hmac.compare_digest(password.encode("utf-8"), expected_password.encode("utf-8")))
            if not valid:
                await self._respond(send, 401, {"detail": "Incorrect role, username, or password."})
                return
            token = self._make_token(role, username)
            secure = "; Secure" if self.secure_cookie or scope.get("scheme") == "https" else ""
            cookie = f"{self.COOKIE}={token}; Path=/; Max-Age={self.TTL_SECONDS}; HttpOnly; SameSite=Strict{secure}".encode("ascii")
            await self._respond(send, 200, {"authenticated": True, "user": {"username": username, "role": role}}, [(b"set-cookie", cookie)])
            return

        if scope["type"] == "http" and path == "/api/auth/logout" and method == "POST":
            cookie = f"{self.COOKIE}=; Path=/; Max-Age=0; HttpOnly; SameSite=Strict".encode("ascii")
            await self._respond(send, 200, {"ok": True}, [(b"set-cookie", cookie)])
            return

        user = self._read_user(headers)
        if scope["type"] == "http" and path == "/api/auth/me" and method == "GET":
            await self._respond(send, 200, {"authenticated": bool(user), "user": user})
            return

        # The login shell and static assets must be available before a session exists.
        protected = path.startswith(("/api/", "/media/", "/eval/", "/live.mjpg", "/ws/"))
        if not protected:
            await self.app(scope, receive, send)
            return
        if not user:
            if scope["type"] == "websocket":
                await send({"type": "websocket.close", "code": 1008})
            else:
                await self._respond(send, 401, {"detail": "Login required."})
            return

        admin_required = ((method, path) in self.ADMIN_ONLY
                          or any(path.startswith(prefix) for prefix in self.ADMIN_PREFIXES))
        if user["role"] == "employee" and admin_required:
            await self._respond(send, 403, {"detail": "Admin access is required for this action."})
            return

        scope.setdefault("state", {})["auth_user"] = user
        await self.app(scope, receive, send)
