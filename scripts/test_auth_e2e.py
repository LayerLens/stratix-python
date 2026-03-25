#!/usr/bin/env python3
"""Interactive end-to-end test for the CLI auth flow.

Starts a tiny local HTTP server that serves the /auth/cli-config endpoint,
then runs `layerlens login` against it.

Usage:
    python scripts/test_auth_e2e.py

This tests the full flow WITHOUT needing the real backend running.
The device-code step will still fail (no real Cognito) but it validates
that config discovery, credential storage, and the CLI plumbing all work.

To test against the real staging/production backend, set:
    LAYERLENS_STRATIX_BASE_URL=https://api.layerlens.ai/api/v1 layerlens login
"""

import json
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

# Fake Cognito config — replace with real values to test end-to-end
FAKE_AUTH_CONFIG = {
    "region": "us-east-1",
    "client_id": "REPLACE_WITH_REAL_CLIENT_ID",
    "domain": "atlas-production",
    "scopes": "openid profile email",
}

# Fake login response
FAKE_LOGIN_RESPONSE = {
    "access_token": "fake-access-token-xyz",
    "id_token": "fake-id-token-xyz",
    "refresh_token": "fake-refresh-token-xyz",
    "expires_in": 3600,
    "token_type": "Bearer",
    "user": {"email": "test@example.com", "given_name": "Test", "name": "Test User"},
}

# Fake device-code response
FAKE_DEVICE_CODE_RESPONSE = {
    "device_code": "test-device-code-000",
    "user_code": "ABCD-1234",
    "verification_uri": "https://atlas-production.auth.us-east-1.amazoncognito.com/activate",
    "verification_uri_complete": "https://atlas-production.auth.us-east-1.amazoncognito.com/activate?user_code=ABCD-1234",
    "expires_in": 60,
    "interval": 5,
}


class MockHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.endswith("/dgklmnr/auth/cli-config"):
            self._json_response(200, FAKE_AUTH_CONFIG)
        else:
            self._json_response(404, {"error": "not found"})

    def do_POST(self):
        if self.path.endswith("/dgklmnr/auth/cli-login"):
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            if not body.get("email") or not body.get("password"):
                self._json_response(400, {"error": "Email and password are required"})
            elif body["email"] == "test@example.com" and body["password"] == "password123":
                self._json_response(200, FAKE_LOGIN_RESPONSE)
            else:
                self._json_response(401, {"error": "Invalid email or password"})
        elif "deviceAuthorization" in self.path:
            self._json_response(200, FAKE_DEVICE_CODE_RESPONSE)
        elif "oauth2/token" in self.path:
            # Simulate authorization_pending
            self._json_response(400, {"error": "authorization_pending"})
        else:
            self._json_response(404, {"error": "not found"})

    def _json_response(self, status, body):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(body).encode())

    def log_message(self, format, *args):
        print(f"  [mock-server] {format % args}")


def main():
    port = 18923
    server = HTTPServer(("127.0.0.1", port), MockHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    base_url = f"http://127.0.0.1:{port}/api/v1"
    print(f"\n  Mock server running at {base_url}")
    print(f"  Testing config discovery...\n")

    # Test 1: Config discovery
    import layerlens.cli._auth as auth_mod
    from layerlens.cli._auth import fetch_auth_config

    auth_mod._cached_auth_config = None  # clear cache

    config = fetch_auth_config(base_url)
    print(f"  Config fetched: {json.dumps(config, indent=2)}")
    assert config["client_id"] == FAKE_AUTH_CONFIG["client_id"], "Config mismatch!"
    print("  [PASS] Config discovery works\n")

    # Test 2: Credential storage round-trip
    from layerlens.cli._auth import load_credentials, save_credentials, clear_credentials

    test_creds = {"access_token": "test-tok", "auth_config": config}
    save_credentials(test_creds)
    loaded = load_credentials()
    assert loaded["access_token"] == "test-tok"
    print("  [PASS] Credential storage works\n")

    clear_credentials()
    assert load_credentials() is None
    print("  [PASS] Credential clearing works\n")

    # Test 3: CLI login via email/password
    print("  Testing CLI login command (email/password)...\n")

    import os

    os.environ["LAYERLENS_STRATIX_BASE_URL"] = base_url
    auth_mod._cached_auth_config = None  # clear cache

    from layerlens.cli._auth import cli_login

    creds = cli_login("test@example.com", "password123", base_url=base_url)
    assert creds["access_token"] == FAKE_LOGIN_RESPONSE["access_token"]
    assert creds["user"]["email"] == "test@example.com"
    print("  [PASS] CLI login works\n")

    # Test 4: CLI login command via CliRunner
    print("  Testing CLI login command via CliRunner...\n")
    clear_credentials()
    auth_mod._cached_auth_config = None

    from click.testing import CliRunner

    from layerlens.cli._app import cli

    runner = CliRunner()
    result = runner.invoke(cli, ["login"], input="test@example.com\npassword123\n")
    print(f"  Exit code: {result.exit_code}")
    print(f"  Output: {result.output}")
    assert result.exit_code == 0
    loaded = load_credentials()
    assert loaded is not None and loaded["access_token"] == FAKE_LOGIN_RESPONSE["access_token"]
    print("  [PASS] CLI login command works\n")

    # Test 5: whoami command
    print("  Testing whoami command...\n")
    result = runner.invoke(cli, ["whoami"])
    print(f"  Exit code: {result.exit_code}")
    print(f"  Output: {result.output}")
    assert result.exit_code == 0
    print("  [PASS] whoami works\n")

    # Test 6: logout command
    result = runner.invoke(cli, ["logout"])
    assert result.exit_code == 0
    assert load_credentials() is None
    print("  [PASS] logout works\n")

    server.shutdown()
    print("\n  Done! All checks passed.")


if __name__ == "__main__":
    main()
