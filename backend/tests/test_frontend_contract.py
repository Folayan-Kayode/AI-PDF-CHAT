"""
The frontend and backend must agree on the API key.

The bug this guards: the backend read ``API_KEY`` from the repo-root ``.env``
while the frontend read ``BACKEND_API_KEY`` from the process environment. A
normal local run therefore sent an empty header, and every upload or chat came
back 401. Each side's tests passed because they never exercised the pair.
"""

import importlib.util
from pathlib import Path

from app.core.config import settings

FRONTEND_CLIENT = Path(__file__).resolve().parents[2] / "frontend" / "backend_client.py"


def _load_frontend_client():
    """Import frontend/backend_client.py by path (it is not a package)."""
    spec = importlib.util.spec_from_file_location("contract_backend_client", FRONTEND_CLIENT)
    module = importlib.util.module_from_spec(spec)

    assert spec.loader is not None

    spec.loader.exec_module(module)

    return module


def test_the_frontend_can_resolve_the_backend_key_from_a_dotenv(tmp_path):
    dotenv = tmp_path / ".env"
    dotenv.write_text(f"API_KEY={settings.API_KEY}\n", encoding="utf-8")

    client = _load_frontend_client()

    value, source = client.resolve_api_key({}, dotenv)

    assert value == settings.API_KEY
    assert source == str(dotenv)


def test_the_frontend_header_is_accepted_by_the_real_api(no_key_client, monkeypatch):
    from app.services import chat_service

    monkeypatch.setattr(
        chat_service.ChatService,
        "chat",
        staticmethod(lambda question: {"answer": "ok", "sources": [], "retrieval": {}}),
    )

    client = _load_frontend_client()

    client.BACKEND_API_KEY = settings.API_KEY

    response = no_key_client.post(
        "/chat/",
        json={"question": "q"},
        headers=client.api_headers(),
    )

    assert response.status_code == 200


def test_the_header_is_still_required_without_it(no_key_client):
    assert no_key_client.post("/chat/", json={"question": "q"}).status_code == 401
