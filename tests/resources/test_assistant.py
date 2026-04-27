"""Tests for the Stratix Assistant SDK resource handler."""

from unittest.mock import Mock, patch

import httpx
import pytest

from layerlens.models import (
    AssistantConversation,
    AssistantConversationList,
    AssistantMessageList,
    AssistantStreamEvent,
)
from layerlens.resources.assistant.assistant import Assistant, _parse_sse_block


class TestAssistantParsing:
    """Tests for the SSE block parser — runs without HTTP."""

    def test_parses_token_event(self):
        block = 'event: token\ndata: {"content": "hello"}'
        ev = _parse_sse_block(block)
        assert ev is not None
        assert ev.event == "token"
        assert ev.data == {"content": "hello"}
        assert ev.text() == "hello"
        assert not ev.is_terminal()

    def test_parses_done_event_terminal(self):
        block = 'event: done\ndata: {"content": "full text", "tool_calls": [], "turns": 1, "latency_ms": 42, "tokens_used": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}}'
        ev = _parse_sse_block(block)
        assert ev is not None
        assert ev.event == "done"
        assert ev.is_terminal()
        assert ev.text() == "full text"

    def test_parses_moderation_refused_terminal(self):
        block = 'event: moderation_refused\ndata: {"gate": "input", "categories": ["prompt_injection"], "severity": 0.9, "provider": "bedrock"}'
        ev = _parse_sse_block(block)
        assert ev is not None
        assert ev.event == "moderation_refused"
        assert ev.is_terminal()
        assert ev.data["categories"] == ["prompt_injection"]

    def test_parses_error_terminal(self):
        block = 'event: error\ndata: {"message": "boom", "code": "all_models_unavailable"}'
        ev = _parse_sse_block(block)
        assert ev is not None
        assert ev.event == "error"
        assert ev.is_terminal()
        assert ev.data["code"] == "all_models_unavailable"

    def test_unknown_event_skipped_for_forward_compat(self):
        # Server might add new event types in the future; clients
        # should silently skip them rather than crash.
        block = 'event: future_event_type\ndata: {"x": 1}'
        ev = _parse_sse_block(block)
        assert ev is None

    def test_malformed_json_returns_none(self):
        block = "event: token\ndata: {not json"
        assert _parse_sse_block(block) is None

    def test_missing_event_line_returns_none(self):
        block = 'data: {"content": "x"}'
        assert _parse_sse_block(block) is None

    def test_missing_data_line_returns_none(self):
        block = "event: token"
        assert _parse_sse_block(block) is None

    def test_text_returns_empty_for_non_text_events(self):
        ev = AssistantStreamEvent(event="tool_call", data={"name": "search", "args": {}})
        assert ev.text() == ""


class TestAssistantResource:
    """Tests for the sync Assistant resource methods."""

    @pytest.fixture
    def mock_client(self):
        client = Mock()
        client.organization_id = "org-123"
        client.project_id = "proj-456"
        client.base_url = "https://api.example.com/api/v1"
        client.default_headers = {"x-api-key": "test-key"}
        client.get_cast = Mock()
        client.post_cast = Mock()
        client.patch_cast = Mock()
        client.delete_cast = Mock()
        return client

    @pytest.fixture
    def resource(self, mock_client):
        return Assistant(mock_client)

    def test_base_url_includes_org_and_project(self, resource):
        assert resource._base_url() == (
            "/organizations/org-123/projects/proj-456/assistant"
        )

    def test_list_conversations_unwraps_envelope(self, mock_client, resource):
        mock_client.get_cast.return_value = {
            "status": 200,
            "data": {
                "conversations": [
                    {
                        "id": "c1",
                        "organization_id": "org-123",
                        "project_id": "proj-456",
                        "user_id": "u1",
                        "title": "Hello",
                        "created_at": "2026-04-01T00:00:00Z",
                        "updated_at": "2026-04-01T00:00:00Z",
                    }
                ],
                "total": 1,
            },
        }
        result = resource.list_conversations()
        assert isinstance(result, AssistantConversationList)
        assert result.total == 1
        assert result.conversations[0].title == "Hello"

    def test_list_conversations_returns_empty_on_unexpected_shape(self, mock_client, resource):
        mock_client.get_cast.return_value = "not a dict"
        result = resource.list_conversations()
        assert result.total == 0
        assert result.conversations == []

    def test_create_conversation_omits_title_when_none(self, mock_client, resource):
        mock_client.post_cast.return_value = {
            "id": "c1",
            "organization_id": "org-123",
            "project_id": "proj-456",
            "user_id": "u1",
            "title": "New conversation",
            "created_at": "2026-04-01T00:00:00Z",
            "updated_at": "2026-04-01T00:00:00Z",
        }
        result = resource.create_conversation()
        # body must NOT include title key when caller didn't pass one
        call_kwargs = mock_client.post_cast.call_args.kwargs
        assert call_kwargs["body"] == {}
        assert isinstance(result, AssistantConversation)

    def test_create_conversation_sends_title(self, mock_client, resource):
        mock_client.post_cast.return_value = {
            "id": "c1",
            "organization_id": "org-123",
            "project_id": "proj-456",
            "user_id": "u1",
            "title": "Custom",
            "created_at": "2026-04-01T00:00:00Z",
            "updated_at": "2026-04-01T00:00:00Z",
        }
        resource.create_conversation(title="Custom")
        assert mock_client.post_cast.call_args.kwargs["body"] == {"title": "Custom"}

    def test_rename_conversation_uses_patch(self, mock_client, resource):
        mock_client.patch_cast.return_value = {
            "id": "c1",
            "organization_id": "org-123",
            "project_id": "proj-456",
            "user_id": "u1",
            "title": "Renamed",
            "created_at": "2026-04-01T00:00:00Z",
            "updated_at": "2026-04-01T00:00:01Z",
        }
        result = resource.rename_conversation("c1", title="Renamed")
        assert result.title == "Renamed"
        assert mock_client.patch_cast.call_args.kwargs["body"] == {"title": "Renamed"}

    def test_delete_conversation_calls_delete(self, mock_client, resource):
        mock_client.delete_cast.return_value = None
        resource.delete_conversation("c1")
        assert mock_client.delete_cast.called
        # Path passed positionally
        called_path = mock_client.delete_cast.call_args.args[0]
        assert called_path == "/organizations/org-123/projects/proj-456/assistant/conversations/c1"

    def test_list_messages_with_limit(self, mock_client, resource):
        mock_client.get_cast.return_value = {"messages": [], "total": 0}
        resource.list_messages("c1", limit=50)
        assert mock_client.get_cast.call_args.kwargs.get("params") == {"limit": 50}

    def test_list_messages_returns_empty_on_unexpected_shape(self, mock_client, resource):
        mock_client.get_cast.return_value = ["not", "a", "dict"]
        result = resource.list_messages("c1")
        assert isinstance(result, AssistantMessageList)
        assert result.total == 0


class TestAssistantChatStreaming:
    """Tests for the SSE chat() iterator."""

    @pytest.fixture
    def mock_client(self):
        client = Mock()
        client.organization_id = "org-123"
        client.project_id = "proj-456"
        client.base_url = "https://api.example.com/api/v1"
        client.default_headers = {"x-api-key": "test-key"}
        return client

    def test_chat_yields_token_then_done_then_stops(self, mock_client):
        # Mock httpx.Client + stream context manager + iter_lines
        sse_lines = [
            "event: token",
            'data: {"content": "Hello"}',
            "",
            "event: token",
            'data: {"content": " world"}',
            "",
            "event: done",
            'data: {"content": "Hello world", "tool_calls": [], "turns": 1, "latency_ms": 10, "tokens_used": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}}',
            "",
            # If the iterator kept reading past `done`, this token would
            # be emitted — but it must not be (terminal event ends the
            # stream).
            "event: token",
            'data: {"content": "should not be yielded"}',
            "",
        ]

        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.iter_lines = Mock(return_value=iter(sse_lines))

        mock_stream_cm = Mock()
        mock_stream_cm.__enter__ = Mock(return_value=mock_response)
        mock_stream_cm.__exit__ = Mock(return_value=False)

        mock_http = Mock()
        mock_http.stream = Mock(return_value=mock_stream_cm)

        mock_http_cm = Mock()
        mock_http_cm.__enter__ = Mock(return_value=mock_http)
        mock_http_cm.__exit__ = Mock(return_value=False)

        with patch("httpx.Client", return_value=mock_http_cm):
            resource = Assistant(mock_client)
            events = list(resource.chat("c1", "hi"))

        # Two tokens + one done; the last (post-terminal) token must
        # NOT have been yielded.
        assert len(events) == 3
        assert events[0].event == "token"
        assert events[0].text() == "Hello"
        assert events[1].event == "token"
        assert events[1].text() == " world"
        assert events[2].event == "done"
        assert events[2].is_terminal()

    def test_chat_raises_for_status_on_4xx(self, mock_client):
        # 403 SDK-disabled → httpx.HTTPStatusError propagates.
        request = httpx.Request("POST", "https://api.example.com/v1/chat")
        response = httpx.Response(403, request=request)
        mock_response = Mock()
        mock_response.raise_for_status = Mock(side_effect=httpx.HTTPStatusError("403", request=request, response=response))

        mock_stream_cm = Mock()
        mock_stream_cm.__enter__ = Mock(return_value=mock_response)
        mock_stream_cm.__exit__ = Mock(return_value=False)

        mock_http = Mock()
        mock_http.stream = Mock(return_value=mock_stream_cm)

        mock_http_cm = Mock()
        mock_http_cm.__enter__ = Mock(return_value=mock_http)
        mock_http_cm.__exit__ = Mock(return_value=False)

        with patch("httpx.Client", return_value=mock_http_cm):
            resource = Assistant(mock_client)
            with pytest.raises(httpx.HTTPStatusError):
                list(resource.chat("c1", "hi"))
