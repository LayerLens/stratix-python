import logging
from typing import Mapping
from unittest.mock import Mock

import pytest

from layerlens._utils import (
    SENSITIVE_HEADERS,
    SensitiveHeadersFilter,
    is_dict,
    is_mapping,
)


class TestTypeguards:
    """Test utility type guard functions."""

    def test_is_dict_with_dict(self):
        """is_dict returns True for dict objects."""
        test_dict = {"key": "value", "number": 42}

        assert is_dict(test_dict) is True

    def test_is_dict_with_empty_dict(self):
        """is_dict returns True for empty dict."""
        empty_dict = {}

        assert is_dict(empty_dict) is True

    def test_is_dict_with_nested_dict(self):
        """is_dict returns True for nested dict structures."""
        nested_dict = {"outer": {"inner": {"deep": "value"}}}

        assert is_dict(nested_dict) is True

    @pytest.mark.parametrize(
        "non_dict_value",
        [
            "string",
            123,
            [1, 2, 3],
            (1, 2, 3),
            {"key", "value"},  # set
            None,
            True,
            object(),
        ],
    )
    def test_is_dict_with_non_dict_objects(self, non_dict_value):
        """is_dict returns False for non-dict objects."""
        assert is_dict(non_dict_value) is False

    def test_is_mapping_with_dict(self):
        """is_mapping returns True for dict objects."""
        test_dict = {"key": "value"}

        assert is_mapping(test_dict) is True

    def test_is_mapping_with_custom_mapping(self):
        """is_mapping returns True for custom Mapping implementations."""
        from collections import UserDict, OrderedDict

        ordered_dict = OrderedDict([("a", 1), ("b", 2)])
        user_dict = UserDict({"x": 10, "y": 20})

        assert is_mapping(ordered_dict) is True
        assert is_mapping(user_dict) is True

    def test_is_mapping_with_mapping_subclass(self):
        """is_mapping returns True for Mapping subclasses."""

        class CustomMapping(Mapping):
            def __init__(self):
                self._data = {"custom": "mapping"}

            def __getitem__(self, key):
                return self._data[key]

            def __iter__(self):
                return iter(self._data)

            def __len__(self):
                return len(self._data)

        custom_mapping = CustomMapping()

        assert is_mapping(custom_mapping) is True
        assert custom_mapping["custom"] == "mapping"

    @pytest.mark.parametrize(
        "non_mapping_value",
        [
            "string",
            123,
            [1, 2, 3],
            (1, 2, 3),
            {"key", "value"},  # set
            None,
            True,
            object(),
        ],
    )
    def test_is_mapping_with_non_mapping_objects(self, non_mapping_value):
        """is_mapping returns False for non-mapping objects."""
        assert is_mapping(non_mapping_value) is False


class TestSensitiveHeaders:
    """Test sensitive headers constant and handling."""

    def test_sensitive_headers_constant(self):
        """SENSITIVE_HEADERS contains expected header names."""
        assert "x-api-key" in SENSITIVE_HEADERS
        assert "authorization" in SENSITIVE_HEADERS
        assert isinstance(SENSITIVE_HEADERS, set)

    def test_sensitive_headers_lowercase(self):
        """SENSITIVE_HEADERS contains lowercase header names."""
        for header in SENSITIVE_HEADERS:
            assert header == header.lower()


class TestSensitiveHeadersFilter:
    """Test SensitiveHeadersFilter logging functionality."""

    @pytest.fixture
    def filter_instance(self):
        """Create SensitiveHeadersFilter instance."""
        return SensitiveHeadersFilter()

    @pytest.fixture
    def mock_log_record(self):
        """Create mock logging record."""
        record = Mock(spec=logging.LogRecord)
        record.args = {}
        return record

    def test_filter_initialization(self):
        """SensitiveHeadersFilter initializes correctly."""
        filter_instance = SensitiveHeadersFilter()

        assert isinstance(filter_instance, logging.Filter)
        assert hasattr(filter_instance, "filter")

    def test_filter_returns_true_by_default(self, filter_instance, mock_log_record):
        """filter method always returns True to allow logging."""
        result = filter_instance.filter(mock_log_record)

        assert result is True

    def test_filter_handles_record_without_headers(self, filter_instance, mock_log_record):
        """filter handles log records without headers gracefully."""
        mock_log_record.args = {"message": "test", "data": "value"}

        result = filter_instance.filter(mock_log_record)

        assert result is True
        assert mock_log_record.args["message"] == "test"

    def test_filter_handles_non_dict_args(self, filter_instance, mock_log_record):
        """filter handles log records with non-dict args."""
        mock_log_record.args = "string args"

        result = filter_instance.filter(mock_log_record)

        assert result is True

    def test_filter_redacts_sensitive_headers(self, filter_instance, mock_log_record):
        """filter redacts sensitive header values."""
        mock_log_record.args = {
            "headers": {
                "content-type": "application/json",
                "x-api-key": "secret-key-123",
                "authorization": "Bearer token-456",
                "user-agent": "stratix-python-sdk",
            }
        }

        result = filter_instance.filter(mock_log_record)

        assert result is True
        headers = mock_log_record.args["headers"]
        assert headers["content-type"] == "application/json"
        assert headers["x-api-key"] == "<redacted>"
        assert headers["authorization"] == "<redacted>"
        assert headers["user-agent"] == "stratix-python-sdk"

    def test_filter_handles_case_insensitive_headers(self, filter_instance, mock_log_record):
        """filter redacts headers regardless of case."""
        mock_log_record.args = {
            "headers": {
                "X-API-KEY": "secret-key-123",
                "Authorization": "Bearer token-456",
                "AUTHORIZATION": "Bearer another-token",
            }
        }

        result = filter_instance.filter(mock_log_record)

        assert result is True
        headers = mock_log_record.args["headers"]
        assert headers["X-API-KEY"] == "<redacted>"
        assert headers["Authorization"] == "<redacted>"
        assert headers["AUTHORIZATION"] == "<redacted>"

    def test_filter_preserves_original_args_structure(self, filter_instance, mock_log_record):
        """filter preserves the original args structure."""
        original_args = {
            "method": "POST",
            "url": "/test",
            "headers": {
                "x-api-key": "secret",
                "content-type": "application/json",
            },
            "body": {"data": "test"},
        }
        mock_log_record.args = original_args

        result = filter_instance.filter(mock_log_record)

        assert result is True
        assert mock_log_record.args["method"] == "POST"
        assert mock_log_record.args["url"] == "/test"
        assert mock_log_record.args["body"] == {"data": "test"}
        assert mock_log_record.args["headers"]["content-type"] == "application/json"

    def test_filter_creates_copy_of_headers(self, filter_instance, mock_log_record):
        """filter creates a copy of headers dict to avoid modifying original."""
        original_headers = {
            "x-api-key": "secret-key",
            "content-type": "application/json",
        }
        mock_log_record.args = {"headers": original_headers}

        filter_instance.filter(mock_log_record)

        # Original headers should be unchanged
        assert original_headers["x-api-key"] == "secret-key"
        # Record headers should be modified
        assert mock_log_record.args["headers"]["x-api-key"] == "<redacted>"
        # They should be different objects
        assert mock_log_record.args["headers"] is not original_headers

    def test_filter_handles_non_string_header_keys(self, filter_instance, mock_log_record):
        """filter handles non-string header keys gracefully."""
        mock_log_record.args = {
            "headers": {
                123: "numeric-key",
                "x-api-key": "secret-key",
                ("tuple", "key"): "tuple-value",
            }
        }

        result = filter_instance.filter(mock_log_record)

        assert result is True
        headers = mock_log_record.args["headers"]
        assert headers[123] == "numeric-key"  # Non-string keys unchanged
        assert headers["x-api-key"] == "<redacted>"  # String keys processed
        assert headers[("tuple", "key")] == "tuple-value"

    def test_filter_handles_non_dict_headers(self, filter_instance, mock_log_record):
        """filter handles cases where headers is not a dict."""
        mock_log_record.args = {"headers": "not-a-dict", "other": "data"}

        result = filter_instance.filter(mock_log_record)

        assert result is True
        assert mock_log_record.args["headers"] == "not-a-dict"
        assert mock_log_record.args["other"] == "data"

    def test_filter_with_empty_headers(self, filter_instance, mock_log_record):
        """filter handles empty headers dict."""
        mock_log_record.args = {"headers": {}}

        result = filter_instance.filter(mock_log_record)

        assert result is True
        assert mock_log_record.args["headers"] == {}

    def test_filter_with_complex_header_values(self, filter_instance, mock_log_record):
        """filter redacts complex header values."""
        mock_log_record.args = {
            "headers": {
                "authorization": {"type": "Bearer", "token": "complex-token-123"},
                "x-api-key": ["key1", "key2", "key3"],
                "content-type": "application/json",
            }
        }

        result = filter_instance.filter(mock_log_record)

        assert result is True
        headers = mock_log_record.args["headers"]
        assert headers["authorization"] == "<redacted>"
        assert headers["x-api-key"] == "<redacted>"
        assert headers["content-type"] == "application/json"

    @pytest.mark.parametrize("sensitive_header", list(SENSITIVE_HEADERS))
    def test_filter_redacts_all_sensitive_headers(self, filter_instance, mock_log_record, sensitive_header):
        """filter redacts all headers defined in SENSITIVE_HEADERS."""
        mock_log_record.args = {
            "headers": {
                sensitive_header: f"secret-value-for-{sensitive_header}",
                "safe-header": "safe-value",
            }
        }

        result = filter_instance.filter(mock_log_record)

        assert result is True
        headers = mock_log_record.args["headers"]
        assert headers[sensitive_header] == "<redacted>"
        assert headers["safe-header"] == "safe-value"


class TestUtilsIntegration:
    """Test integration scenarios for utility functions."""

    def test_sensitive_filter_with_real_logging(self):
        """SensitiveHeadersFilter works with real logging setup."""
        filter_instance = SensitiveHeadersFilter()

        # Create a mock LogRecord directly
        mock_record = Mock()
        mock_record.args = {
            "headers": {
                "x-api-key": "secret-key-123",
                "content-type": "application/json",
            }
        }

        # Process the record through our filter
        result = filter_instance.filter(mock_record)

        # Verify filter returns True (allowing the log)
        assert result is True

        # Verify sensitive data was redacted
        assert mock_record.args["headers"]["x-api-key"] == "<redacted>"
        assert mock_record.args["headers"]["content-type"] == "application/json"

    def test_typeguards_with_complex_data_structures(self):
        """Type guards work correctly with complex nested structures."""
        complex_structure = {
            "metadata": {
                "headers": {"authorization": "Bearer token", "x-api-key": "secret"},
                "params": ["param1", "param2"],
            },
            "data": {"nested": {"deep": {"value": 42}}},
        }

        # Test type guards at different levels
        assert is_dict(complex_structure)
        assert is_mapping(complex_structure)
        assert is_dict(complex_structure["metadata"])
        assert is_dict(complex_structure["metadata"]["headers"])
        assert not is_dict(complex_structure["metadata"]["params"])

        # Test with the filter
        filter_instance = SensitiveHeadersFilter()
        mock_record = Mock(spec=logging.LogRecord)
        mock_record.args = complex_structure["metadata"]

        result = filter_instance.filter(mock_record)

        assert result is True
        assert mock_record.args["headers"]["authorization"] == "<redacted>"
        assert mock_record.args["headers"]["x-api-key"] == "<redacted>"
