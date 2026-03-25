from __future__ import annotations

import re
from enum import Enum
from datetime import datetime, timezone

from layerlens.attestation._hash import compute_hash, canonical_json


class TestCanonicalJson:
    def test_sorted_keys(self):
        """Key order must not affect output."""
        a = canonical_json({"b": 2, "a": 1})
        b = canonical_json({"a": 1, "b": 2})
        assert a == b

    def test_compact_format(self):
        """No whitespace in output."""
        result = canonical_json({"a": 1, "b": [2, 3]})
        assert " " not in result
        assert result == '{"a":1,"b":[2,3]}'

    def test_nested_structures(self):
        """Nested dicts and lists are handled deterministically."""
        data = {"z": {"y": 1, "x": 2}, "a": [3, 2, 1]}
        result = canonical_json(data)
        assert result == '{"a":[3,2,1],"z":{"x":2,"y":1}}'

    def test_datetime_serialization(self):
        dt = datetime(2026, 3, 23, 12, 0, 0, tzinfo=timezone.utc)
        result = canonical_json({"ts": dt})
        assert "2026-03-23" in result

    def test_enum_serialization(self):
        class Color(Enum):
            RED = "red"

        result = canonical_json({"color": Color.RED})
        assert '"red"' in result


class TestComputeHash:
    def test_format(self):
        """Hash must be 'sha256:' followed by 64 hex chars."""
        h = compute_hash({"test": "data"})
        assert re.match(r"^sha256:[0-9a-f]{64}$", h)

    def test_deterministic(self):
        """Same data always produces the same hash."""
        data = {"key": "value", "num": 42}
        assert compute_hash(data) == compute_hash(data)

    def test_key_order_irrelevant(self):
        """Different key orders produce the same hash."""
        assert compute_hash({"b": 2, "a": 1}) == compute_hash({"a": 1, "b": 2})

    def test_different_data_different_hash(self):
        assert compute_hash({"a": 1}) != compute_hash({"a": 2})

    def test_empty_dict(self):
        h = compute_hash({})
        assert re.match(r"^sha256:[0-9a-f]{64}$", h)
