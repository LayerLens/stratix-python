from __future__ import annotations

import logging
from typing import Mapping
from typing_extensions import TypeGuard, override

SENSITIVE_HEADERS = {"x-api-key", "authorization"}


def is_dict(obj: object) -> TypeGuard[dict[object, object]]:
    return isinstance(obj, dict)


def is_mapping(obj: object) -> TypeGuard[Mapping[str, object]]:
    return isinstance(obj, Mapping)


class SensitiveHeadersFilter(logging.Filter):
    @override
    def filter(self, record: logging.LogRecord) -> bool:
        if is_dict(record.args) and "headers" in record.args and is_dict(record.args["headers"]):
            headers = record.args["headers"] = {**record.args["headers"]}
            for header in headers:
                if str(header).lower() in SENSITIVE_HEADERS:
                    headers[header] = "<redacted>"
        return True
