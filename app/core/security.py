"""API key verification helpers."""

from __future__ import annotations

import hmac
from typing import Optional

from app.core.config import settings


def verify_api_key(candidate: Optional[str]) -> bool:
    """Constant-time comparison against every configured key.

    Returns True when auth is disabled (no key configured) so the service can be
    run unauthenticated on a private network on purpose, not by accident.
    """
    if not settings.auth_enabled:
        return True
    if not candidate:
        return False
    candidate = candidate.strip()
    matched = False
    for key in settings.api_keys:
        # Do not short-circuit: keep the timing flat across the key list.
        if hmac.compare_digest(candidate, key):
            matched = True
    return matched
