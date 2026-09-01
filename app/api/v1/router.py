"""API v1 router."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.v1 import ocr, version

api_router = APIRouter()

# Unauthenticated: probes and version discovery.
api_router.include_router(version.router)

# Authenticated + rate limited: the dependency lives on the endpoint itself.
api_router.include_router(ocr.router)
