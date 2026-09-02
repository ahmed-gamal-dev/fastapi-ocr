"""Application configuration.

Every operationally relevant knob is an environment variable. No secrets in code.
Defaults are chosen so the service runs from a plain virtualenv with no setup.
"""

from __future__ import annotations

import os
import tempfile
from functools import lru_cache
from typing import Annotated, List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

# List settings are written as comma separated values in the environment.
# ``NoDecode`` stops pydantic-settings from trying to JSON-parse them first, so
# the validator below is what actually interprets them.
CSVList = Annotated[List[str], NoDecode]


def _split_csv(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---------------------------------------------------------------- service
    APP_NAME: str = "document-ocr-service"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = False

    # ------------------------------------------------------------------ auth
    # Comma separated to allow key rotation without downtime.
    OCR_API_KEY: str = ""
    API_KEY_HEADER: str = "X-API-Key"

    # ---------------------------------------------------------------- uploads
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # bytes
    MIN_UPLOAD_SIZE: int = 256
    # Guard against decompression bombs: a 50MP image is already absurd for a
    # document scan and would otherwise happily allocate gigabytes.
    MAX_IMAGE_PIXELS: int = 50_000_000
    ALLOWED_MIME_TYPES: CSVList = Field(
        default_factory=lambda: [
            "image/jpeg",
            "image/png",
            "image/webp",
            "image/bmp",
            "image/tiff",
        ]
    )
    ALLOWED_EXTENSIONS: CSVList = Field(
        default_factory=lambda: [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"]
    )
    TEMP_DIR: str = os.path.join(tempfile.gettempdir(), "document-ocr")
    # Off by default. Privacy first: images are never persisted unless asked.
    STORE_UPLOADS: bool = False
    STORE_UPLOADS_DIR: str = "./data/uploads"

    # -------------------------------------------------------------------- ocr
    OCR_PROVIDER: str = "paddle"
    # Languages the pipeline recognises. "en" drives MRZ + latin fields,
    # "arabic" drives the Arabic name / place fields.
    OCR_LANGUAGES: CSVList = Field(default_factory=lambda: ["en", "arabic"])
    OCR_LANGUAGE: Optional[str] = None  # legacy alias, CSV
    OCR_USE_GPU: bool = False
    OCR_DET_LIMIT_SIDE_LEN: int = 1600
    OCR_DROP_SCORE: float = 0.35
    OCR_MODEL_DIR: Optional[str] = None
    # Override the detection/recognition models PaddleOCR 3.x picks by default.
    # The stock choice is accuracy-first; the "mobile" variants are markedly
    # faster on CPU. Empty means "use the engine's default for the language".
    OCR_DET_MODEL_NAME: Optional[str] = None
    OCR_REC_MODEL_NAME: Optional[str] = None
    # Hard ceiling on concurrent OCR inferences inside one worker process.
    OCR_MAX_CONCURRENCY: int = 1
    # Inference threads inside one predictor. 1 keeps concurrent requests from
    # fighting over the same cores; raise it to cut single-request latency on a
    # machine that handles one document at a time.
    OCR_CPU_THREADS: int = 1
    OCR_TIMEOUT_SECONDS: float = 45.0
    OCR_WARMUP_ON_STARTUP: bool = True
    # Return every raw recognised block in the response (useful for debugging
    # and for callers that want to do their own layout analysis).
    INCLUDE_RAW_BLOCKS_DEFAULT: bool = True

    # ---------------------------------------------------------------- imaging
    IMAGE_MAX_DIMENSION: int = 2200
    IMAGE_MIN_DIMENSION: int = 320
    ENABLE_PERSPECTIVE_CORRECTION: bool = True
    ENABLE_ORIENTATION_CORRECTION: bool = True
    ENABLE_DESKEW: bool = True
    # How much of the frame the document must fill before it is cropped out of
    # it. The default is deliberately cautious: cropping the wrong contour
    # loses content outright. The ?crop=true pass lowers it, because a caller
    # asking for that has already seen the ordinary pass come back short.
    PERSPECTIVE_MIN_AREA_RATIO: float = 0.35
    CROP_MIN_AREA_RATIO: float = 0.12
    # Longest side the cropped document is enlarged to. Small print in a
    # hand-held photograph is only small relative to the background around it;
    # once that is gone the pixels can be spent on the page itself.
    CROP_TARGET_SIDE_LEN: int = 1600
    CROP_MAX_UPSCALE: float = 3.0

    # ------------------------------------------------------------------- mrz
    # Machine readable zone detection/parsing (ICAO 9303 TD1/TD2/TD3, MRV).
    ENABLE_MRZ: bool = True
    MRZ_UPSCALE_FACTOR: float = 2.0

    # ------------------------------------------------------------- thresholds
    # Below this a field is still returned but flagged as low confidence.
    MIN_FIELD_CONFIDENCE: float = 0.40
    # Below this the whole response is marked as LOW_CONFIDENCE in warnings.
    MIN_OVERALL_CONFIDENCE: float = 0.55

    # ------------------------------------------------------------------- http
    ALLOWED_ORIGINS: CSVList = Field(default_factory=list)
    ALLOWED_HOSTS: CSVList = Field(default_factory=lambda: ["*"])
    REQUEST_TIMEOUT_SECONDS: float = 60.0
    ROOT_PATH: str = ""
    # Only trust X-Forwarded-* when the service actually sits behind a proxy;
    # otherwise any caller could spoof its address past the rate limiter.
    TRUST_PROXY_HEADERS: bool = False
    DOCS_ENABLED: bool = True

    # ----------------------------------------------------------- rate limiting
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_REQUESTS: int = 60
    RATE_LIMIT_WINDOW_SECONDS: int = 60
    REDIS_URL: Optional[str] = None  # e.g. redis://redis:6379/0

    # ---------------------------------------------------------------- logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"  # json | console
    # Never flip this on in production: it unmasks PII in the logs.
    LOG_SENSITIVE_DATA: bool = False

    # --------------------------------------------------------------- runtime
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    WORKERS: int = 1

    @field_validator(
        "ALLOWED_ORIGINS",
        "ALLOWED_HOSTS",
        "ALLOWED_MIME_TYPES",
        "ALLOWED_EXTENSIONS",
        "OCR_LANGUAGES",
        mode="before",
    )
    @classmethod
    def _coerce_list(cls, v):
        """Accept ``a,b,c`` and ``["a","b","c"]`` alike."""
        if isinstance(v, str):
            text = v.strip()
            if text.startswith("["):
                import json

                try:
                    return json.loads(text)
                except ValueError:
                    return _split_csv(text.strip("[]"))
            return _split_csv(text)
        return v

    @field_validator("LOG_LEVEL", mode="before")
    @classmethod
    def _upper(cls, v):
        return str(v).upper()

    def model_post_init(self, __context) -> None:  # noqa: D105
        # Backwards compatible alias: OCR_LANGUAGE=ar,en
        if self.OCR_LANGUAGE:
            langs = [normalise_lang(x) for x in _split_csv(self.OCR_LANGUAGE)]
            object.__setattr__(self, "OCR_LANGUAGES", langs)
        else:
            object.__setattr__(
                self, "OCR_LANGUAGES", [normalise_lang(x) for x in self.OCR_LANGUAGES]
            )

    @property
    def api_keys(self) -> List[str]:
        return _split_csv(self.OCR_API_KEY)

    @property
    def auth_enabled(self) -> bool:
        return bool(self.api_keys)


def normalise_lang(lang: str) -> str:
    """Map friendly language codes onto PaddleOCR model names."""
    lang = lang.strip().lower()
    return {
        "ar": "arabic",
        "ara": "arabic",
        "arabic": "arabic",
        "en": "en",
        "eng": "en",
        "english": "en",
        "latin": "en",
    }.get(lang, lang)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

os.makedirs(settings.TEMP_DIR, exist_ok=True)
