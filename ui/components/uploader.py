"""
ui/components/uploader.py — Streamlit file uploader widget component.

Renders the upload area, validates the file, and returns raw bytes.
All validation errors are surfaced as friendly st.error() messages
rather than raw exceptions.
"""

from __future__ import annotations

import streamlit as st

import config
from tradevision.utils.validators import (
    CorruptImageError,
    FileTooLargeError,
    UnsupportedFormatError,
    validate_upload,
)
from tradevision.utils.logger import get_logger

logger = get_logger(__name__)


def render_uploader() -> tuple[bytes | None, dict[str, str | float | None]]:
    """Render the file uploader widget and validate the uploaded file.

    Returns:
        Tuple of raw file bytes and UI metadata describing the upload state.
    """
    st.markdown("### Upload chart image")
    st.markdown(
        "Add a candlestick or OHLC chart screenshot to run a single-pass pattern classification. "
        f"Accepted formats: **JPG, JPEG, PNG, WEBP**. Maximum file size: **{config.MAX_FILE_SIZE_MB} MB**."
    )

    uploaded = st.file_uploader(
        label="Drop your chart image here",
        type=[ext.lstrip(".") for ext in config.SUPPORTED_FORMATS],
        accept_multiple_files=False,
        label_visibility="collapsed",
    )

    if uploaded is None:
        return None, {"status": "idle", "file_name": None, "file_size_mb": None}

    max_bytes = config.MAX_FILE_SIZE_MB * 1024 * 1024
    file_size_mb = uploaded.size / (1024 * 1024) if uploaded.size is not None else None
    if uploaded.size is not None and uploaded.size > max_bytes:
        exc = FileTooLargeError(
            f"This file is too large ({uploaded.size / (1024 * 1024):.1f} MB). "
            f"Maximum allowed size is {config.MAX_FILE_SIZE_MB} MB."
        )
        logger.warning("Oversized upload rejected before read: %s (%d bytes)", uploaded.name, uploaded.size)
        st.error(f"File too large. {exc}")
        return None, {"status": "invalid", "file_name": uploaded.name, "file_size_mb": file_size_mb}

    file_bytes = uploaded.read()
    logger.info("File uploaded: name='%s' size=%d bytes", uploaded.name, len(file_bytes))

    try:
        validate_upload(uploaded.name, file_bytes)
    except UnsupportedFormatError as exc:
        logger.warning("Unsupported upload rejected: %s", uploaded.name, exc_info=True)
        st.error(f"Unsupported file type. {exc}")
        return None, {"status": "invalid", "file_name": uploaded.name, "file_size_mb": file_size_mb}
    except FileTooLargeError as exc:
        logger.warning("Large upload rejected after read: %s", uploaded.name, exc_info=True)
        st.error(f"File too large. {exc}")
        return None, {"status": "invalid", "file_name": uploaded.name, "file_size_mb": file_size_mb}
    except CorruptImageError as exc:
        logger.warning("Corrupt upload rejected: %s", uploaded.name, exc_info=True)
        st.error(f"Unreadable image. {exc}")
        return None, {"status": "invalid", "file_name": uploaded.name, "file_size_mb": file_size_mb}

    return file_bytes, {
        "status": "ready",
        "file_name": uploaded.name,
        "file_size_mb": file_size_mb,
    }
