"""
tradevision/utils/logger.py — Application-wide logging setup.

Every module imports `get_logger(__name__)` to obtain a properly
configured logger that writes to both the console and a rotating
log file under logs/tradevision.log.
"""

import logging
import sys
from logging.handlers import RotatingFileHandler


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger configured for TradeVision.

    The logger is idempotent: calling get_logger() multiple times
    for the same name will not add duplicate handlers.

    Args:
        name: Typically ``__name__`` of the calling module.

    Returns:
        A configured :class:`logging.Logger` instance.
    """
    # Import here to avoid circular imports at module level
    from config import LOGS_DIR, LOG_FILE  # noqa: PLC0415

    logger = logging.getLogger(name)

    if logger.handlers:
        # Already configured – return as-is
        return logger

    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── Console handler ────────────────────────────────────────────────────
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    # ── File handler (rotating, max 5 MB × 3 backups) ─────────────────────
    try:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            LOG_FILE,
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    except (OSError, PermissionError) as exc:
        # Don't crash if log dir is not writable; just warn via console
        logger.warning("Could not create log file at %s: %s", LOG_FILE, exc)

    return logger
