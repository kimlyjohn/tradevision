"""
data/download_dataset.py — Download and extract the Kaggle chart pattern dataset.

DATASET: [INSERT KAGGLE DATASET SLUG HERE e.g. username/chart-patterns-dataset]

Usage::

    python data/download_dataset.py

Prerequisites:
    - kaggle Python package installed  (pip install kaggle)
    - Kaggle API credentials in ~/.kaggle/kaggle.json  OR  .env file
      containing KAGGLE_USERNAME and KAGGLE_KEY
"""

from __future__ import annotations

import os
import sys
import zipfile
from pathlib import Path

# ── Ensure project root is importable ────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402

# ── Load .env if present (before importing kaggle) ────────────────────────────
try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass  # python-dotenv is optional; user may configure env vars manually


def _configure_kaggle_credentials() -> None:
    """Set KAGGLE_USERNAME / KAGGLE_KEY env vars from os.environ.

    The Kaggle client reads these vars automatically.  If they are not set
    and ~/.kaggle/kaggle.json also doesn't exist, a helpful error is raised.
    """
    username = os.environ.get("KAGGLE_USERNAME", "")
    key = os.environ.get("KAGGLE_KEY", "")

    if not username or not key:
        kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
        if not kaggle_json.exists():
            print(
                "❌  Kaggle credentials not found.\n\n"
                "Option A — Add to .env file:\n"
                "  KAGGLE_USERNAME=your_username\n"
                "  KAGGLE_KEY=your_api_key\n\n"
                "Option B — Create ~/.kaggle/kaggle.json:\n"
                '  {"username":"your_username","key":"your_api_key"}\n\n'
                "Get your API key at: https://www.kaggle.com/account",
                file=sys.stderr,
            )
            sys.exit(1)
    else:
        os.environ["KAGGLE_USERNAME"] = username
        os.environ["KAGGLE_KEY"] = key


def download_dataset(
    dataset_slug: str = config.KAGGLE_DATASET_SLUG,
    dest_dir: Path = config.DATA_RAW_DIR,
) -> None:
    """Download and extract a Kaggle dataset into *dest_dir*.

    Args:
        dataset_slug: Kaggle slug in ``owner/dataset-name`` format.
        dest_dir: Directory where the extracted files will land.

    Raises:
        SystemExit: On any unrecoverable error (missing creds, not found, etc.)
    """
    _configure_kaggle_credentials()

    # Validate slug placeholder hasn't been left as-is
    if "YOUR_KAGGLE" in dataset_slug:
        print(
            "❌  Please update KAGGLE_DATASET_SLUG in config.py with the real\n"
            "    Kaggle dataset slug, e.g.  'username/chart-patterns-dataset'",
            file=sys.stderr,
        )
        sys.exit(1)

    if not dataset_slug:
        print(
            "❌  KAGGLE_DATASET_SLUG is not configured.\n"
            "    Set it in your environment or `.env` file, e.g.:\n"
            "    KAGGLE_DATASET_SLUG=username/chart-patterns-dataset",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi  # noqa: PLC0415
    except ImportError:
        print(
            "❌  kaggle package not installed.\n"
            "    Run:  pip install kaggle",
            file=sys.stderr,
        )
        sys.exit(1)

    dest_dir.mkdir(parents=True, exist_ok=True)

    print(f"📡  Downloading dataset: {dataset_slug}")
    print(f"📂  Destination:         {dest_dir}\n")

    try:
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(
            dataset_slug,
            path=str(dest_dir),
            unzip=False,
            quiet=False,
            force=False,
        )
    except Exception as exc:  # noqa: BLE001
        exc_str = str(exc)
        if "404" in exc_str or "Not Found" in exc_str:
            print(f"❌  Dataset not found: '{dataset_slug}'", file=sys.stderr)
            print("    Double-check the slug at https://www.kaggle.com/datasets",
                  file=sys.stderr)
        elif "401" in exc_str or "Unauthorized" in exc_str:
            print("❌  Kaggle API authentication failed.", file=sys.stderr)
            print("    Verify your KAGGLE_USERNAME and KAGGLE_KEY.", file=sys.stderr)
        else:
            print(f"❌  Download failed: {exc}", file=sys.stderr)
        sys.exit(1)

    # ── Extract ────────────────────────────────────────────────────────────
    # kaggle may have already unzipped; check for zip file
    zip_files = list(dest_dir.glob("*.zip"))
    if zip_files:
        zip_file = zip_files[0]
        print(f"\n📦  Extracting {zip_file.name} …")
        with zipfile.ZipFile(zip_file, "r") as zf:
            zf.extractall(dest_dir)
        zip_file.unlink()
        print("✅  Extraction complete.")
    else:
        print("✅  Files already extracted (no zip found).")

    # ── Summary ───────────────────────────────────────────────────────────
    image_count = sum(1 for _ in dest_dir.rglob("*")
                      if _.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    print(f"\n📊  Total images in {dest_dir}: {image_count:,}")
    print("\nNext step → train the model:")
    print("  python -m tradevision.model.trainer")


if __name__ == "__main__":
    download_dataset()
