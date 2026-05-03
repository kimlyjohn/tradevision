"""
data/download_dataset.py — Download and extract the Roboflow chart pattern dataset.

This script downloads a chart pattern dataset from Roboflow as a YOLOv8 export.
Use ``data/reorganize_yolov8.py`` afterwards to rebuild the classifier-ready
``data/processed/`` directory.

Usage::

    python data/download_dataset.py

Prerequisites:
    - roboflow Python package installed (pip install roboflow)
    - Roboflow API key configured (set ROBOFLOW_API_KEY environment variable)

Configuration:
    Update the ROBOFLOW_WORKSPACE, ROBOFLOW_PROJECT, and ROBOFLOW_VERSION
    constants below to match your Roboflow project.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# ── Ensure project root is importable ────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402

# ── Roboflow configuration ────────────────────────────────────────────────────
# Update these to match your Roboflow workspace and project
ROBOFLOW_WORKSPACE = os.environ.get("ROBOFLOW_WORKSPACE", "ebtihel")
ROBOFLOW_PROJECT = os.environ.get("ROBOFLOW_PROJECT", "chart-pattern")
ROBOFLOW_VERSION = int(os.environ.get("ROBOFLOW_VERSION", "2"))

# ── Load .env if present ──────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv  # noqa: PLC0415

    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass  # python-dotenv is optional


def download_dataset(
    workspace: str = ROBOFLOW_WORKSPACE,
    project: str = ROBOFLOW_PROJECT,
    version: int = ROBOFLOW_VERSION,
    dest_dir: Path | None = None,
) -> None:
    """Download and extract a Roboflow dataset.

    Args:
        workspace: Roboflow workspace name.
        project: Roboflow project name.
        version: Dataset version number.
        dest_dir: Directory where the extracted files will land.
                  Defaults to config.DATA_RAW_DIR.

    Raises:
        SystemExit: On any unrecoverable error (missing API key, invalid project, etc.)
    """
    if dest_dir is None:
        dest_dir = config.DATA_RAW_DIR

    api_key = os.environ.get("ROBOFLOW_API_KEY", "")

    if not api_key:
        print(
            "❌  Roboflow API key not found.\n\n"
            "Set an environment variable before running the script:\n"
            "  export ROBOFLOW_API_KEY=your_api_key\n\n"
            "Get your API key at: https://roboflow.com/account/api",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        from roboflow import Roboflow  # noqa: PLC0415
    except ImportError:
        print(
            "❌  roboflow package not installed.\n    Run:  pip install roboflow",
            file=sys.stderr,
        )
        sys.exit(1)

    dest_dir.mkdir(parents=True, exist_ok=True)

    print("📡  Downloading dataset from Roboflow")
    print(f"    Workspace: {workspace}")
    print(f"    Project:   {project}")
    print(f"    Version:   {version}")
    print(f"📂  Destination: {dest_dir}\n")

    try:
        rf = Roboflow(api_key=api_key)
        workspace_obj = rf.workspace(workspace)
        project_obj = workspace_obj.project(project)
        dataset = project_obj.version(version).download(
            "yolov8", location=str(dest_dir)
        )

        print("\n✅  Download and extraction complete.")
        print(f"📂  Dataset location: {dataset.location}")

        # Count images in the downloaded dataset
        image_count = sum(
            1
            for _ in dest_dir.rglob("*")
            if _.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        )
        print(f"📊  Total images: {image_count:,}")
        print("\nNext steps:")
        print("  python data/reorganize_yolov8.py")
        print("  python -m tradevision.model.trainer")

    except Exception as exc:  # noqa: BLE001
        exc_str = str(exc)
        if (
            "401" in exc_str
            or "Unauthorized" in exc_str
            or "Invalid API Key" in exc_str
        ):
            print("❌  Roboflow API authentication failed.", file=sys.stderr)
            print("    Verify your ROBOFLOW_API_KEY.", file=sys.stderr)
        elif "404" in exc_str or "Not Found" in exc_str:
            print(
                f"❌  Roboflow project not found: '{workspace}/{project}'",
                file=sys.stderr,
            )
            print("    Double-check the workspace and project names.", file=sys.stderr)
        else:
            print(f"❌  Download failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    download_dataset()
