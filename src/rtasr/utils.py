"""Utils functions for rtasr."""

import asyncio
import subprocess
from pathlib import Path
from typing import List


async def async_run_subprocess(command: List[str]) -> tuple:
    """
    Run a subprocess asynchronously.

    Args:
        command (List[str]):
            Command to run asynchronously.

    Returns:
        tuple: Tuple with the return code, stdout and stderr.
    """
    process = await asyncio.create_subprocess_exec(
        *command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    stdout, stderr = await process.communicate()

    return process.returncode, stdout, stderr


def run_subprocess(command: List[str]) -> tuple:
    """
    Run a subprocess synchronously.

    Args:
        command (List[str]):
            Command to run.

    Returns:
        tuple: Tuple with the return code, stdout and stderr.
    """
    process = subprocess.Popen(  # noqa: S603,S607
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    stdout, stderr = process.communicate()

    return process.returncode, stdout, stderr


def download_file(url: str, output_dir: Path) -> Path:
    """
    Download a file from url to output_dir.

    Args:
        url (str):
            URL to download from.
        output_dir (Path):
            Path to output directory.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / Path(url).name

    if not output_file.exists():
        command = ["wget", "-P", str(output_file), url]
        r = asyncio.run(async_run_subprocess(command))

        if r[0] != 0:
            raise RuntimeError(f"Failed to download {url} to {output_file}")
        else:
            print(f"Downloaded {url} to {output_file}")

    return output_file


def resolve_cache_dir() -> Path:
    """Resolve the cache directory for rtasr."""
    return Path.home() / ".cache" / "rtasr"
