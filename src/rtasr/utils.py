"""Utils functions for rtasr."""

from pathlib import Path

import aiohttp


async def download_file(
    url: str, output_dir: Path, session: aiohttp.ClientSession,
) -> Path:
    """
    Download a file from url to output_dir.

    Args:
        url (str):
            URL to download from.
        output_dir (Path):
            Path to output directory.
        session (aiohttp.ClientSession):
            aiohttp session to use for downloading.`
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / Path(url).name

    async with session.get(url) as response:
        with output_file.open("wb") as f:
            while True:
                chunk = await response.content.read(1024)

                if not chunk:
                    break

                f.write(chunk)

    return output_file


def resolve_cache_dir() -> Path:
    """Resolve the cache directory for rtasr."""
    return Path.home() / ".cache" / "rtasr"
