"""Utils functions for rtasr."""

import asyncio
import urllib.parse
import zipfile
from pathlib import Path
from typing import Any, List, Mapping, Tuple

import aiohttp
from dotenv import dotenv_values


def build_query_string(params: Mapping[str, Any] = None) -> str:
    """
    Build a query string from a dictionary of parameters for API calls.

    Args:
        params (Mapping[str, Any]):
            Dictionary of parameters for API calls.

    Returns:
        Query string.
    """
    if params is None:
        params = {}

    filtered_parameters: List[Tuple[str, str]] = []
    for key, value in params.items():
        if value is None or value == "":
            continue
        else:
            filtered_parameters.append((key, str(value).lower()))

    return (
        "?" + urllib.parse.urlencode(filtered_parameters) if filtered_parameters else ""
    )


async def download_file(
    url: str,
    output_dir: Path,
    session: aiohttp.ClientSession,
    use_cache: bool,
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
        use_cache (bool):
            Whether to use the cache or not.

    Returns:
        Path to output file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / Path(url).name

    if use_cache and output_file.exists():
        return output_file
    else:
        output_file.unlink(missing_ok=True)

    async with session.get(url) as response:
        with output_file.open("wb") as f:
            while True:
                chunk = await response.content.read(1024)

                if not chunk:
                    break

                f.write(chunk)

    return output_file


def get_api_key(provider: str) -> str:
    """Get the API key for the provider.

    All the API keys must be stored in a .env file in the root directory of the
    project.

    Args:
        provider (str):
            The provider to get the API key for.

    Returns:
        The API key for the provider.
    """
    config = dotenv_values(".env")

    key = config[f"{provider.upper()}_API_KEY"]
    if key is None or key == "" or key == "<your key here>":
        raise ValueError(
            f"No API key found for {provider.upper()}. "
            f"Please add `{provider.upper()}_API_KEY` to the `.env` file."
        )

    return key


async def get_files(path: Path) -> Path:
    """
    Get all files in a directory.

    Args:
        path (Path):
            Path to directory to get files from.

    Returns:
        Generator of files in directory.
    """
    for p in path.iterdir():
        if p.is_file():
            yield p


async def unzip_file(zip_path: Path, output_dir: Path, use_cache: bool = True) -> Path:
    """
    Unzip a file to a directory.

    Args:
        zip_path (Path):
            Path to zip file.
        output_dir (Path):
            Path to output directory.
        use_cache (bool):
            Whether to use the cache or not. Defaults to True.

    Returns:
        Path to output directory.
    """

    def extract_zip_sync(zip_path: Path, extract_to: Path) -> None:
        """Synchronous function to extract ZIP."""
        with zipfile.ZipFile(zip_path, "r") as archive:
            archive.extractall(extract_to)

    output_dir.mkdir(parents=True, exist_ok=True)
    if use_cache and (output_dir / zip_path.stem).exists():
        return output_dir / zip_path.stem
    else:
        (output_dir / zip_path.stem).unlink(missing_ok=True)

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, extract_zip_sync, zip_path, output_dir)

    return output_dir / zip_path.stem


def resolve_cache_dir() -> Path:
    """Resolve the cache directory for rtasr."""
    return Path.home() / ".cache" / "rtasr"
