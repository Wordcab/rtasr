"""Test the utils module."""

import re
from pathlib import Path
from typing import Any, List, Mapping
from unittest.mock import patch

import pytest
import rich

from rtasr.utils import (
    build_query_string,
    create_live_panel,
    get_api_key,
    get_files,
    resolve_cache_dir,
    unzip_file,
)


class TestUtilsModule:
    """Test the utils module."""

    # Sample .env content for mocking
    MOCK_ENV_VALUES = {
        "PROVIDER1_API_KEY": "test_key_1234",
        "PROVIDER2_API_KEY": "<your key here>",
        "PROVIDER3_API_KEY": "",
    }
    # Mock async no-op function
    async def async_noop(*args, **kwargs) -> None:
        """An async no-op function to use as a mock."""
        return

    def test_get_api_key(self) -> None:
        """Test getting an API key."""
        with patch("dotenv.dotenv_values") as mock_dotenv_values:
            mock_dotenv_values.return_value = self.MOCK_ENV_VALUES

            key = get_api_key("provider1")
            assert key == "test_key_1234"

            with pytest.raises(
                ValueError,
                match=re.escape(
                    "No API key found for PROVIDER4. "
                    "Please add `PROVIDER4_API_KEY` to the `.env` file."
                ),
            ):
                get_api_key("provider4")

            with pytest.raises(
                ValueError,
                match=re.escape(
                    "No API key found for PROVIDER2. "
                    "Please add `PROVIDER2_API_KEY` to the `.env` file."
                ),
            ):
                get_api_key("provider2")

            with pytest.raises(
                ValueError,
                match=re.escape(
                    "No API key found for PROVIDER3. "
                    "Please add `PROVIDER3_API_KEY` to the `.env` file."
                ),
            ):
                get_api_key("provider3")

    @pytest.mark.parametrize("part_name", [".cache", "rtasr"])
    def test_resolve_cache_dir(self, part_name: List[str]) -> None:
        """Test the resolve_cache_dir function."""
        cache_dir = resolve_cache_dir()
        expected_path = Path.home() / ".cache" / "rtasr"

        assert isinstance(cache_dir, Path)
        assert cache_dir.name == "rtasr"
        assert cache_dir.parent.name == ".cache"
        assert cache_dir.parent.parent == Path.home()

        assert cache_dir == expected_path
        assert part_name in cache_dir.parts

    @pytest.mark.asyncio
    async def test_get_files(self, tmp_path: Path) -> None:
        """Test the get_files function."""
        dir_with_files = tmp_path / "testdir"
        dir_with_files.mkdir()

        subdir = dir_with_files / "subdir"
        subdir.mkdir()

        empty_dir = tmp_path / "emptydir"
        empty_dir.mkdir()

        file1 = dir_with_files / "file1.txt"
        file1.write_text("content")
        file2 = dir_with_files / "file2.txt"
        file2.write_text("content")

        expected_files = [file2, file1]
        files = [f async for f in get_files(dir_with_files)]
        empty_dir_files = [f async for f in get_files(empty_dir)]

        assert all(isinstance(file, Path) for file in files)
        assert expected_files == files
        assert subdir not in files
        assert len(empty_dir_files) == 0

    @pytest.mark.parametrize(
        "params, expected",
        [
            ({"param1": "Value1", "param2": "Value2"}, "?param1=value1&param2=value2"),
            (
                {"param1": "Value1", "emptyParam": "", "noneParam": None},
                "?param1=value1",
            ),
            ({"param1": "VALUE", "param2": "VaLuE2"}, "?param1=value&param2=value2"),
            ({"diarize": True, "model": "the_best"}, "?diarize=true&model=the_best"),
            ({}, ""),
            (None, ""),
        ],
    )
    def test_build_query_string(self, params: Mapping[str, Any], expected: str) -> None:
        """Test the build_query_string function."""
        query = build_query_string(params)

        assert query == expected

    def test_create_live_panel(self) -> None:
        """Test the create_live_panel function."""
        (
            current_progress,
            step_progress,
            splits_progress,
            progress_group,
        ) = create_live_panel()

        assert current_progress is not None
        assert step_progress is not None
        assert splits_progress is not None
        assert progress_group is not None

        assert isinstance(current_progress, rich.progress.Progress)
        assert isinstance(step_progress, rich.progress.Progress)
        assert isinstance(splits_progress, rich.progress.Progress)
        assert isinstance(progress_group, rich.console.Group)

        assert len(current_progress.tasks) == 0
        assert len(step_progress.tasks) == 0
        assert len(splits_progress.tasks) == 0

    @pytest.mark.asyncio
    async def test_unzip_file_with_cache(self, tmp_path: Path) -> None:
        """Test the unzip_file function."""
        zip_path = tmp_path / "dummy.zip"
        output_dir = tmp_path / "output"
        output_subdir = output_dir / "dummy"

        # Simulate directory already existing
        output_subdir.mkdir(parents=True)

        result = await unzip_file(zip_path, output_dir)
        assert result == output_subdir
        assert output_subdir.exists()

    # @pytest.mark.asyncio
    # @pytest.mark.usefixtures("mock_response")
    # async def test_downoad_file_without_cache(self, tmp_path, mock_response) -> None:
    #     """Test the download_file function without cache."""
    #     url = "http://example.com/mock_file.txt"
    #     output_dir = tmp_path / "output"
    #     output_file = output_dir / "mock_file.txt"

    #     with patch("aiohttp.ClientSession.get", return_value=mock_response) as mock_get:
    #         result = await download_file(url, output_dir, mock_get, use_cache=False)
    #         assert result == output_file
    #         assert output_file.read_text() == "mock_content"
    #         mock_get.assert_called_once_with(url)
