"""Utility functions for evaluation."""

import json

import aiofiles
from aiopath import AsyncPath


async def _store_evaluation_results(
    results: dict,
    save_path: AsyncPath,
) -> None:
    """Store the evaluation results in a JSON file."""
    results.pop("status")

    await save_path.parent.mkdir(parents=True, exist_ok=True)

    async with aiofiles.open(save_path, mode="w") as file:
        await file.write(json.dumps(results, indent=4, ensure_ascii=False))
