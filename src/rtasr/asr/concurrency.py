"""Define the handler class for dealing with API concurrency limits."""

import asyncio
from typing import Union


class ConcurrencyToken:
    """A concurrency token."""

    def __init__(self, value: int) -> None:
        """Initialize the concurrency token."""
        self._value = value

    @property
    def value(self) -> int:
        """Return the value of the token."""
        return self._value


class ConcurrencyHandler:
    """Handle concurrency limits for API requests."""

    def __init__(self, limit: Union[int, None] = None):
        """Initialize the concurrency handler."""
        self.limit = limit
        self.queue = asyncio.Queue(maxsize=limit)

        if self.limit is not None:
            self._fill_queue()

    async def get(self):
        """Get a concurrency token."""
        if self.limit is None:
            return None

        return await self.queue.get()

    def put(self, token: ConcurrencyToken):
        """Put a concurrency token back in the queue."""
        if self.limit is not None:
            self.queue.put_nowait(token)

    def _fill_queue(self):
        """Fill the queue with tokens."""
        for i in range(self.limit):
            self.queue.put_nowait(ConcurrencyToken(i))
