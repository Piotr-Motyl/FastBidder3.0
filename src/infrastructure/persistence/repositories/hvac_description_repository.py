"""
Redis-based session cache for HVACDescription entities during job processing.

Implements HVACDescriptionRepositoryProtocol. Key: "hvac:description:{uuid}", TTL: 1h.
All CRUD methods raise NotImplementedError — not yet needed by production flow.
"""

import os
from typing import Optional
from uuid import UUID

from redis import Redis

from src.domain.hvac.entities.hvac_description import HVACDescription


class HVACDescriptionRepository:
    """
    Redis session cache for HVACDescription entities during matching.

    Key: "hvac:description:{uuid}" | TTL: 1h | Value: JSON of to_dict()
    All CRUD methods raise NotImplementedError — production flow doesn't use this path yet.
    """

    def __init__(
        self,
        redis_host: Optional[str] = None,
        redis_port: Optional[int] = None,
        redis_db: int = 0,
    ) -> None:
        """Connect to Redis. TTL from REDIS_CACHE_TTL env (default 3600s)."""
        self.redis_host = redis_host or os.getenv("REDIS_HOST", "localhost")
        self.redis_port = redis_port or int(os.getenv("REDIS_PORT", "6379"))
        self.redis_db = redis_db

        # Phase 1: Direct Redis connection
        self.redis: Redis = Redis(
            host=self.redis_host,
            port=self.redis_port,
            db=self.redis_db,
            decode_responses=True,
        )

        # TTL for cached descriptions (1 hour)
        self.ttl: int = int(os.getenv("REDIS_CACHE_TTL", "3600"))

    def _get_key(self, description_id: UUID) -> str:
        """Return Redis key "hvac:description:{uuid}"."""
        return f"hvac:description:{str(description_id)}"

    async def save(self, description: HVACDescription) -> None:
        """Not implemented."""
        raise NotImplementedError(
            "save() to be implemented in Phase 3. "
            "Will serialize HVACDescription.to_dict() to JSON and store in Redis with TTL."
        )

    async def get_by_id(self, description_id: UUID) -> Optional[HVACDescription]:
        """Not implemented."""
        raise NotImplementedError(
            "get_by_id() to be implemented in Phase 3. "
            "Will retrieve JSON from Redis and deserialize to HVACDescription."
        )

    async def get_all(self) -> list[HVACDescription]:
        """Not implemented."""
        raise NotImplementedError(
            "get_all() to be implemented in Phase 3. "
            "Will scan Redis keys matching 'hvac:description:*' pattern and deserialize all."
        )
