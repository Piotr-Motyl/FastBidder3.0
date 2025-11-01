"""
HVAC Description Repository Implementation

Concrete implementation of HVACDescriptionRepositoryProtocol from Domain Layer.
Uses Redis for temporary caching during job processing (session storage).

Responsibility:
    - Implement Domain repository interface
    - Store/retrieve HVACDescription entities in Redis
    - Session-based cache (TTL 1 hour)
    - Serialize/deserialize entities to/from JSON

Architecture Notes:
    - Infrastructure Layer (implements Domain interface)
    - Dependency Inversion: Domain defines interface, Infrastructure implements
    - Phase 1: Redis-only (no PostgreSQL)
    - Phase 2: Add PostgreSQL for long-term storage
"""

import json
import os
from typing import Optional
from uuid import UUID

from redis import Redis
from redis.exceptions import RedisError

from src.domain.hvac import HVACDescription


class HVACDescriptionRepository:
    """
    Redis-based implementation of HVACDescriptionRepositoryProtocol.

    This repository provides temporary storage for HVAC descriptions
    during job processing. Data is cached in Redis with TTL.

    Storage Strategy:
        - Key pattern: "hvac:description:{uuid}"
        - Value: JSON serialized HVACDescription.to_dict()
        - TTL: 1 hour (auto-expire after job completion)

    Use Cases:
        - Store reference catalog descriptions (from REF file)
        - Cache working file descriptions during matching
        - Retrieve matched descriptions for result generation

    Phase 1 Scope:
        - Implements HVACDescriptionRepositoryProtocol (minimal CRUD)
        - Redis-only storage (no PostgreSQL)
        - Embedded Redis connection (no separate client)

    Examples:
        >>> repo = HVACDescriptionRepository()
        >>>
        >>> # Save description
        >>> desc = HVACDescription.from_excel_row("Zawór DN50")
        >>> await repo.save(desc)
        >>>
        >>> # Retrieve by ID
        >>> retrieved = await repo.get_by_id(desc.id)
        >>> print(retrieved.raw_text)
        'Zawór DN50'
        >>>
        >>> # Get all (e.g., reference catalog)
        >>> all_refs = await repo.get_all()
        >>> print(f"Found {len(all_refs)} descriptions")
    """

    def __init__(
        self,
        redis_host: Optional[str] = None,
        redis_port: Optional[int] = None,
        redis_db: int = 0,
    ) -> None:
        """
        Initialize repository with Redis connection.

        Phase 1: Simple Redis connection (same as RedisProgressTracker).
        Phase 2: Will use shared Redis client/connection pool.

        Args:
            redis_host: Redis hostname (default from env: REDIS_HOST)
            redis_port: Redis port (default from env: REDIS_PORT)
            redis_db: Redis database number (default 0)

        Raises:
            RedisError: If connection cannot be established
        """
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
        """
        Generate Redis key for HVACDescription.

        Args:
            description_id: UUID of the description

        Returns:
            Redis key in format "hvac:description:{uuid}"

        Examples:
            >>> repo._get_key(UUID("3fa85f64-5717-4562-b3fc-2c963f66afa6"))
            'hvac:description:3fa85f64-5717-4562-b3fc-2c963f66afa6'
        """
        return f"hvac:description:{str(description_id)}"

    async def save(self, description: HVACDescription) -> None:
        """
        Store HVACDescription in Redis (implements Protocol method).

        Serializes entity to JSON and stores with TTL.
        Overwrites existing description with same UUID.

        Args:
            description: HVACDescription entity to store

        Raises:
            RedisError: If Redis operation fails

        Examples:
            >>> repo = HVACDescriptionRepository()
            >>> desc = HVACDescription(
            ...     raw_text="Zawór DN50",
            ...     extracted_parameters={'dn': 50}
            ... )
            >>> await repo.save(desc)
        """
        raise NotImplementedError(
            "save() to be implemented in Phase 3. "
            "Will serialize HVACDescription.to_dict() to JSON and store in Redis with TTL."
        )

    async def get_by_id(self, description_id: UUID) -> Optional[HVACDescription]:
        """
        Retrieve HVACDescription by UUID (implements Protocol method).

        Deserializes JSON from Redis and reconstructs entity.

        Args:
            description_id: UUID of the description to retrieve

        Returns:
            HVACDescription if found, None if not found or expired

        Raises:
            RedisError: If Redis operation fails
            ValidationError: If stored JSON is invalid

        Examples:
            >>> repo = HVACDescriptionRepository()
            >>> desc = await repo.get_by_id(UUID("3fa85f64-..."))
            >>> if desc:
            ...     print(f"Found: {desc.raw_text}")
            ... else:
            ...     print("Description not found or expired")
        """
        raise NotImplementedError(
            "get_by_id() to be implemented in Phase 3. "
            "Will retrieve JSON from Redis and deserialize to HVACDescription."
        )

    async def get_all(self) -> list[HVACDescription]:
        """
        Retrieve all HVACDescriptions (implements Protocol method).

        Scans Redis for all description keys and deserializes.
        Used to load reference catalog for matching.

        Returns:
            List of all HVACDescription entities in storage
            Empty list if none found

        Raises:
            RedisError: If Redis operation fails

        Examples:
            >>> repo = HVACDescriptionRepository()
            >>> ref_catalog = await repo.get_all()
            >>> print(f"Found {len(ref_catalog)} reference descriptions")
            >>> for desc in ref_catalog:
            ...     print(f"- {desc.raw_text}")

        Performance Notes:
            - Phase 1: SCAN pattern "hvac:description:*" (OK for 100-400 items)
            - Phase 5: Optimize for 10,000+ items (pagination, indexing)
        """
        raise NotImplementedError(
            "get_all() to be implemented in Phase 3. "
            "Will scan Redis keys matching 'hvac:description:*' pattern and deserialize all."
        )
