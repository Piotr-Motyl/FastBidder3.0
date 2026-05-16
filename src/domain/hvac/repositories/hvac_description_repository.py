"""
HVACDescriptionRepository Interface — Protocol-based contract for persistence.
Domain defines the interface; Infrastructure (Redis/PostgreSQL) implements it.
"""

from typing import Optional, Protocol
from uuid import UUID

from ..entities.hvac_description import HVACDescription


class HVACDescriptionRepositoryProtocol(Protocol):
    """
    Protocol for HVACDescription persistence.

    Domain Layer defines this interface; Infrastructure Layer (Redis, PostgreSQL) implements it.
    Injected into Application Layer use cases via dependency injection.
    """

    async def save(self, description: HVACDescription) -> None:
        """Store an HVAC description. Overwrites existing entry with same UUID."""
        ...

    async def get_by_id(self, description_id: UUID) -> Optional[HVACDescription]:
        """Retrieve by UUID. Returns None if not found or TTL expired."""
        ...

    async def get_all(self) -> list[HVACDescription]:
        """
        Retrieve all stored HVAC descriptions.

        Order is not guaranteed. Returns empty list if storage is empty.
        Note: designed for catalogs of 100-400 items; needs pagination beyond that.
        """
        ...
