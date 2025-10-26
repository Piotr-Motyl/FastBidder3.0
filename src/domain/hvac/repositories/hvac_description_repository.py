"""
HVACDescriptionRepository Interface

Repository pattern interface for HVACDescription persistence.
Defines the contract for storing and retrieving HVAC descriptions.

Responsibility:
    - Define data access contract (interface)
    - Enable Dependency Inversion (Domain defines, Infrastructure implements)
    - Support testing (easy to mock)
    - Provide minimal CRUD operations for Phase 1

Architecture Notes:
    - Repository Pattern (Martin Fowler)
    - Protocol-based interface (structural typing)
    - Async methods (for future database operations)
    - Implementation in Infrastructure layer (Redis/PostgreSQL)
    - Phase 1: Minimal scope (only what's needed for happy path)
"""

from typing import Optional, Protocol
from uuid import UUID

from ..entities.hvac_description import HVACDescription


class HVACDescriptionRepositoryProtocol(Protocol):
    """
    Protocol defining the contract for HVACDescription persistence.

    This is a repository interface defined in the Domain Layer but implemented
    in the Infrastructure Layer. It follows the Dependency Inversion Principle:
    - Domain Layer defines what it needs (this interface)
    - Infrastructure Layer provides the implementation (Redis, PostgreSQL, etc.)

    Phase 1 Scope (Minimal):
        Only the minimum operations needed for happy path workflow:
        - save(): Store a description (cache in Redis)
        - get_by_id(): Retrieve by UUID (lookup from cache)
        - get_all(): Retrieve all descriptions (for reference catalog)

    Not included in Phase 1 (added in Phase 2+):
        - find_by_parameters(): Query by DN, PN, etc.
        - delete(): Remove description
        - update(): Modify existing description
        - batch operations: save_many(), delete_many()
        - filtering/pagination: find_with_filters(), paginate()

    Usage:
        Repository is injected into Application Layer services:

        >>> # In Application Layer (ProcessMatchingUseCase)
        >>> class ProcessMatchingUseCase:
        ...     def __init__(
        ...         self,
        ...         repository: HVACDescriptionRepositoryProtocol,
        ...         matching_engine: MatchingEngine
        ...     ):
        ...         self.repository = repository
        ...         self.matching_engine = matching_engine
        ...
        ...     async def execute(self, command):
        ...         # Load reference catalog from cache
        ...         ref_descriptions = await self.repository.get_all()
        ...
        ...         # Match working descriptions
        ...         for wf_desc in working_descriptions:
        ...             result = await self.matching_engine.match(
        ...                 wf_desc, ref_descriptions
        ...             )
        ...
        ...             # Save matched description
        ...             if result:
        ...                 wf_desc.match_score = result.score.final_score
        ...                 wf_desc.matched_reference_id = result.matched_item_id
        ...                 await self.repository.save(wf_desc)

    Implementation Notes:
        - Infrastructure layer will implement this Protocol
        - Redis implementation: Cache for session duration (TTL 1h)
        - Future PostgreSQL implementation: Long-term storage
        - All methods are async (even if not strictly necessary in Phase 1)
    """

    async def save(self, description: HVACDescription) -> None:
        """
        Store an HVAC description.

        Used for:
        - Caching reference catalog descriptions (from REF file)
        - Storing working file descriptions after matching
        - Temporary storage during processing session

        Business Rules:
            - Overwrites existing description with same UUID
            - No validation (assumes description is valid)
            - TTL in Redis: 1 hour (configurable)

        Args:
            description: HVACDescription entity to store

        Returns:
            None

        Raises:
            InfrastructureException: If storage operation fails

        Examples:
            >>> repo = get_repository()  # Infrastructure implementation
            >>> desc = HVACDescription.from_excel_row("Zawór DN50")
            >>> await repo.save(desc)
        """
        ...

    async def get_by_id(self, description_id: UUID) -> Optional[HVACDescription]:
        """
        Retrieve an HVAC description by its UUID.

        Used for:
        - Looking up matched reference description (to get price, details)
        - Retrieving cached working descriptions
        - Debugging and verification

        Args:
            description_id: UUID of the description to retrieve

        Returns:
            HVACDescription if found, None if not found or expired

        Raises:
            InfrastructureException: If storage operation fails

        Examples:
            >>> repo = get_repository()
            >>> desc = await repo.get_by_id(UUID("3fa85f64-5717-4562-b3fc-2c963f66afa6"))
            >>> if desc:
            ...     print(f"Found: {desc.raw_text}")
            ... else:
            ...     print("Description not found or expired")
        """
        ...

    async def get_all(self) -> list[HVACDescription]:
        """
        Retrieve all stored HVAC descriptions.

        Used for:
        - Loading reference catalog for matching
        - Batch operations on working file descriptions
        - Exporting results

        Business Rules:
            - Returns all non-expired descriptions from storage
            - Order is not guaranteed (use sorting in Application layer if needed)
            - Empty list if no descriptions stored

        Returns:
            List of all HVACDescription entities in storage
            Empty list if none found

        Raises:
            InfrastructureException: If storage operation fails

        Examples:
            >>> repo = get_repository()
            >>> ref_catalog = await repo.get_all()
            >>> print(f"Found {len(ref_catalog)} reference descriptions")
            >>> for desc in ref_catalog:
            ...     print(f"- {desc.raw_text}")

        Performance Notes:
            - Phase 1 limit: 100-400 descriptions (acceptable to load all)
            - Phase 5+: Will need pagination for 10,000+ descriptions
            - Consider caching the full list in memory for session duration
        """
        ...
