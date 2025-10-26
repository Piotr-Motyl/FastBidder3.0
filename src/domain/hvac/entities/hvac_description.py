"""
HVACDescription Entity

Core domain entity representing an HVAC/plumbing product description.
This is the central business object in the FastBidder domain model.

Responsibility:
    - Represent HVAC item descriptions (from WF or REF files)
    - Store extracted parameters (DN, PN, material, type)
    - Track matching state (score, matched reference)
    - Encapsulate business validation rules

Architecture Notes:
    - Entity (has identity, has lifecycle, mutable)
    - Part of HVAC subdomain
    - Uses Pydantic for validation (NOT frozen - entity is mutable)
    - Phase 1: Minimal version (only essential fields for happy path)
"""

from typing import Any, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator


class HVACDescription(BaseModel):
    """
    Domain entity representing an HVAC/plumbing product description.

    This entity is used for both:
    - Working file (WF) descriptions: Items to be priced
    - Reference catalog (REF) descriptions: Catalog with prices

    Lifecycle:
    1. Created from Excel row with raw text
    2. Parameters extracted (DN, PN, material, etc.) → extracted_parameters populated
    3. Matched against catalog → match_score and matched_reference_id set
    4. Serialized for result export

    Attributes:
        id: Unique identifier for this description
            - Generated automatically (uuid4)
            - Used for references and lookups
            - Immutable once created

        raw_text: Description text from Excel (trimmed whitespace)
            - Leading/trailing whitespace automatically removed for consistency
            - Preserves internal formatting and structure
            - Validated to ensure not empty
            - Example input: "  Zawór kulowy DN50  " → stored as "Zawór kulowy DN50"

        extracted_parameters: Dictionary of extracted HVAC parameters
            - Populated by ParameterExtractor domain service
            - Keys: 'dn', 'pn', 'material', 'valve_type', 'drive_type', etc.
            - Values: Parsed values (int for DN/PN, str for types)
            - Empty dict {} if no parameters extracted
            - Example: {'dn': 50, 'pn': 16, 'valve_type': 'kulowy', 'drive': 'elektryczny'}

        match_score: Final matching score (0-100)
            - None before matching (initial state)
            - Set by MatchingEngine after matching
            - Based on hybrid algorithm (40% param + 60% semantic)
            - Only set if match found above threshold

        matched_reference_id: UUID of matched reference description
            - None before matching (initial state)
            - Set to UUID of best match from reference catalog
            - Used to retrieve price and details from reference
            - Only set if match found above threshold

    Examples:
        >>> # Create new description from Excel
        >>> desc = HVACDescription(
        ...     raw_text="  Zawór kulowy DN50 PN16 mosiężny  "
        ... )
        >>> desc.raw_text  # Whitespace trimmed
        'Zawór kulowy DN50 PN16 mosiężny'
        >>> desc.id  # Auto-generated UUID
        UUID('...')
        >>> desc.extracted_parameters
        {}
        >>> desc.has_parameters()
        False

        >>> # After parameter extraction
        >>> desc.extracted_parameters = {'dn': 50, 'pn': 16, 'material': 'brass'}
        >>> desc.has_parameters()
        True
        >>> desc.is_valid()
        True

        >>> # After matching
        >>> desc.match_score = 95.2
        >>> desc.matched_reference_id = UUID("...")

    Business Rules:
        - raw_text cannot be empty or only whitespace
        - extracted_parameters defaults to {} (empty dict)
        - match_score must be in range [0, 100] if present
        - Leading/trailing whitespace is automatically removed from raw_text

    Phase 1 Scope (Minimal):
        - Only essential fields for happy path workflow
        - No price/metadata fields (added in Phase 2)
        - No created_at/updated_at timestamps (added when persistence needed)
        - No validation_errors field (added in Phase 2)
        - No Pydantic model validator for match consistency (added in Phase 2)
    """

    id: UUID = Field(
        default_factory=uuid4, description="Unique identifier for this description"
    )

    raw_text: str = Field(
        ...,
        description="Description text from Excel (leading/trailing whitespace trimmed)",
        min_length=1,
        max_length=1000,
    )

    extracted_parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Extracted HVAC parameters (DN, PN, material, type, etc.)",
    )

    match_score: Optional[float] = Field(
        default=None,
        description="Final matching score (0-100), None if not matched yet",
        ge=0.0,
        le=100.0,
    )

    matched_reference_id: Optional[UUID] = Field(
        default=None,
        description="UUID of matched reference description, None if not matched yet",
    )

    model_config = {
        "frozen": False,  # Entity is mutable (can update match_score, etc.)
        "json_schema_extra": {
            "examples": [
                {
                    "id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                    "raw_text": "Zawór kulowy DN50 PN16 mosiężny z napędem elektrycznym",
                    "extracted_parameters": {
                        "dn": 50,
                        "pn": 16,
                        "material": "brass",
                        "valve_type": "ball",
                        "drive_type": "electric",
                    },
                    "match_score": 95.2,
                    "matched_reference_id": "7b9c3d8e-4f21-4a5b-8c9d-1e2f3a4b5c6d",
                }
            ]
        },
    }

    @field_validator("raw_text")
    @classmethod
    def validate_raw_text_not_empty(cls, v: str) -> str:
        """
        Validate that raw_text is not empty and trim whitespace.

        Business rule: Description must contain meaningful text.
        Normalization: Leading/trailing whitespace is removed for consistency.

        Args:
            v: Raw text value to validate

        Returns:
            Validated raw text with leading/trailing whitespace removed

        Raises:
            ValueError: If text is empty or only whitespace

        Examples:
            >>> # Whitespace trimming
            >>> desc = HVACDescription(raw_text="  Zawór DN50  ")
            >>> desc.raw_text
            'Zawór DN50'

            >>> # Empty text rejected
            >>> desc = HVACDescription(raw_text="   ")
            ValueError: raw_text cannot be empty or only whitespace
        """
        stripped = v.strip()
        if not stripped:
            raise ValueError("raw_text cannot be empty or only whitespace")
        return stripped

    def has_parameters(self) -> bool:
        """
        Check if any parameters have been extracted.

        Used to determine if ParameterExtractor has been run on this description.
        Empty dict means either:
        - Parameters not extracted yet
        - No recognizable parameters in the text

        Returns:
            True if extracted_parameters is not empty, False otherwise

        Examples:
            >>> desc = HVACDescription(raw_text="Zawór DN50")
            >>> desc.has_parameters()
            False
            >>> desc.extracted_parameters = {'dn': 50}
            >>> desc.has_parameters()
            True
        """
        return bool(self.extracted_parameters)

    def is_valid(self) -> bool:
        """
        Check if description has minimum required data for matching.

        Business rule: A valid description must have:
        - Non-empty raw text
        - At least one extracted parameter

        This method is used to filter out invalid descriptions before
        sending them to the MatchingEngine.

        Returns:
            True if description has raw_text and parameters, False otherwise

        Examples:
            >>> desc = HVACDescription(raw_text="Zawór DN50")
            >>> desc.is_valid()
            False  # No parameters extracted yet

            >>> desc.extracted_parameters = {'dn': 50}
            >>> desc.is_valid()
            True  # Has text and parameters
        """
        return bool(self.raw_text.strip()) and self.has_parameters()

    def is_matched(self) -> bool:
        """
        Check if this description has been successfully matched.

        A description is considered matched if:
        - match_score is set (not None)
        - matched_reference_id is set (not None)

        Note: Phase 1 does not enforce consistency between these fields
        (no Pydantic validator). This may be added in Phase 2 if needed.

        Returns:
            True if both match_score and matched_reference_id are set

        Examples:
            >>> desc = HVACDescription(raw_text="Zawór DN50")
            >>> desc.is_matched()
            False

            >>> desc.match_score = 95.2
            >>> desc.matched_reference_id = UUID("...")
            >>> desc.is_matched()
            True
        """
        return self.match_score is not None and self.matched_reference_id is not None

    def to_dict(self) -> dict[str, Any]:
        """
        Convert entity to dictionary representation.

        Useful for:
        - JSON serialization for API responses
        - Celery task results
        - Redis cache storage
        - Excel export
        - Logging and debugging

        Returns:
            Dictionary with all entity fields

        Examples:
            >>> desc = HVACDescription(
            ...     raw_text="Zawór DN50 PN16",
            ...     extracted_parameters={'dn': 50, 'pn': 16},
            ...     match_score=95.2,
            ...     matched_reference_id=UUID("7b9c3d8e-...")
            ... )
            >>> desc.to_dict()
            {
                'id': '3fa85f64-...',
                'raw_text': 'Zawór DN50 PN16',
                'extracted_parameters': {'dn': 50, 'pn': 16},
                'match_score': 95.2,
                'matched_reference_id': '7b9c3d8e-...'
            }
        """
        return {
            "id": str(self.id),
            "raw_text": self.raw_text,
            "extracted_parameters": self.extracted_parameters,
            "match_score": self.match_score,
            "matched_reference_id": (
                str(self.matched_reference_id) if self.matched_reference_id else None
            ),
        }

    @classmethod
    def from_excel_row(cls, raw_text: str) -> "HVACDescription":
        """
        Factory method to create HVACDescription from Excel row.

        This is the primary entry point for creating descriptions from user input.
        Used by ExcelReaderService in Infrastructure layer.

        Args:
            raw_text: Description text from Excel cell (whitespace will be trimmed)

        Returns:
            New HVACDescription instance with:
            - Auto-generated UUID
            - Trimmed raw_text
            - Empty extracted_parameters (to be populated later)
            - None match fields (to be populated after matching)

        Raises:
            ValidationError: If raw_text is empty or invalid

        Examples:
            >>> desc = HVACDescription.from_excel_row(
            ...     "  Zawór kulowy DN50 PN16 mosiężny  "
            ... )
            >>> desc.raw_text
            'Zawór kulowy DN50 PN16 mosiężny'
            >>> desc.has_parameters()
            False
            >>> desc.is_matched()
            False
        """
        return cls(raw_text=raw_text)
