"""
ParameterExtractor Domain Service Contract

Protocol (interface) for HVAC parameter extraction from text descriptions.
Defines the contract for extracting technical parameters using regex patterns
and domain dictionaries.

Part of: Task 2.1.3 - ParameterExtractor domain service
Phase: 2.1 - Domain Layer Details

Architecture Notes:
- Protocol pattern (not ABC) for dependency inversion
- Returns ExtractedParameters Value Object with raw values
- Stateless service - no internal state between calls
- Implements Single Responsibility Principle (only extraction, no validation)

Design Decisions:
- Returns raw values (int/str) not Value Objects (decided in Pytanie 5)
- Simple confidence scoring 0.0-1.0 (decided in Pytanie 2)
- Returns empty ExtractedParameters when nothing found (decided in Pytanie 4)
- 10-15 terms per dictionary category (decided in Pytanie 3)
"""

from typing import Protocol, Optional
from src.domain.hvac.value_objects.extracted_parameters import ExtractedParameters


class ParameterExtractorProtocol(Protocol):
    """
    Protocol defining interface for HVAC parameter extraction service.

    This service extracts technical parameters from HVAC equipment descriptions
    using regex patterns and domain-specific dictionaries. It is the first step
    in the hybrid matching pipeline (40% parameter matching, 60% semantic matching).

    Responsibilities:
    - Extract DN (Diameter Nominal) from text
    - Extract PN (Pressure Nominal) from text
    - Identify valve types using dictionary matching
    - Identify materials using dictionary matching
    - Identify drive types using dictionary matching
    - Extract voltage for electric drives
    - Identify manufacturer if present
    - Calculate confidence scores for each extraction

    Does NOT:
    - Validate business rules (that's for Value Objects or Entities)
    - Perform semantic matching (that's for MatchingEngine)
    - Store state between calls (stateless service)

    Usage Example:
        extractor = ConcreteParameterExtractor()

        text = "Zawór kulowy DN50 PN16 z napędem elektrycznym 230V"
        params = extractor.extract_parameters(text)

        if params.has_critical_parameters():
            print(f"Found DN{params.dn} PN{params.pn}")
            print(f"Confidence: {params.get_average_confidence():.2f}")
    """

    def extract_parameters(self, text: str) -> ExtractedParameters:
        """
        Extract all HVAC parameters from text description.

        This is the main entry point for parameter extraction. It orchestrates
        all specific extraction methods and combines results into a single
        ExtractedParameters object.

        Args:
            text: HVAC equipment description text (typically from Excel cell)
                  Can be in Polish or mixed PL/EN
                  Expected length: 10-500 characters

        Returns:
            ExtractedParameters object with:
            - All extracted parameters (dn, pn, valve_type, etc.)
            - Confidence scores for each parameter
            - Empty object if no parameters found (all fields None)

        Business Rules:
        - Text is normalized (lowercase, trimmed) before processing
        - Extraction order matters: DN/PN first, then type, then material
        - Confidence 1.0 = exact regex/dict match, 0.5-0.9 = partial/synonym
        - Returns empty ExtractedParameters (not None) when nothing found

        Algorithm (Happy Path):
        1. Normalize input text (lowercase, strip, remove double spaces)
        2. Extract DN using regex patterns
        3. Extract PN using regex patterns
        4. Extract valve type using dictionary matching
        5. Extract material using dictionary matching
        6. Extract drive type using dictionary matching
        7. Extract voltage if drive is electric
        8. Extract manufacturer using dictionary matching
        9. Compile confidence scores
        10. Return ExtractedParameters object

        Performance:
        - Expected runtime: <10ms per description
        - No external API calls (all local regex/dict matching)

        Example:
            >>> text = "Montaż zaworu kulowego DN50 PN16 mosiądz"
            >>> params = extractor.extract_parameters(text)
            >>> params.dn
            50
            >>> params.pn
            16
            >>> params.valve_type
            'kulowy'
            >>> params.material
            'mosiądz'
            >>> params.get_confidence('dn')
            1.0
        """
        ...

    def extract_dn(self, text: str) -> tuple[Optional[int], float]:
        """
        Extract DN (Diameter Nominal) value from text.

        Uses regex patterns to find DN specifications in various formats:
        - Standard: DN50, DN 50, dn50
        - With separators: DN-50, DN=50
        - Diameter symbol: Ø50
        - Word format: "średnica 50mm"

        Args:
            text: Text to search for DN value

        Returns:
            Tuple of (dn_value, confidence_score)
            - dn_value: Extracted DN as integer, or None if not found
            - confidence_score: 1.0 for exact match, 0.8 for word pattern, 0.0 if not found

        Business Rules:
        - Only extracts standard DN values (15, 20, 25, 32, 40, 50, ...)
        - Non-standard values are extracted but confidence is lower
        - First match wins (doesn't search for multiple DNs)

        Example:
            >>> extractor.extract_dn("Zawór DN50 PN16")
            (50, 1.0)
            >>> extractor.extract_dn("Rura średnica 100mm")
            (100, 0.8)
            >>> extractor.extract_dn("Zawór bez DN")
            (None, 0.0)
        """
        ...

    def extract_pn(self, text: str) -> tuple[Optional[int], float]:
        """
        Extract PN (Pressure Nominal) value from text.

        Uses regex patterns to find PN specifications in various formats:
        - Standard: PN16, PN 16, pn16
        - With separators: PN-16, PN=16
        - Word format: "ciśnienie 16 bar"

        Args:
            text: Text to search for PN value

        Returns:
            Tuple of (pn_value, confidence_score)
            - pn_value: Extracted PN as integer, or None if not found
            - confidence_score: 1.0 for exact match, 0.8 for word pattern, 0.0 if not found

        Business Rules:
        - Only extracts standard PN values (6, 10, 16, 25, 40, 63, 100)
        - Non-standard values are extracted but confidence is lower
        - First match wins

        Example:
            >>> extractor.extract_pn("Zawór DN50 PN16")
            (16, 1.0)
            >>> extractor.extract_pn("Ciśnienie 10 bar")
            (10, 0.8)
        """
        ...

    def extract_valve_type(self, text: str) -> tuple[Optional[str], float]:
        """
        Extract valve type from text using dictionary matching.

        Matches against VALVE_TYPES dictionary and resolves synonyms:
        - Direct matches: "kulowy", "zwrotny", "grzybkowy"
        - Synonyms: "kurek kulowy" -> "kulowy", "klapka zwrotna" -> "zwrotny"

        Args:
            text: Text to search for valve type

        Returns:
            Tuple of (valve_type, confidence_score)
            - valve_type: Canonical valve type name, or None if not found
            - confidence_score: 1.0 for exact match, 0.9 for synonym, 0.0 if not found

        Business Rules:
        - Case-insensitive matching
        - Synonym resolution to canonical forms
        - Longest match wins (e.g., "zawór kulowy" beats "zawór")
        - Returns canonical form (not original text)

        Example:
            >>> extractor.extract_valve_type("Zawór kulowy DN50")
            ('kulowy', 1.0)
            >>> extractor.extract_valve_type("Kurek kulowy")
            ('kulowy', 0.9)  # Synonym match
        """
        ...

    def extract_material(self, text: str) -> tuple[Optional[str], float]:
        """
        Extract material from text using dictionary matching.

        Matches against MATERIALS dictionary and resolves synonyms:
        - Metals: "mosiądz", "stal", "stal nierdzewna", "żeliwo"
        - Plastics: "PP-R", "PVC", "PE", "PE-X"

        Args:
            text: Text to search for material

        Returns:
            Tuple of (material, confidence_score)
            - material: Canonical material name, or None if not found
            - confidence_score: 1.0 for exact match, 0.9 for synonym, 0.0 if not found

        Business Rules:
        - Case-insensitive matching
        - Synonym resolution (e.g., "mosiężny" -> "mosiądz")
        - Returns canonical form

        Example:
            >>> extractor.extract_material("Zawór mosiężny DN50")
            ('mosiądz', 0.9)  # Synonym match
            >>> extractor.extract_material("Rura PP-R")
            ('PP-R', 1.0)
        """
        ...

    def extract_drive_type(self, text: str) -> tuple[Optional[str], float]:
        """
        Extract drive/actuation type from text using dictionary matching.

        Matches against DRIVE_TYPES dictionary and resolves synonyms:
        - Manual: "ręczny", "dźwignia", "pokrętło"
        - Electric: "elektryczny", "siłownik elektryczny", "serwomotor"
        - Pneumatic: "pneumatyczny", "siłownik pneumatyczny"

        Args:
            text: Text to search for drive type

        Returns:
            Tuple of (drive_type, confidence_score)
            - drive_type: Canonical drive type name, or None if not found
            - confidence_score: 1.0 for exact match, 0.9 for synonym, 0.0 if not found

        Business Rules:
        - Case-insensitive matching
        - Synonym resolution to canonical forms
        - Returns canonical form

        Example:
            >>> extractor.extract_drive_type("Zawór z siłownikiem elektrycznym")
            ('elektryczny', 0.9)  # Synonym match
        """
        ...

    def extract_voltage(self, text: str) -> tuple[Optional[str], float]:
        """
        Extract voltage from text (relevant for electric drives).

        Uses regex to find voltage specifications:
        - Common formats: 230V, 24V, 400V
        - With space: 230 V

        Args:
            text: Text to search for voltage

        Returns:
            Tuple of (voltage, confidence_score)
            - voltage: Voltage string (e.g., "230V"), or None if not found
            - confidence_score: 1.0 if found, 0.0 if not found

        Business Rules:
        - Only relevant when drive_type is "elektryczny"
        - Returns formatted string with "V" suffix

        Example:
            >>> extractor.extract_voltage("Napęd 230V")
            ('230V', 1.0)
        """
        ...

    def extract_manufacturer(self, text: str) -> tuple[Optional[str], float]:
        """
        Extract manufacturer name from text using dictionary matching.

        Matches against MANUFACTURERS dictionary:
        - International: KSB, Danfoss, Belimo, Grundfos, Wilo, Honeywell
        - Polish/Regional: FERRO, INSTAL, AFRISO

        Args:
            text: Text to search for manufacturer

        Returns:
            Tuple of (manufacturer, confidence_score)
            - manufacturer: Manufacturer name (uppercase), or None if not found
            - confidence_score: 1.0 if found, 0.0 if not found

        Business Rules:
        - Case-insensitive matching
        - Exact match only (no synonyms for manufacturers)
        - Returns uppercase version

        Example:
            >>> extractor.extract_manufacturer("Zawór KSB DN50")
            ('KSB', 1.0)
        """
        ...
