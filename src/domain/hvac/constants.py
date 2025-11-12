"""
HVAC Domain Constants

Dictionaries and constants for HVAC equipment terminology used in parameter extraction.
Contains the most common terms in Polish HVAC industry (10-15 per category for PoC).

Part of: Task 2.1.3 - ParameterExtractor domain service
Phase: 2.1 - Domain Layer Details

Note: These are baseline dictionaries for Happy Path. Can be extended in future phases.
"""

from typing import Dict, Set, List


# ============================================================================
# VALVE TYPES - Most common valve types in HVAC installations
# ============================================================================

VALVE_TYPES: Set[str] = {
    # Ball valves (most common)
    "kulowy",
    "kurek kulowy",
    "zawór kulowy",
    "zawór odcinający kulowy",
    # Check valves
    "zwrotny",
    "zawór zwrotny",
    "klapka zwrotna",
    # Globe valves
    "grzybkowy",
    "zawór grzybkowy",
    # Butterfly valves
    "motylkowy",
    "zawór motylkowy",
    # Diaphragm valves
    "membranowy",
    "zawór membranowy",
    # Needle valves
    "iglicowy",
    "zawór iglicowy",
    # Gate valves
    "zasuwowy",
    "zasuwa",
}

# Synonyms mapping - different names for the same valve type
VALVE_SYNONYMS: Dict[str, str] = {
    "kurek kulowy": "kulowy",
    "zawór kulowy": "kulowy",
    "zawór odcinający kulowy": "kulowy",
    "klapka zwrotna": "zwrotny",
    "zawór zwrotny": "zwrotny",
    "zawór grzybkowy": "grzybkowy",
    "zawór motylkowy": "motylkowy",
    "zawór membranowy": "membranowy",
    "zawór iglicowy": "iglicowy",
    "zasuwa": "zasuwowy",
}


# ============================================================================
# MATERIALS - Common materials in HVAC piping and equipment
# ============================================================================

MATERIALS: Set[str] = {
    # Metals
    "mosiądz",
    "mosiężny",
    "stal",
    "stalowy",
    "stal nierdzewna",
    "nierdzewny",
    "żeliwo",
    "żeliwny",
    "miedź",
    "miedziany",
    # Plastics
    "PP-R",
    "PP",
    "polipropylen",
    "PVC",
    "PE",
    "PE-X",
    "polietylen",
}

# Material synonyms and variations
MATERIAL_SYNONYMS: Dict[str, str] = {
    "mosiężny": "mosiądz",
    "stalowy": "stal",
    "nierdzewny": "stal nierdzewna",
    "żeliwny": "żeliwo",
    "miedziany": "miedź",
    "polipropylen": "PP-R",
    "PP": "PP-R",
    "polietylen": "PE",
}


# ============================================================================
# DRIVE/ACTUATION TYPES - How valves are operated
# ============================================================================

DRIVE_TYPES: Set[str] = {
    # Manual
    "ręczny",
    "dźwignia",
    "pokrętło",
    # Electric
    "elektryczny",
    "siłownik elektryczny",
    "napęd elektryczny",
    "serwomotor",
    "servomotor",
    # Pneumatic
    "pneumatyczny",
    "siłownik pneumatyczny",
    "napęd pneumatyczny",
    # Hydraulic (less common in HVAC)
    "hydrauliczny",
}

# Drive type synonyms
DRIVE_SYNONYMS: Dict[str, str] = {
    "siłownik elektryczny": "elektryczny",
    "napęd elektryczny": "elektryczny",
    "serwomotor": "elektryczny",
    "servomotor": "elektryczny",
    "siłownik pneumatyczny": "pneumatyczny",
    "napęd pneumatyczny": "pneumatyczny",
    "dźwignia": "ręczny",
    "pokrętło": "ręczny",
}


# ============================================================================
# MANUFACTURERS - Common HVAC equipment manufacturers
# ============================================================================

MANUFACTURERS: Set[str] = {
    # Major international brands
    "KSB",
    "Danfoss",
    "Belimo",
    "Grundfos",
    "Wilo",
    "Honeywell",
    "Siemens",
    # Polish/Regional brands
    "FERRO",
    "INSTAL",
    "AFRISO",
    # Valve specialists
    "Flamco",
    "Reflex",
    "Caleffi",
}


# ============================================================================
# STANDARD DN VALUES - Valid Diameter Nominal sizes per ISO standards
# ============================================================================

STANDARD_DN_VALUES: List[int] = [
    15,
    20,
    25,
    32,
    40,
    50,
    65,
    80,
    100,
    125,
    150,
    200,
    250,
    300,
    350,
    400,
    450,
    500,
    600,
    700,
    800,
    900,
    1000,
]

# Inch to DN conversion mapping (common in Polish market)
INCH_TO_DN: Dict[str, int] = {
    '1/2"': 15,
    "1/2": 15,
    '3/4"': 20,
    "3/4": 20,
    '1"': 25,
    "1": 25,
    '1 1/4"': 32,
    '1.25"': 32,
    '1 1/2"': 40,
    '1.5"': 40,
    '2"': 50,
    "2": 50,
    '2 1/2"': 65,
    '2.5"': 65,
    '3"': 80,
    "3": 80,
    '4"': 100,
    "4": 100,
}


# ============================================================================
# STANDARD PN VALUES - Valid Pressure Nominal classes per ISO standards
# ============================================================================

STANDARD_PN_VALUES: List[int] = [6, 10, 16, 25, 40, 63, 100]


# ============================================================================
# VOLTAGE STANDARDS - Common electrical voltages in HVAC systems
# ============================================================================

VOLTAGE_STANDARDS: Set[str] = {
    "230V",
    "24V",
    "400V",
    "110V",
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def normalize_term(term: str) -> str:
    """
    Normalize a term for matching (lowercase, strip whitespace).

    Args:
        term: Input term to normalize

    Returns:
        Normalized term (lowercase, stripped)

    Usage:
        Used internally by ParameterExtractor for case-insensitive matching
    """
    return term.lower().strip()


def resolve_synonym(term: str, synonym_dict: Dict[str, str]) -> str:
    """
    Resolve a term to its canonical form using synonym dictionary.

    Args:
        term: Term to resolve
        synonym_dict: Dictionary mapping synonyms to canonical forms

    Returns:
        Canonical form if term is a synonym, otherwise original term

    Example:
        resolve_synonym("kurek kulowy", VALVE_SYNONYMS) -> "kulowy"
    """
    normalized = normalize_term(term)
    return synonym_dict.get(normalized, normalized)


def is_valid_dn(value: int) -> bool:
    """
    Check if DN value is in standard HVAC range.

    Args:
        value: DN value to validate

    Returns:
        True if value is in STANDARD_DN_VALUES

    Business Rule:
        Only standard DN values should be considered valid for matching
    """
    return value in STANDARD_DN_VALUES


def is_valid_pn(value: int) -> bool:
    """
    Check if PN value is in standard pressure class range.

    Args:
        value: PN value to validate

    Returns:
        True if value is in STANDARD_PN_VALUES

    Business Rule:
        Only standard PN values should be considered valid for matching
    """
    return value in STANDARD_PN_VALUES
