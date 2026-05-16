"""
ConcreteParameterExtractor — stateless implementation of ParameterExtractorProtocol.
Uses regex patterns and domain dictionaries. No external API calls; target <5ms per call.
"""

from typing import Optional

from src.domain.hvac.patterns import (
    extract_dn_from_text,
    extract_pn_from_text,
    extract_voltage_from_text,
    normalize_text,
)
from src.domain.hvac.constants import (
    VALVE_TYPES,
    VALVE_SYNONYMS,
    MATERIALS,
    MATERIAL_SYNONYMS,
    DRIVE_TYPES,
    DRIVE_SYNONYMS,
    MANUFACTURERS,
)
from src.domain.hvac.value_objects.extracted_parameters import ExtractedParameters


class ConcreteParameterExtractor:
    """Extracts DN/PN (regex), valve type/material/drive/manufacturer (dict + synonym resolution)."""

    def extract_parameters(self, text: str) -> ExtractedParameters:
        """Normalize text then extract all parameters; returns ExtractedParameters with confidence scores."""
        # 1. Normalize text for consistent matching
        normalized = normalize_text(text)

        # 2. Extract all parameters
        dn, dn_conf = self.extract_dn(normalized)
        pn, pn_conf = self.extract_pn(normalized)
        valve_type, valve_conf = self.extract_valve_type(normalized)
        material, material_conf = self.extract_material(normalized)
        drive_type, drive_conf = self.extract_drive_type(normalized)
        voltage, voltage_conf = self.extract_voltage(normalized)
        manufacturer, manuf_conf = self.extract_manufacturer(normalized)

        # 3. Build confidence scores dict (only for non-None values)
        confidence_scores = {}
        if dn is not None:
            confidence_scores["dn"] = dn_conf
        if pn is not None:
            confidence_scores["pn"] = pn_conf
        if valve_type is not None:
            confidence_scores["valve_type"] = valve_conf
        if material is not None:
            confidence_scores["material"] = material_conf
        if drive_type is not None:
            confidence_scores["drive_type"] = drive_conf
        if voltage is not None:
            confidence_scores["voltage"] = voltage_conf
        if manufacturer is not None:
            confidence_scores["manufacturer"] = manuf_conf

        # 4. Return ExtractedParameters
        return ExtractedParameters(
            dn=dn,
            pn=pn,
            valve_type=valve_type,
            material=material,
            drive_type=drive_type,
            voltage=voltage,
            manufacturer=manufacturer,
            confidence_scores=confidence_scores,
        )

    def extract_dn(self, text: str) -> tuple[Optional[int], float]:
        return extract_dn_from_text(text)

    def extract_pn(self, text: str) -> tuple[Optional[int], float]:
        return extract_pn_from_text(text)

    def extract_valve_type(self, text: str) -> tuple[Optional[str], float]:
        """Longest-match dictionary search; synonyms resolved to canonical form."""
        normalized_text = normalize_text(text)
        sorted_valve_types = sorted(VALVE_TYPES, key=len, reverse=True)

        for valve_type in sorted_valve_types:
            if valve_type in normalized_text:
                if valve_type in VALVE_SYNONYMS:
                    canonical = VALVE_SYNONYMS[valve_type]
                    return canonical, 0.9
                else:
                    return valve_type, 1.0

        return None, 0.0

    def extract_material(self, text: str) -> tuple[Optional[str], float]:
        normalized_text = normalize_text(text)
        sorted_materials = sorted(MATERIALS, key=len, reverse=True)

        for material in sorted_materials:
            if material.lower() in normalized_text:
                if material in MATERIAL_SYNONYMS:
                    canonical = MATERIAL_SYNONYMS[material]
                    return canonical, 0.9
                else:
                    return material, 1.0

        return None, 0.0

    def extract_drive_type(self, text: str) -> tuple[Optional[str], float]:
        normalized_text = normalize_text(text)
        sorted_drive_types = sorted(DRIVE_TYPES, key=len, reverse=True)

        for drive_type in sorted_drive_types:
            if drive_type in normalized_text:
                if drive_type in DRIVE_SYNONYMS:
                    canonical = DRIVE_SYNONYMS[drive_type]
                    return canonical, 0.9
                else:
                    return drive_type, 1.0

        return None, 0.0

    def extract_voltage(self, text: str) -> tuple[Optional[str], float]:
        return extract_voltage_from_text(text)

    def extract_manufacturer(self, text: str) -> tuple[Optional[str], float]:
        """Exact match against MANUFACTURERS dict; returns uppercase name."""
        normalized_text = normalize_text(text)
        sorted_manufacturers = sorted(MANUFACTURERS, key=len, reverse=True)

        for manufacturer in sorted_manufacturers:
            if manufacturer.lower() in normalized_text:
                return manufacturer.upper(), 1.0

        return None, 0.0
