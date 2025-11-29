"""
Tests for ConcreteParameterExtractor Service.
Covers: parameter extraction methods, dictionary matching, confidence scoring, integration scenarios.
"""

import pytest

from src.domain.hvac.services.concrete_parameter_extractor import (
    ConcreteParameterExtractor,
)
from src.domain.hvac.value_objects.extracted_parameters import ExtractedParameters


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def extractor():
    """Fixture for ConcreteParameterExtractor instance."""
    return ConcreteParameterExtractor()


# ============================================================================
# TESTS - extract_dn()
# ============================================================================


def test_extract_dn_standard_format(extractor):
    """Test extract_dn() with standard DN notation."""
    dn, confidence = extractor.extract_dn("zawór dn50 pn16")
    assert dn == 50
    assert confidence == 1.0


def test_extract_dn_diameter_symbol(extractor):
    """Test extract_dn() with Ø symbol."""
    dn, confidence = extractor.extract_dn("zawór ø80")
    assert dn == 80
    assert confidence == 1.0


def test_extract_dn_word_pattern(extractor):
    """Test extract_dn() with word pattern (średnica)."""
    dn, confidence = extractor.extract_dn("rura średnica 100mm")
    assert dn == 100
    assert confidence == 0.8  # Lower confidence for word pattern


def test_extract_dn_not_found(extractor):
    """Test extract_dn() when DN not found."""
    dn, confidence = extractor.extract_dn("zawór bez specyfikacji")
    assert dn is None
    assert confidence == 0.0


# ============================================================================
# TESTS - extract_pn()
# ============================================================================


def test_extract_pn_standard_format(extractor):
    """Test extract_pn() with standard PN notation."""
    pn, confidence = extractor.extract_pn("zawór dn50 pn16")
    assert pn == 16
    assert confidence == 1.0


def test_extract_pn_word_pattern(extractor):
    """Test extract_pn() with word pattern (ciśnienie)."""
    pn, confidence = extractor.extract_pn("ciśnienie 10 bar")
    assert pn == 10
    assert confidence == 0.8  # Lower confidence for word pattern


def test_extract_pn_not_found(extractor):
    """Test extract_pn() when PN not found."""
    pn, confidence = extractor.extract_pn("zawór bez specyfikacji")
    assert pn is None
    assert confidence == 0.0


# ============================================================================
# TESTS - extract_valve_type()
# ============================================================================


def test_extract_valve_type_canonical_match(extractor):
    """Test extract_valve_type() with canonical form."""
    # "kulowy" alone is canonical, should have confidence 1.0
    valve_type, confidence = extractor.extract_valve_type("kulowy dn50")
    assert valve_type == "kulowy"
    assert confidence == 1.0


def test_extract_valve_type_synonym_match(extractor):
    """Test extract_valve_type() with synonym."""
    valve_type, confidence = extractor.extract_valve_type("kurek kulowy dn50")
    assert valve_type == "kulowy"
    assert confidence == 0.9  # Synonym has lower confidence


def test_extract_valve_type_longest_match_first(extractor):
    """Test that longest match is preferred (zawór kulowy over kulowy)."""
    # "zawór kulowy" should match before "kulowy" alone
    valve_type, confidence = extractor.extract_valve_type("zawór kulowy pn16")
    assert valve_type == "kulowy"
    # Both are in VALVE_SYNONYMS, so confidence should be 0.9
    assert confidence == 0.9


def test_extract_valve_type_check_valve(extractor):
    """Test extract_valve_type() with check valve."""
    valve_type, confidence = extractor.extract_valve_type("zawór zwrotny dn25")
    assert valve_type == "zwrotny"
    assert confidence == 0.9


def test_extract_valve_type_butterfly_valve(extractor):
    """Test extract_valve_type() with butterfly valve."""
    valve_type, confidence = extractor.extract_valve_type("zawór motylkowy dn100")
    assert valve_type == "motylkowy"
    assert confidence == 0.9


def test_extract_valve_type_not_found(extractor):
    """Test extract_valve_type() when valve type not found."""
    valve_type, confidence = extractor.extract_valve_type("rura bez zaworu")
    assert valve_type is None
    assert confidence == 0.0


# ============================================================================
# TESTS - extract_material()
# ============================================================================


def test_extract_material_canonical_match(extractor):
    """Test extract_material() with canonical form."""
    material, confidence = extractor.extract_material("zawór mosiądz dn50")
    assert material == "mosiądz"
    assert confidence == 1.0


def test_extract_material_synonym_match(extractor):
    """Test extract_material() with synonym (mosiężny -> mosiądz)."""
    material, confidence = extractor.extract_material("zawór mosiężny dn50")
    assert material == "mosiądz"
    assert confidence == 0.9


def test_extract_material_stainless_steel(extractor):
    """Test extract_material() with stainless steel."""
    material, confidence = extractor.extract_material("zawór stal nierdzewna dn50")
    assert material == "stal nierdzewna"
    assert confidence == 1.0


def test_extract_material_synonym_nierdzewny(extractor):
    """Test extract_material() with synonym (nierdzewny -> stal nierdzewna)."""
    material, confidence = extractor.extract_material("zawór nierdzewny dn50")
    assert material == "stal nierdzewna"
    assert confidence == 0.9


def test_extract_material_plastic(extractor):
    """Test extract_material() with plastic material (PP-R)."""
    material, confidence = extractor.extract_material("rura pp-r dn25")
    assert material == "PP-R"
    assert confidence == 1.0


def test_extract_material_plastic_synonym(extractor):
    """Test extract_material() with plastic synonym (polipropylen -> PP-R)."""
    material, confidence = extractor.extract_material("rura polipropylen dn25")
    assert material == "PP-R"
    assert confidence == 0.9


def test_extract_material_not_found(extractor):
    """Test extract_material() when material not found."""
    material, confidence = extractor.extract_material("zawór bez materiału")
    assert material is None
    assert confidence == 0.0


# ============================================================================
# TESTS - extract_drive_type()
# ============================================================================


def test_extract_drive_type_canonical_manual(extractor):
    """Test extract_drive_type() with manual drive."""
    drive_type, confidence = extractor.extract_drive_type("zawór ręczny dn50")
    assert drive_type == "ręczny"
    assert confidence == 1.0


def test_extract_drive_type_canonical_electric(extractor):
    """Test extract_drive_type() with electric drive canonical."""
    drive_type, confidence = extractor.extract_drive_type("zawór elektryczny dn50")
    assert drive_type == "elektryczny"
    assert confidence == 1.0


def test_extract_drive_type_synonym_electric(extractor):
    """Test extract_drive_type() with electric drive synonym."""
    drive_type, confidence = extractor.extract_drive_type("siłownik elektryczny 230v")
    assert drive_type == "elektryczny"
    assert confidence == 0.9


def test_extract_drive_type_synonym_servomotor(extractor):
    """Test extract_drive_type() with servomotor synonym."""
    drive_type, confidence = extractor.extract_drive_type("zawór serwomotor 24v")
    assert drive_type == "elektryczny"
    assert confidence == 0.9


def test_extract_drive_type_pneumatic(extractor):
    """Test extract_drive_type() with pneumatic drive."""
    drive_type, confidence = extractor.extract_drive_type("zawór pneumatyczny dn50")
    assert drive_type == "pneumatyczny"
    assert confidence == 1.0


def test_extract_drive_type_synonym_pneumatic(extractor):
    """Test extract_drive_type() with pneumatic synonym."""
    drive_type, confidence = extractor.extract_drive_type("napęd pneumatyczny dn50")
    assert drive_type == "pneumatyczny"
    assert confidence == 0.9


def test_extract_drive_type_not_found(extractor):
    """Test extract_drive_type() when drive type not found."""
    drive_type, confidence = extractor.extract_drive_type("zawór bez napędu")
    assert drive_type is None
    assert confidence == 0.0


# ============================================================================
# TESTS - extract_voltage()
# ============================================================================


def test_extract_voltage_230v(extractor):
    """Test extract_voltage() with 230V."""
    voltage, confidence = extractor.extract_voltage("napęd elektryczny 230v")
    assert voltage == "230V"
    assert confidence == 1.0


def test_extract_voltage_24v(extractor):
    """Test extract_voltage() with 24V."""
    voltage, confidence = extractor.extract_voltage("napęd 24v")
    assert voltage == "24V"
    assert confidence == 1.0


def test_extract_voltage_400v(extractor):
    """Test extract_voltage() with 400V."""
    voltage, confidence = extractor.extract_voltage("zasilanie 400 v")
    assert voltage == "400V"
    assert confidence == 1.0


def test_extract_voltage_not_found(extractor):
    """Test extract_voltage() when voltage not found."""
    voltage, confidence = extractor.extract_voltage("zawór ręczny")
    assert voltage is None
    assert confidence == 0.0


# ============================================================================
# TESTS - extract_manufacturer()
# ============================================================================


def test_extract_manufacturer_ksb(extractor):
    """Test extract_manufacturer() with KSB."""
    manufacturer, confidence = extractor.extract_manufacturer("zawór ksb dn50")
    assert manufacturer == "KSB"
    assert confidence == 1.0


def test_extract_manufacturer_danfoss(extractor):
    """Test extract_manufacturer() with Danfoss."""
    manufacturer, confidence = extractor.extract_manufacturer("zawór danfoss dn50")
    assert manufacturer == "DANFOSS"
    assert confidence == 1.0


def test_extract_manufacturer_case_insensitive(extractor):
    """Test extract_manufacturer() is case-insensitive."""
    manufacturer, confidence = extractor.extract_manufacturer("zawór GRUNDFOS dn50")
    assert manufacturer == "GRUNDFOS"
    assert confidence == 1.0


def test_extract_manufacturer_belimo(extractor):
    """Test extract_manufacturer() with Belimo."""
    manufacturer, confidence = extractor.extract_manufacturer("napęd belimo 230v")
    assert manufacturer == "BELIMO"
    assert confidence == 1.0


def test_extract_manufacturer_not_found(extractor):
    """Test extract_manufacturer() when manufacturer not found."""
    manufacturer, confidence = extractor.extract_manufacturer("zawór bez producenta")
    assert manufacturer is None
    assert confidence == 0.0


# ============================================================================
# TESTS - extract_parameters() - Main Orchestration
# ============================================================================


def test_extract_parameters_full_description(extractor):
    """Test extract_parameters() with complete HVAC description."""
    text = "Zawór kulowy DN50 PN16 mosiądz napęd elektryczny 230V KSB"
    params = extractor.extract_parameters(text)

    assert isinstance(params, ExtractedParameters)
    assert params.dn == 50
    assert params.pn == 16
    assert params.valve_type == "kulowy"
    assert params.material == "mosiądz"
    assert params.drive_type == "elektryczny"
    assert params.voltage == "230V"
    assert params.manufacturer == "KSB"

    # Check confidence scores
    assert params.confidence_scores["dn"] == 1.0
    assert params.confidence_scores["pn"] == 1.0
    assert params.confidence_scores["valve_type"] == 0.9  # "zawór kulowy" is synonym
    assert params.confidence_scores["material"] == 1.0
    assert params.confidence_scores["drive_type"] == 0.9  # "napęd elektryczny" is synonym
    assert params.confidence_scores["voltage"] == 1.0
    assert params.confidence_scores["manufacturer"] == 1.0


def test_extract_parameters_partial_description(extractor):
    """Test extract_parameters() with partial description (only DN and PN)."""
    text = "Zawór DN50 PN16"
    params = extractor.extract_parameters(text)

    assert params.dn == 50
    assert params.pn == 16
    assert params.valve_type is None
    assert params.material is None
    assert params.drive_type is None
    assert params.voltage is None
    assert params.manufacturer is None

    # Only DN and PN should have confidence scores
    assert "dn" in params.confidence_scores
    assert "pn" in params.confidence_scores
    assert "valve_type" not in params.confidence_scores
    assert "material" not in params.confidence_scores


def test_extract_parameters_empty_string(extractor):
    """Test extract_parameters() with empty string."""
    params = extractor.extract_parameters("")

    assert params.dn is None
    assert params.pn is None
    assert params.valve_type is None
    assert params.material is None
    assert params.drive_type is None
    assert params.voltage is None
    assert params.manufacturer is None
    assert params.confidence_scores == {}


def test_extract_parameters_no_matches(extractor):
    """Test extract_parameters() with text containing no HVAC parameters."""
    text = "To jest jakiś losowy tekst bez parametrów HVAC"
    params = extractor.extract_parameters(text)

    assert params.dn is None
    assert params.pn is None
    assert params.valve_type is None
    assert params.material is None
    assert params.drive_type is None
    assert params.voltage is None
    assert params.manufacturer is None
    assert params.confidence_scores == {}


def test_extract_parameters_normalizes_text(extractor):
    """Test that extract_parameters() normalizes text before extraction."""
    text = "  ZAWÓR  KULOWY  DN50  PN16  "
    params = extractor.extract_parameters(text)

    assert params.dn == 50
    assert params.pn == 16
    assert params.valve_type == "kulowy"


def test_extract_parameters_confidence_scores_only_for_found_values(extractor):
    """Test that confidence_scores dict only contains entries for found values."""
    text = "Zawór DN50"
    params = extractor.extract_parameters(text)

    # Only DN was found
    assert len(params.confidence_scores) == 1
    assert "dn" in params.confidence_scores
    assert params.confidence_scores["dn"] == 1.0


# ============================================================================
# INTEGRATION TESTS - Real-world scenarios
# ============================================================================


def test_integration_ball_valve_with_electric_drive(extractor):
    """Integration test: Ball valve with electric drive and voltage."""
    text = "Zawór kulowy DN50 PN16 mosiężny z napędem elektrycznym 230V KSB"
    params = extractor.extract_parameters(text)

    assert params.dn == 50
    assert params.pn == 16
    assert params.valve_type == "kulowy"
    assert params.material == "mosiądz"
    assert params.drive_type == "elektryczny"
    assert params.voltage == "230V"
    assert params.manufacturer == "KSB"

    # Average confidence should be high
    avg_confidence = params.get_average_confidence()
    assert avg_confidence >= 0.9


def test_integration_check_valve_manual(extractor):
    """Integration test: Check valve, manual operation, stainless steel."""
    text = "Zawór zwrotny DN25 PN10 stal nierdzewna ręczny Danfoss"
    params = extractor.extract_parameters(text)

    assert params.dn == 25
    assert params.pn == 10
    assert params.valve_type == "zwrotny"
    assert params.material == "stal nierdzewna"
    assert params.drive_type == "ręczny"
    assert params.voltage is None  # Manual valve has no voltage
    assert params.manufacturer == "DANFOSS"


def test_integration_butterfly_valve_pneumatic(extractor):
    """Integration test: Butterfly valve with pneumatic drive."""
    text = "Zawór motylkowy DN100 PN16 siłownik pneumatyczny Belimo"
    params = extractor.extract_parameters(text)

    assert params.dn == 100
    assert params.pn == 16
    assert params.valve_type == "motylkowy"
    assert params.material is None
    assert params.drive_type == "pneumatyczny"
    assert params.voltage is None  # Pneumatic drive has no voltage
    assert params.manufacturer == "BELIMO"


def test_integration_plastic_pipe(extractor):
    """Integration test: Plastic pipe description."""
    text = "Rura PP-R średnica 50mm ciśnienie 10 bar"
    params = extractor.extract_parameters(text)

    assert params.dn == 50
    assert params.confidence_scores["dn"] == 0.8  # Word pattern
    assert params.pn == 10
    assert params.confidence_scores["pn"] == 0.8  # Word pattern
    assert params.valve_type is None
    assert params.material == "PP-R"
    assert params.drive_type is None
    assert params.voltage is None
    assert params.manufacturer is None


def test_integration_mixed_synonyms(extractor):
    """Integration test: Description with multiple synonyms."""
    text = "Kurek kulowy Ø80 nierdzewny serwomotor 24V Grundfos"
    params = extractor.extract_parameters(text)

    assert params.dn == 80
    assert params.pn is None
    assert params.valve_type == "kulowy"
    assert params.confidence_scores["valve_type"] == 0.9  # "kurek kulowy" is synonym
    assert params.material == "stal nierdzewna"
    assert params.confidence_scores["material"] == 0.9  # "nierdzewny" is synonym
    assert params.drive_type == "elektryczny"
    assert params.confidence_scores["drive_type"] == 0.9  # "serwomotor" is synonym
    assert params.voltage == "24V"
    assert params.manufacturer == "GRUNDFOS"


def test_integration_word_patterns_only(extractor):
    """Integration test: Description using only word patterns (no standard notation)."""
    text = "Rura średnica 100 ciśnienie 16 bar polipropylen"
    params = extractor.extract_parameters(text)

    assert params.dn == 100
    assert params.confidence_scores["dn"] == 0.8  # Word pattern
    assert params.pn == 16
    assert params.confidence_scores["pn"] == 0.8  # Word pattern
    assert params.valve_type is None
    assert params.material == "PP-R"
    assert params.confidence_scores["material"] == 0.9  # "polipropylen" is synonym


def test_integration_minimal_description(extractor):
    """Integration test: Minimal description with just DN."""
    text = "DN50"
    params = extractor.extract_parameters(text)

    assert params.dn == 50
    assert params.pn is None
    assert params.valve_type is None
    assert params.material is None
    assert params.drive_type is None
    assert params.voltage is None
    assert params.manufacturer is None
    assert len(params.confidence_scores) == 1


def test_integration_polish_specific_terms(extractor):
    """Integration test: Polish-specific terminology."""
    text = "Zasuwa DN200 PN25 żeliwo dźwignia FERRO"
    params = extractor.extract_parameters(text)

    assert params.dn == 200
    assert params.pn == 25
    assert params.valve_type == "zasuwowy"
    assert params.confidence_scores["valve_type"] == 0.9  # "zasuwa" is synonym
    assert params.material == "żeliwo"
    assert params.drive_type == "ręczny"
    assert params.confidence_scores["drive_type"] == 0.9  # "dźwignia" is synonym
    assert params.voltage is None
    assert params.manufacturer == "FERRO"


# ============================================================================
# EDGE CASES
# ============================================================================


def test_edge_case_multiple_manufacturers(extractor):
    """Edge case: Text contains multiple manufacturers (should match first/longest)."""
    text = "Zawór DN50 KSB Danfoss"
    params = extractor.extract_parameters(text)

    # Should match one of them (sorted by length descending, so likely first found)
    assert params.manufacturer in ["KSB", "DANFOSS"]


def test_edge_case_multiple_dn_values(extractor):
    """Edge case: Text contains multiple DN values (should match first)."""
    text = "Zawór DN50 DN100"
    params = extractor.extract_parameters(text)

    # Should match first occurrence
    assert params.dn == 50


def test_edge_case_whitespace_variations(extractor):
    """Edge case: Various whitespace patterns."""
    text = "Zawór   kulowy    DN50     PN16"
    params = extractor.extract_parameters(text)

    assert params.dn == 50
    assert params.pn == 16
    assert params.valve_type == "kulowy"
