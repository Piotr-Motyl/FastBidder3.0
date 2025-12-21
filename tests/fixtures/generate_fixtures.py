"""
Generate Sample Excel Files for E2E Testing

This script generates sample Excel files for end-to-end testing:
1. sample_working_file.xlsx - 20 HVAC descriptions without prices
2. sample_reference_file.xlsx - 50 catalog descriptions with prices
3. performance_working_file.xlsx - 100 HVAC descriptions (for performance tests)
4. performance_reference_file.xlsx - 200 catalog descriptions (for performance tests)

The files contain realistic HVAC equipment descriptions in Polish
with DN (Diameter Nominal) and PN (Pressure Nominal) parameters.

Usage:
    python tests/fixtures/generate_fixtures.py
    python tests/fixtures/generate_fixtures.py --performance  # Generate performance fixtures

Output:
    - tests/fixtures/sample_working_file.xlsx
    - tests/fixtures/sample_reference_file.xlsx
    - tests/fixtures/performance_working_file.xlsx (optional)
    - tests/fixtures/performance_reference_file.xlsx (optional)
"""

import argparse
import pandas as pd
from pathlib import Path


def generate_working_file():
    """
    Generate sample working file with 20 HVAC descriptions.

    Working file contains:
        - Description: HVAC equipment description (Polish)
        - No prices (to be matched from reference file)
        - Various equipment types: valves, taps, flaps, pipes, elbows
        - Various DN sizes: 15-150mm
        - Various PN ratings: 6-40 bar
        - Various materials: brass, steel, cast iron, PVC

    Returns:
        pd.DataFrame with working file data
    """
    # Sample HVAC descriptions for working file (20 items)
    working_descriptions = [
        "Zawór kulowy DN50 PN16 mosiądz",
        "Kurek odcinający DN80 PN10 stal",
        "Klapka zwrotna DN100 z napędem 230V",
        "Zawór kulowy DN25 PN16 mosiądz",
        "Rura stalowa DN100 PN10 L=6m",
        "Kolano 90° DN50 PN16 stal",
        "Zawór zwrotny DN65 PN10 mosiądz klapowy",
        "Kurek kulowy DN32 PN25 mosiądz z dźwignią",
        "Zawór odcinający DN150 PN16 żeliwo z kołnierzem",
        "Klapka przeciwpożarowa DN125 PN10 230V siłownik",
        "Zawór kulowy DN20 PN16 mosiądz gwint wewnętrzny",
        "Rura ocynkowana DN80 PN6 L=3m",
        "Kolano 45° DN100 PN10 stal spawane",
        "Zawór motylkowy DN150 PN16 z przekładnią",
        "Kurek odcinający DN40 PN25 mosiądz z nakrętką",
        "Klapka zwrotna DN80 PN10 mosiądz międzykołnierzowa",
        "Zawór kulowy DN15 PN40 mosiądz 3-drogowy",
        "Rura PVC DN110 PN6 klej L=6m",
        "Kolano 90° DN32 PN16 mosiądz gwintowane",
        "Zawór grzejnikowy DN20 PN10 mosiądz termostatyczny",
    ]

    # Create DataFrame
    df = pd.DataFrame({
        "Description": working_descriptions
    })

    return df


def generate_reference_file():
    """
    Generate sample reference file with 50 catalog descriptions and prices.

    Reference file contains:
        - Description: HVAC equipment description (Polish)
        - Price: Price in PLN (50-500 PLN range)
        - Similar items to working file (for successful matching)
        - Additional catalog items (for realistic catalog)
        - Various equipment types and sizes

    Returns:
        pd.DataFrame with reference file data
    """
    # Sample catalog descriptions with prices (50 items)
    # Items 1-25: Similar to working file (should match)
    reference_data = [
        # Exact or close matches to working file
        {"Description": "Zawór kulowy DN50 PN16 mosiądz", "Price": 125.00},
        {"Description": "Kurek odcinający DN80 PN10 stal", "Price": 245.00},
        {"Description": "Klapka zwrotna DN100 z napędem elektrycznym 230V", "Price": 1250.00},
        {"Description": "Zawór kulowy DN25 PN16 mosiądz gwintowany", "Price": 78.50},
        {"Description": "Rura stalowa czarna DN100 PN10 L=6m", "Price": 380.00},
        {"Description": "Kolano stalowe 90° DN50 PN16", "Price": 45.00},
        {"Description": "Zawór zwrotny klapowy DN65 PN10 mosiądz", "Price": 165.00},
        {"Description": "Kurek kulowy DN32 PN25 mosiądz z dźwignią motylkową", "Price": 95.00},
        {"Description": "Zawór odcinający kołnierzowy DN150 PN16 żeliwo", "Price": 890.00},
        {"Description": "Klapka przeciwpożarowa DN125 PN10 z siłownikiem 230V", "Price": 2100.00},
        {"Description": "Zawór kulowy DN20 PN16 mosiądz GW-GW", "Price": 42.00},
        {"Description": "Rura stalowa ocynkowana DN80 PN6 L=3m", "Price": 145.00},
        {"Description": "Kolano spawane 45° DN100 PN10 stal", "Price": 67.00},
        {"Description": "Zawór motylkowy DN150 PN16 z przekładnią ślimakową", "Price": 1450.00},
        {"Description": "Kurek odcinający DN40 PN25 mosiądz z nakrętką", "Price": 112.00},
        {"Description": "Klapka zwrotna międzykołnierzowa DN80 PN10 mosiądz", "Price": 185.00},
        {"Description": "Zawór kulowy 3-drogowy DN15 PN40 mosiądz", "Price": 156.00},
        {"Description": "Rura PVC-U DN110 PN6 klejona L=6m", "Price": 89.00},
        {"Description": "Kolano gwintowane 90° DN32 PN16 mosiądz", "Price": 34.50},
        {"Description": "Zawór grzejnikowy termostatyczny DN20 PN10 mosiądz", "Price": 78.00},
        # Additional catalog items (no exact match in working file)
        {"Description": "Zawór kulowy DN65 PN25 mosiądz kołnierzowy", "Price": 235.00},
        {"Description": "Kurek odcinający DN50 PN16 stal nierdzewna", "Price": 198.00},
        {"Description": "Klapka zwrotna DN50 PN16 mosiądz sprężynowa", "Price": 95.00},
        {"Description": "Zawór motylkowy DN200 PN10 żeliwo z napędem ręcznym", "Price": 1850.00},
        {"Description": "Rura stalowa przewodowa DN125 PN16 L=6m", "Price": 520.00},
        {"Description": "Kolano segmentowe 90° DN150 PN10 spawane", "Price": 145.00},
        {"Description": "Zawór zwrotny talerzowy DN40 PN25 mosiądz", "Price": 125.00},
        {"Description": "Kurek kulowy DN20 PN40 mosiądz z motylkiem", "Price": 58.00},
        {"Description": "Klapka regulacyjna DN100 PN10 z siłownikiem 24V", "Price": 890.00},
        {"Description": "Zawór grzejnikowy DN15 PN10 mosiądz prosty", "Price": 45.00},
        {"Description": "Rura ocynkowana DN65 PN10 L=3m", "Price": 98.00},
        {"Description": "Kolano redukcyjne 90° DN100/80 PN10 stal", "Price": 78.00},
        {"Description": "Zawór kulowy DN40 PN16 mosiądz pełnoprzepływowy", "Price": 108.00},
        {"Description": "Kurek spustowy DN15 PN16 mosiądz z kluczem", "Price": 28.00},
        {"Description": "Klapka przeciwpożarowa DN80 PN6 z bezpiecznikiem topikowym", "Price": 560.00},
        {"Description": "Zawór odcinający DN100 PN10 żeliwo kołnierzowy", "Price": 485.00},
        {"Description": "Rura PE DN110 PN6 spawana L=6m", "Price": 125.00},
        {"Description": "Kolano gwintowane 45° DN40 PN16 mosiądz", "Price": 42.00},
        {"Description": "Zawór motylkowy DN125 PN16 z dźwignią", "Price": 890.00},
        {"Description": "Kurek kulowy DN25 PN25 mosiądz z gwintem wewnętrznym", "Price": 85.00},
        {"Description": "Klapka zwrotna DN32 PN16 mosiądz kołnierzowa", "Price": 78.00},
        {"Description": "Zawór zwrotny DN50 PN10 mosiądz międzykołnierzowy", "Price": 145.00},
        {"Description": "Rura stalowa czarna DN150 PN10 L=6m spawana", "Price": 680.00},
        {"Description": "Kolano spawane 90° DN125 PN16 stal", "Price": 128.00},
        {"Description": "Zawór kulowy DN80 PN16 mosiądz z dźwignią", "Price": 298.00},
        {"Description": "Kurek odcinający DN65 PN10 żeliwo kołnierzowy", "Price": 325.00},
        {"Description": "Klapka regulacyjna DN150 PN10 ręczna", "Price": 450.00},
        {"Description": "Zawór grzejnikowy DN20 PN16 mosiądz kątowy", "Price": 68.00},
        {"Description": "Rura ocynkowana DN40 PN16 L=3m gwintowana", "Price": 56.00},
        {"Description": "Zawór motylkowy DN100 PN16 z przekładnią zębatą", "Price": 950.00},
    ]

    # Create DataFrame
    df = pd.DataFrame(reference_data)

    return df


def generate_performance_working_file():
    """
    Generate performance test working file with 100 HVAC descriptions.

    Performance working file contains:
        - Description: HVAC equipment description (Polish)
        - 100 items (max limit for Phase 3)
        - Generated by repeating base patterns with variations

    Returns:
        pd.DataFrame with working file data
    """
    # Base description templates
    valve_types = ["kulowy", "odcinający", "zwrotny", "motylkowy", "grzejnikowy"]
    dn_sizes = [15, 20, 25, 32, 40, 50, 65, 80, 100, 125, 150]
    pn_ratings = [6, 10, 16, 25, 40]
    materials = ["mosiądz", "stal", "żeliwo", "PVC"]

    descriptions = []

    # Generate 100 descriptions by combining parameters
    for i in range(100):
        valve_type = valve_types[i % len(valve_types)]
        dn = dn_sizes[i % len(dn_sizes)]
        pn = pn_ratings[i % len(pn_ratings)]
        material = materials[i % len(materials)]

        # Variation in description format
        if i % 3 == 0:
            desc = f"Zawór {valve_type} DN{dn} PN{pn} {material}"
        elif i % 3 == 1:
            desc = f"Kurek {valve_type} DN{dn} PN{pn} {material} gwintowany"
        else:
            desc = f"Zawór {valve_type} DN{dn} PN{pn} {material} kołnierzowy"

        descriptions.append(desc)

    df = pd.DataFrame({"Description": descriptions})
    return df


def generate_performance_reference_file():
    """
    Generate performance test reference file with 200 catalog descriptions and prices.

    Performance reference file contains:
        - Description: HVAC equipment description (Polish)
        - Price: Price in PLN (50-2000 PLN range)
        - 200 items (2x working file for realistic catalog)
        - Generated by repeating base patterns with variations

    Returns:
        pd.DataFrame with reference file data
    """
    # Base description templates
    valve_types = ["kulowy", "odcinający", "zwrotny", "motylkowy", "grzejnikowy", "klapowy"]
    dn_sizes = [15, 20, 25, 32, 40, 50, 65, 80, 100, 125, 150, 200]
    pn_ratings = [6, 10, 16, 25, 40, 63]
    materials = ["mosiądz", "stal", "stal nierdzewna", "żeliwo", "PVC", "PP-R"]

    reference_data = []

    # Generate 200 descriptions by combining parameters
    for i in range(200):
        valve_type = valve_types[i % len(valve_types)]
        dn = dn_sizes[i % len(dn_sizes)]
        pn = pn_ratings[i % len(pn_ratings)]
        material = materials[i % len(materials)]

        # Variation in description format
        if i % 4 == 0:
            desc = f"Zawór {valve_type} DN{dn} PN{pn} {material}"
        elif i % 4 == 1:
            desc = f"Kurek {valve_type} DN{dn} PN{pn} {material} gwintowany"
        elif i % 4 == 2:
            desc = f"Zawór {valve_type} DN{dn} PN{pn} {material} kołnierzowy"
        else:
            desc = f"Zawór {valve_type} DN{dn} PN{pn} {material} z napędem"

        # Price calculation: base price depends on DN and PN
        base_price = 50 + (dn * 2) + (pn * 5)
        # Add material modifier
        if "nierdzewna" in material:
            base_price *= 1.5
        elif material == "żeliwo":
            base_price *= 1.2

        reference_data.append({"Description": desc, "Price": round(base_price, 2)})

    df = pd.DataFrame(reference_data)
    return df


def generate_polish_chars_working_file():
    """
    Generate working file with Polish characters for E2E testing.

    Contains 3 descriptions with Polish diacritics: ą, ć, ę, ł, ń, ó, ś, ź, ż
    Tests encoding preservation through entire stack.

    Returns:
        pd.DataFrame with Polish character descriptions
    """
    descriptions = [
        "Zawór kulowy DN50 PN16 mosiądz",  # ą
        "Zawór zwrotny DN80 PN10 żeliwo szare",  # ż
        "Kompensator długości DN100 stal nierdzewna",  # ł, ó
    ]

    df = pd.DataFrame({"Description": descriptions})
    return df


def generate_polish_chars_reference_file():
    """
    Generate reference file matching Polish character working file.

    Contains matching items with prices for Polish character test.

    Returns:
        pd.DataFrame with matching items
    """
    reference_data = [
        {"Description": "Zawór kulowy DN50 PN16 mosiądz", "Price": 250.00},
        {"Description": "Zawór zwrotny DN80 PN10 żeliwo", "Price": 180.00},
        {"Description": "Kompensator DN100 nierdzewny", "Price": 450.00},
    ]

    df = pd.DataFrame(reference_data)
    return df


def generate_single_item_working_file():
    """
    Generate working file with single item for boundary testing.

    Contains only 1 description to test minimum viable input.
    Tests off-by-one errors and edge case handling.

    Returns:
        pd.DataFrame with single description
    """
    descriptions = [
        "Zawór kulowy DN50 PN16 mosiądz",
    ]

    df = pd.DataFrame({"Description": descriptions})
    return df


def main():
    """
    Main function to generate Excel fixtures.

    Creates:
        - tests/fixtures/sample_working_file.xlsx (20 rows)
        - tests/fixtures/sample_reference_file.xlsx (50 rows)
        - tests/fixtures/performance_working_file.xlsx (100 rows, optional)
        - tests/fixtures/performance_reference_file.xlsx (200 rows, optional)
        - tests/fixtures/polish_chars_working.xlsx (3 rows, with --data-variations)
        - tests/fixtures/polish_chars_reference.xlsx (3 rows, with --data-variations)
        - tests/fixtures/single_item_working.xlsx (1 row, with --data-variations)
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate sample Excel fixtures for testing")
    parser.add_argument(
        "--performance",
        action="store_true",
        help="Generate performance test fixtures (100/200 items)",
    )
    parser.add_argument(
        "--data-variations",
        action="store_true",
        help="Generate data variation fixtures (Polish chars, single item)",
    )
    args = parser.parse_args()

    # Get fixtures directory path
    fixtures_dir = Path(__file__).parent

    # Generate standard fixtures (always)
    print("Generating sample_working_file.xlsx...")
    working_df = generate_working_file()
    working_file_path = fixtures_dir / "sample_working_file.xlsx"
    working_df.to_excel(working_file_path, index=False, engine="openpyxl")
    print(f"[OK] Created {working_file_path}")
    print(f"  Rows: {len(working_df)}")
    print(f"  Columns: {list(working_df.columns)}")

    print("\nGenerating sample_reference_file.xlsx...")
    reference_df = generate_reference_file()
    reference_file_path = fixtures_dir / "sample_reference_file.xlsx"
    reference_df.to_excel(reference_file_path, index=False, engine="openpyxl")
    print(f"[OK] Created {reference_file_path}")
    print(f"  Rows: {len(reference_df)}")
    print(f"  Columns: {list(reference_df.columns)}")

    # Generate performance fixtures if requested
    if args.performance:
        print("\n" + "=" * 60)
        print("GENERATING PERFORMANCE FIXTURES")
        print("=" * 60)

        print("\nGenerating performance_working_file.xlsx...")
        perf_working_df = generate_performance_working_file()
        perf_working_path = fixtures_dir / "performance_working_file.xlsx"
        perf_working_df.to_excel(perf_working_path, index=False, engine="openpyxl")
        print(f"[OK] Created {perf_working_path}")
        print(f"  Rows: {len(perf_working_df)}")
        print(f"  Columns: {list(perf_working_df.columns)}")

        print("\nGenerating performance_reference_file.xlsx...")
        perf_reference_df = generate_performance_reference_file()
        perf_reference_path = fixtures_dir / "performance_reference_file.xlsx"
        perf_reference_df.to_excel(perf_reference_path, index=False, engine="openpyxl")
        print(f"[OK] Created {perf_reference_path}")
        print(f"  Rows: {len(perf_reference_df)}")
        print(f"  Columns: {list(perf_reference_df.columns)}")

    # Generate data variation fixtures if requested
    if args.data_variations:
        print("\n" + "=" * 60)
        print("GENERATING DATA VARIATION FIXTURES")
        print("=" * 60)

        print("\nGenerating polish_chars_working.xlsx...")
        polish_working_df = generate_polish_chars_working_file()
        polish_working_path = fixtures_dir / "polish_chars_working.xlsx"
        polish_working_df.to_excel(polish_working_path, index=False, engine="openpyxl")
        print(f"[OK] Created {polish_working_path}")
        print(f"  Rows: {len(polish_working_df)}")
        print(f"  Columns: {list(polish_working_df.columns)}")

        print("\nGenerating polish_chars_reference.xlsx...")
        polish_reference_df = generate_polish_chars_reference_file()
        polish_reference_path = fixtures_dir / "polish_chars_reference.xlsx"
        polish_reference_df.to_excel(polish_reference_path, index=False, engine="openpyxl")
        print(f"[OK] Created {polish_reference_path}")
        print(f"  Rows: {len(polish_reference_df)}")
        print(f"  Columns: {list(polish_reference_df.columns)}")

        print("\nGenerating single_item_working.xlsx...")
        single_item_df = generate_single_item_working_file()
        single_item_path = fixtures_dir / "single_item_working.xlsx"
        single_item_df.to_excel(single_item_path, index=False, engine="openpyxl")
        print(f"[OK] Created {single_item_path}")
        print(f"  Rows: {len(single_item_df)}")
        print(f"  Columns: {list(single_item_df.columns)}")

    # Preview
    print("\n" + "=" * 60)
    print("WORKING FILE PREVIEW (first 5 rows):")
    print("=" * 60)
    print(working_df.head())

    print("\n" + "=" * 60)
    print("REFERENCE FILE PREVIEW (first 5 rows):")
    print("=" * 60)
    print(reference_df.head())

    print("\n[OK] Fixture generation completed successfully!")
    print("\nFiles created:")
    print(f"  - {working_file_path}")
    print(f"  - {reference_file_path}")
    if args.performance:
        print(f"  - {perf_working_path}")
        print(f"  - {perf_reference_path}")
    if args.data_variations:
        print(f"  - {polish_working_path}")
        print(f"  - {polish_reference_path}")
        print(f"  - {single_item_path}")


if __name__ == "__main__":
    main()
