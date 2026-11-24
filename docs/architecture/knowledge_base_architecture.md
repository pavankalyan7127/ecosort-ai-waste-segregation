# EcoSort AI — Waste Taxonomy & Knowledge Base Architecture

**Project**: EcoSort AI (Bharat AI-SoC Challenge 2026–27)  
**Problem Statement 2**: Smart Recycling and Waste Segregation  
**Document Classification**: Technical Architecture & Regulatory Conformance Specification  

---

## 1. Executive Summary & Regulatory Grounding

The **EcoSort AI Knowledge Base** serves as the authoritative, deterministic ground-truth engine powering the kiosk's multimodal classification, user guidance, and conversational assistant. 

### Strict Anti-Hallucination Directive
Under the challenge directives, **no synthetic, fictional, guessed, or AI-invented waste-management facts are permitted**. Every classification, preparation instruction, handling precaution, and disposal directive is strictly traceable to authentic national statutory legislation and state notifications:

1. **Primary National Benchmark**:
   - **Solid Waste Management Rules, 2026** (Notification S.O. 388(E), dated 27th January, 2026; effective 1st April, 2026, superseding SWM Rules 2016).
2. **Specialized Extended Producer & Hazardous Regimes**:
   - **E-Waste (Management) Rules, 2022** (Notification G.S.R. 814(E))
   - **Battery Waste Management Rules, 2022** (Notification G.S.R. 664(E))
   - **Plastic Waste Management Rules, 2016** (as amended through 2024, including Single-Use Plastic ban)
   - **Bio-Medical Waste Management Rules, 2016** (Notification G.S.R. 343(E))
3. **State & Local Enforcement (Tamil Nadu)**:
   - **Government of Tamil Nadu G.O. (Ms.) No. 84** (Environment and Forests Department, dated 25.06.2018; state-wide ban on 14 throwaway plastic items)
   - **Tamil Nadu Urban Local Bodies Rules, 2023** (Notification G.O. (Ms.) No. 5, Municipal Administration & Water Supply Department)
   - **Tamil Nadu Pollution Control Board (TNPCB)** Solid Waste Management & Source Segregation Guidelines

---

## 2. Four-Stream Statutory Taxonomy

Unlike the older 2016 framework (which recognized only three broad streams: wet, dry, and domestic hazardous), the **Solid Waste Management Rules, 2026** establish a mandatory **four-stream segregation protocol at source** (Rule 5(1)(b), Rule 39(4), Rule 39(51)):

```
                                  SOLID WASTE GENERATION AT SOURCE
                                       (Rule 5(1)(b) & Rule 39)
                                                  │
         ┌─────────────────────────┬──────────────┴──────────────┬─────────────────────────┐
         ▼                         ▼                             ▼                         ▼
   1. WET WASTE              2. DRY WASTE                3. SANITARY WASTE         4. SPECIAL CARE WASTE
   (Green Bin)               (Blue Bin)                  (Dedicated Pouch/Bin)     (Deposition Centre / E-Waste)
   Rule 3(zzl)               Rule 3(s), 3(l), 3(z)       Rule 3(zp), Rule 5(1)(c)  Rule 3(zx), E-Waste, Battery
   ───────────               ─────────────────────       ────────────────────────  ────────────────────────────
   • Cooked food scraps      • Paper & cardboard         • Used baby/adult diapers • CFLs & fluorescent tubes
   • Fruit & veg peels       • PET bottles (Resin 1)     • Sanitary napkins/pads   • Broken mercury thermometers
   • Meat/fish protein waste • HDPE containers (Resin 2) • Used tampons & wipes    • Expired pharmaceuticals
   • Spent tea & coffee      • Aluminium & tin cans      • Condoms & prophylactics • Discarded paint/solvent cans
   • Eggshells               • Glass bottles & cullet    • Incontinence underpads  • Pesticide & aerosol cans
   • Floral worship waste    • Clean textiles                                      • Waste batteries (Dry/Li-ion)
   • Coconut shells          • Non-recyclable MLP/RDF                              • E-waste (Cables, phones, PCB)
         │                         │                             │                         │
         ▼                         ▼                             ▼                         ▼
   Decentralized MCCs,       Material Recovery           Dedicated High-Temp       Municipal Deposition Centres
   Vermi-composting, or      Facilities (MRFs),          Incineration or CBWTF     (1 per 5 sq. km), Registered
   Biomethanation Plants     Recyclers, or Cement Kiln   Autoclave/Incineration    EPR Recyclers, or Hazardous
   (Rule 39(19), 39(23))     Co-Processing (RDF)         (Rule 39(2)(xii), 8(f))   Disposal (Rule 39(48), 8(ii))
```

### Detailed Stream Specifications

#### Stream 1: Wet Waste (Organic / Biodegradable)
- **Legal Definition**: Rule 3(zzl) — `"wet waste" means and includes organic waste including kitchen waste, food waste, vegetable waste, meat waste, fruits waste, flower waste, and such similar waste and biodegradable waste.`
- **Mandatory Bin Color**: **Green** (Rule 39(51)).
- **Handling Protocol**: Segregate at source without plastic wrappers (Rule 5(1)(b)).
- **Processing Pathway**: On-site home composting, decentralized Micro Composting Centres (MCCs), or biomethanation plants for compressed biogas generation (Rule 3(e), Rule 39(19), Rule 39(23)).

#### Stream 2: Dry Waste (Recyclable & Combustible Inerts)
- **Legal Definition**: Rule 3(s) — `"dry waste" means waste other than wet waste, sanitary waste, special care waste and includes recyclable waste and non-recyclable waste.`
- **Mandatory Bin Color**: **Blue** (Rule 39(51)).
- **Sub-streams**:
  1. *Recyclable Fraction*: Paper, cardboard, PET (Resin 1), HDPE (Resin 2), PP (Resin 5), aluminium cans, ferrous tin cans, glass bottles.
  2. *Non-Recyclable Combustible Fraction*: Multi-layered plastic (MLP) wrappers, laminated pouches, single-use plastics conforming to minimum calorific value (> 1500 kcal/kg) under Rule 3(l) and Rule 3(zl) for Refuse Derived Fuel (RDF) and cement kiln co-processing.
  3. *Inert Fraction*: Road dust sweepings, surface drain silt (Rule 3(z)), strictly prohibited from mixing with wet waste (Rule 39(33)) and restricted to engineered sanitary landfills (Rule 39(49), Schedule II).

#### Stream 3: Sanitary Waste (Absorbent Hygiene Products)
- **Legal Definition**: Rule 3(zp) — `"sanitary waste" means waste comprising of used diapers, sanitary towels or napkins, tampons, condoms, incontinence sheets and any other similar waste.`
- **Mandatory Bin Color**: Red bin in public toilets (Rule 39(51)); wrapped at home (Rule 5(1)(c)).
- **Handling Protocol**: Must be securely wrapped in manufacturer-provided pouches or wrapping material instructed by local bodies, placed in a separate sanitary bin (Rule 5(1)(c)). Never flushed into commodes or mixed with dry/wet streams.
- **Processing Pathway**: Routed to dedicated high-temperature incinerators or Common Bio-medical Waste Treatment Facilities (CBWTF) under Rule 39(2)(xii) and Rule 8(f).

#### Stream 4: Special Care Waste (Domestic Hazardous & Extended Producer Streams)
- **Legal Definition**: Rule 3(zx) — `"special care waste" means and includes discarded paint drums, pesticide cans or containers or bottles, compact fluorescent lamp or bulbs, tube lights, expired medicines, broken mercury thermometers, waste batteries, used or waste needles and syringes and contaminated gauge, or any other waste notified by Central Pollution Control Board from time to time, generated at the household level.`
- **Specialized Statutory Regimes**:
  - *Battery Waste*: Governed under Battery Waste Management Rules, 2022 (lead-acid, lithium-ion, alkaline). Prohibited from municipal solid waste and landfills.
  - *Electronic Waste*: Governed under E-Waste (Management) Rules, 2022. IT and consumer electronics channelized to registered EPR dismantlers.
  - *Household Sharps*: Needles, lancets, and syringes contained in puncture-proof containers for treatment at CBWTFs.
- **Processing Pathway**: Urban Local Bodies are mandated to set up Deposition Centres (at least one per 5 sq. km in cities and towns) with notified reception timings (Rule 39(48)).

---

## 3. Data Schema & Source Traceability Model

The knowledge base JSON (`backend/data/waste_knowledge_base.json`) enforces a strict, verifiable schema. Every factual field must be substantiated by an identifiable source reference:

```json
{
  "class_id": "string (unique identifier, e.g. 'wet_kitchen_food_scraps')",
  "item_name": "string (official display name)",
  "alternative_names": ["array of synonyms, colloquial terms, and Tamil names"],
  "primary_stream": "Wet waste | Dry waste | Sanitary waste | Special care waste",
  "subcategory": "string (granulated categorization)",
  "description": "string (grounded in statutory definitions)",
  "examples": ["array of specific common items"],
  "segregation_instructions": "string (statutory generator obligations)",
  "preparation_instructions": "string | null (physical pre-disposal steps)",
  "handling_precautions": "string | null (safety and hazard protocols)",
  "recycling_or_disposal_guidance": "string | null (statutory processing facilities)",
  "environmental_information": "string | null (scientific/ecological facts, no hallucinated stats)",
  "source_references": [
    {
      "document_title": "string (official gazette/order title)",
      "issuing_organisation": "string (MoEFCC, CPCB, TNPCB, etc.)",
      "official_url": "string (government domain link)",
      "rule_or_section": "string (specific rule/section citation)",
      "page_number": "string (exact gazette/document page)",
      "publication_or_notification_date": "YYYY-MM-DD"
    }
  ],
  "verification_status": "verified | pending_verification"
}
```

### Representation of Missing Information
Where statutory sources do not explicitly establish a fact (such as a specific municipal collection schedule or private recycler address), the field is strictly set to `null` or marked as "Not established by current sources". Fabricated data (e.g. invented recycling percentages or fake local depot addresses) is explicitly prevented by automated validation tests.

---

## 4. Knowledge Retrieval Architecture (`waste_service.py`)

The retrieval service located at [`backend/app/knowledge/waste_service.py`](file:///c:/Users/pavan/Documents/Bharat%20AI-SoC%20Challenge%20%E2%80%94%202026%E2%80%9327/ecosort-ai/backend/app/knowledge/waste_service.py) implements pure, deterministic query methods:

1. **`get_item_by_class_id(class_id)`**:
   - Time Complexity: $\mathcal{O}(1)$ dictionary lookup.
   - Returns the complete verified entry or `None`.
2. **`search_by_name(query)`**:
   - Performs bilingual substring matching across `item_name`, `alternative_names` (including Tamil script terms), and `examples`.
   - Returns matched items without synthetic expansions.
3. **`get_items_by_stream(primary_stream)`**:
   - Filters the database by one of the four statutory streams.
4. **`get_handling_and_disposal_recommendation(class_id)`**:
   - Returns targeted disposal, handling, and preparation advice paired with exact source citations and a boolean `is_pending_verification` flag.
5. **`get_stream_coverage_summary()`**:
   - Computes live breakdown of verified vs. pending entries across streams.

---

## 5. Automated Validation & Quality Assurance

The test suite in [`backend/tests/test_waste_knowledge_base.py`](file:///c:/Users/pavan/Documents/Bharat%20AI-SoC%20Challenge%20%E2%80%94%202026%E2%80%9327/ecosort-ai/backend/tests/test_waste_knowledge_base.py) runs on `pytest` and verifies 13 critical statutory constraints:

| Test Name | Validation Criterion | Result |
| :--- | :--- | :--- |
| `test_metadata_present` | Verifies metadata block references SWM Rules 2026 | **PASSED** |
| `test_no_duplicate_class_ids` | Enforces uniqueness of all `class_id` keys | **PASSED** |
| `test_every_entry_has_valid_primary_stream` | Rejects any stream outside the 4 statutory categories | **PASSED** |
| `test_four_stream_coverage` | Asserts non-zero verified coverage across all 4 streams | **PASSED** |
| `test_every_entry_has_source_references` | Enforces non-empty source list for all items | **PASSED** |
| `test_source_references_completeness` | Asserts presence of title, organisation, rule, and page | **PASSED** |
| `test_no_unsupported_disposal_recommendations` | Rejects disposal guidance lacking source backing | **PASSED** |
| `test_missing_information_is_explicit` | Verifies unestablished information is explicitly `null` | **PASSED** |
| `test_pending_entries_are_identifiable` | Confirms pending verification flags are active | **PASSED** |
| `test_verified_entries_are_clean` | Validates complete recommendations for verified entries | **PASSED** |
| `test_retrieve_by_class_id` | Verifies exact retrieval and `None` on unknown keys | **PASSED** |
| `test_search_by_name_and_tamil_alias` | Verifies English and Tamil search capability | **PASSED** |
| `test_retrieve_by_stream` | Verifies stream filtering and statutory section presence | **PASSED** |
