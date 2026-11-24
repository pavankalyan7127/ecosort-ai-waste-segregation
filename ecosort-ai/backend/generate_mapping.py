"""Script to generate dataset_taxonomy_mapping.json for all 60 TACO categories."""

import json
from pathlib import Path

TACO_MAPPING = [
    {
        "taco_category_id": 0,
        "taco_category_name": "Aluminium foil",
        "taco_supercategory": "Aluminium foil",
        "ecosort_class_id": "dry_metal_aluminium_cans",
        "primary_stream": "Dry waste",
        "mapping_status": "direct",
        "explanation": "Clean household aluminium foil is statutory dry recyclable metal waste under Solid Waste Management Rules, 2026 (Rule 3(s), Rule 3(zw))."
    },
    {
        "taco_category_id": 1,
        "taco_category_name": "Battery",
        "taco_supercategory": "Battery",
        "ecosort_class_id": "special_batteries_household_dry_cell",
        "primary_stream": "Special care waste",
        "mapping_status": "direct",
        "explanation": "Portable dry-cell and household batteries are statutory Special care waste under SWM Rules 2026 Rule 3(zy) and Battery Waste Management Rules 2022."
    },
    {
        "taco_category_id": 2,
        "taco_category_name": "Aluminium blister pack",
        "taco_supercategory": "Blister pack",
        "ecosort_class_id": "dry_plastic_mlp_non_recyclable",
        "primary_stream": "Dry waste",
        "mapping_status": "partial",
        "explanation": "Empty consumer foil packaging is non-recyclable multi-layered composite. Note: Blister packs with residual pharmaceutical tablets belong to Special care waste."
    },
    {
        "taco_category_id": 3,
        "taco_category_name": "Carded blister pack",
        "taco_supercategory": "Blister pack",
        "ecosort_class_id": "dry_paper_cardboard_cartons",
        "primary_stream": "Dry waste",
        "mapping_status": "partial",
        "explanation": "Composite cardboard backing with thermoformed plastic bubble; requires manual separation of paper from polymer before recycling."
    },
    {
        "taco_category_id": 4,
        "taco_category_name": "Other plastic bottle",
        "taco_supercategory": "Bottle",
        "ecosort_class_id": "dry_plastic_hdpe_containers",
        "primary_stream": "Dry waste",
        "mapping_status": "partial",
        "explanation": "Non-clear rigid plastic bottles predominantly consist of High-Density Polyethylene (HDPE, Resin Code 2) or PP, collected as dry recyclable plastics."
    },
    {
        "taco_category_id": 5,
        "taco_category_name": "Clear plastic bottle",
        "taco_supercategory": "Bottle",
        "ecosort_class_id": "dry_plastic_pet_bottles",
        "primary_stream": "Dry waste",
        "mapping_status": "direct",
        "explanation": "Clear beverage and drinking water bottles are standard Polyethylene Terephthalate (PET, Resin Code 1) segregated into the Dry waste stream."
    },
    {
        "taco_category_id": 6,
        "taco_category_name": "Glass bottle",
        "taco_supercategory": "Bottle",
        "ecosort_class_id": "dry_glass_bottles_jars",
        "primary_stream": "Dry waste",
        "mapping_status": "direct",
        "explanation": "Intact glass bottles (beverage, food) are 100% recyclable dry waste under SWM Rules 2026 Rule 3(zw)."
    },
    {
        "taco_category_id": 7,
        "taco_category_name": "Plastic bottle cap",
        "taco_supercategory": "Bottle cap",
        "ecosort_class_id": "dry_plastic_pp_containers",
        "primary_stream": "Dry waste",
        "mapping_status": "direct",
        "explanation": "Threaded bottle closures are injection-molded Polypropylene (PP, Resin Code 5) or HDPE, segregated with dry plastics."
    },
    {
        "taco_category_id": 8,
        "taco_category_name": "Metal bottle cap",
        "taco_supercategory": "Bottle cap",
        "ecosort_class_id": "dry_metal_steel_tin_cans",
        "primary_stream": "Dry waste",
        "mapping_status": "direct",
        "explanation": "Crown corks and metal twist closures (steel/tinplate) belong to the dry recyclable metal stream."
    },
    {
        "taco_category_id": 9,
        "taco_category_name": "Broken glass",
        "taco_supercategory": "Broken glass",
        "ecosort_class_id": "dry_glass_broken_shards",
        "primary_stream": "Dry waste",
        "mapping_status": "direct",
        "explanation": "Broken glass shards are dry recyclable cullet but require puncture-resistant packaging per SWM Rules 2026 Schedule II, Part A."
    },
    {
        "taco_category_id": 10,
        "taco_category_name": "Food Can",
        "taco_supercategory": "Can",
        "ecosort_class_id": "dry_metal_steel_tin_cans",
        "primary_stream": "Dry waste",
        "mapping_status": "direct",
        "explanation": "Tinned food cans are tin-plated steel containers, segregated into the dry recyclable metal stream."
    },
    {
        "taco_category_id": 11,
        "taco_category_name": "Aerosol",
        "taco_supercategory": "Can",
        "ecosort_class_id": "pending_cosmetic_aerosols",
        "primary_stream": "Special care waste",
        "mapping_status": "partial",
        "explanation": "Pressurised propellant canisters present explosion hazards during compaction; pesticide/chemical aerosols are statutory Special care waste."
    },
    {
        "taco_category_id": 12,
        "taco_category_name": "Drink can",
        "taco_supercategory": "Can",
        "ecosort_class_id": "dry_metal_aluminium_cans",
        "primary_stream": "Dry waste",
        "mapping_status": "direct",
        "explanation": "Beverage cans are manufactured from high-grade aluminium alloy, universally collected as dry recyclable metal."
    },
    {
        "taco_category_id": 13,
        "taco_category_name": "Toilet tube",
        "taco_supercategory": "Carton",
        "ecosort_class_id": "dry_paper_cardboard_cartons",
        "primary_stream": "Dry waste",
        "mapping_status": "direct",
        "explanation": "Paperboard and cardboard roll cores are recyclable cellulose fibers segregated into the dry waste stream."
    },
    {
        "taco_category_id": 14,
        "taco_category_name": "Other carton",
        "taco_supercategory": "Carton",
        "ecosort_class_id": "dry_paper_cardboard_cartons",
        "primary_stream": "Dry waste",
        "mapping_status": "partial",
        "explanation": "Clean dry paperboard packaging boxes belong to dry paper streams provided they are uncontaminated by liquids."
    },
    {
        "taco_category_id": 15,
        "taco_category_name": "Egg carton",
        "taco_supercategory": "Carton",
        "ecosort_class_id": "dry_paper_cardboard_cartons",
        "primary_stream": "Dry waste",
        "mapping_status": "direct",
        "explanation": "Molded pulp paper egg cartons are biodegradable cellulose fibers classified as dry recyclable or compostable."
    },
    {
        "taco_category_id": 16,
        "taco_category_name": "Drink carton",
        "taco_supercategory": "Carton",
        "ecosort_class_id": "dry_paper_tetra_packs",
        "primary_stream": "Dry waste",
        "mapping_status": "direct",
        "explanation": "Aseptic composite beverage cartons (Tetra Pak) composed of paperboard, polyethylene, and aluminium foil are dry recyclable."
    },
    {
        "taco_category_id": 17,
        "taco_category_name": "Corrugated carton",
        "taco_supercategory": "Carton",
        "ecosort_class_id": "dry_paper_cardboard_cartons",
        "primary_stream": "Dry waste",
        "mapping_status": "direct",
        "explanation": "Corrugated fiberboard shipping boxes are prime high-yield dry recyclable packaging under SWM Rules 2026."
    },
    {
        "taco_category_id": 18,
        "taco_category_name": "Meal carton",
        "taco_supercategory": "Carton",
        "ecosort_class_id": "dry_paper_cardboard_cartons",
        "primary_stream": "Dry waste",
        "mapping_status": "partial",
        "explanation": "Unsoiled paper meal boxes belong to dry paper; heavily oil-saturated containers cannot be recycled and must be diverted to composting."
    },
    {
        "taco_category_id": 19,
        "taco_category_name": "Pizza box",
        "taco_supercategory": "Carton",
        "ecosort_class_id": "dry_paper_cardboard_cartons",
        "primary_stream": "Dry waste",
        "mapping_status": "partial",
        "explanation": "Clean cardboard box lids are dry recyclable; oil-greased box bottoms must be separated and composted with wet waste."
    },
    {
        "taco_category_id": 20,
        "taco_category_name": "Paper cup",
        "taco_supercategory": "Cup",
        "ecosort_class_id": "dry_paper_cardboard_cartons",
        "primary_stream": "Dry waste",
        "mapping_status": "partial",
        "explanation": "Polyethylene-lined disposable paper cups require specialized delamination; collected under dry non-recyclable or composite paper."
    },
    {
        "taco_category_id": 21,
        "taco_category_name": "Disposable plastic cup",
        "taco_supercategory": "Cup",
        "ecosort_class_id": "dry_plastic_pp_containers",
        "primary_stream": "Dry waste",
        "mapping_status": "direct",
        "explanation": "Polypropylene (PP) or PET beverage cups belong to dry waste (or banned single-use plastics under PWM Rules depending on thickness)."
    },
    {
        "taco_category_id": 22,
        "taco_category_name": "Foam cup",
        "taco_supercategory": "Cup",
        "ecosort_class_id": "dry_plastic_single_use_banned",
        "primary_stream": "Dry waste",
        "mapping_status": "direct",
        "explanation": "Expanded polystyrene (thermocol) cups are strictly prohibited under Plastic Waste Management (Amendment) Rules, 2021/2022."
    },
    {
        "taco_category_id": 23,
        "taco_category_name": "Glass cup",
        "taco_supercategory": "Cup",
        "ecosort_class_id": "dry_glass_bottles_jars",
        "primary_stream": "Dry waste",
        "mapping_status": "direct",
        "explanation": "Intact drinking glassware is classified under dry recyclable glass streams."
    },
    {
        "taco_category_id": 24,
        "taco_category_name": "Other plastic cup",
        "taco_supercategory": "Cup",
        "ecosort_class_id": "dry_plastic_pp_containers",
        "primary_stream": "Dry waste",
        "mapping_status": "partial",
        "explanation": "Rigid polymeric drinkware belongs to dry plastic recycling."
    },
    {
        "taco_category_id": 25,
        "taco_category_name": "Food waste",
        "taco_supercategory": "Food waste",
        "ecosort_class_id": "wet_kitchen_food_scraps",
        "primary_stream": "Wet waste",
        "mapping_status": "direct",
        "explanation": "Discarded organic food scraps are statutory Wet waste (compostable) under SWM Rules 2026 Rule 3(zzl)."
    },
    {
        "taco_category_id": 26,
        "taco_category_name": "Glass jar",
        "taco_supercategory": "Glass jar",
        "ecosort_class_id": "dry_glass_bottles_jars",
        "primary_stream": "Dry waste",
        "mapping_status": "direct",
        "explanation": "Glass preserve, jam, and condiment jars are 100% recyclable dry glass packaging."
    },
    {
        "taco_category_id": 27,
        "taco_category_name": "Plastic lid",
        "taco_supercategory": "Lid",
        "ecosort_class_id": "dry_plastic_pp_containers",
        "primary_stream": "Dry waste",
        "mapping_status": "direct",
        "explanation": "Rigid plastic container lids (PP/PE) are dry recyclable plastics."
    },
    {
        "taco_category_id": 28,
        "taco_category_name": "Metal lid",
        "taco_supercategory": "Lid",
        "ecosort_class_id": "dry_metal_steel_tin_cans",
        "primary_stream": "Dry waste",
        "mapping_status": "direct",
        "explanation": "Steel twist-off lids from glass jars are dry recyclable metal."
    },
    {
        "taco_category_id": 29,
        "taco_category_name": "Other plastic",
        "taco_supercategory": "Other plastic",
        "ecosort_class_id": None,
        "primary_stream": "Unsupported",
        "mapping_status": "ambiguous",
        "explanation": "Catch-all category with unverified resin identity; cannot be mapped without risking misclassifying non-recyclable or hazardous polymers."
    },
    {
        "taco_category_id": 30,
        "taco_category_name": "Magazine paper",
        "taco_supercategory": "Paper",
        "ecosort_class_id": "dry_paper_newspaper",
        "primary_stream": "Dry waste",
        "mapping_status": "direct",
        "explanation": "Glossy printed magazine periodicals and coated paper are dry recyclable paper."
    },
    {
        "taco_category_id": 31,
        "taco_category_name": "Tissues",
        "taco_supercategory": "Paper",
        "ecosort_class_id": None,
        "primary_stream": "Unsupported",
        "mapping_status": "ambiguous",
        "explanation": "Ambiguous status: Tissues soiled with bodily fluids/mucus are sanitary waste, while clean cellulose or food napkins are compostable/inert."
    },
    {
        "taco_category_id": 32,
        "taco_category_name": "Wrapping paper",
        "taco_supercategory": "Paper",
        "ecosort_class_id": "dry_paper_newspaper",
        "primary_stream": "Dry waste",
        "mapping_status": "direct",
        "explanation": "Plain cellulose gift wrapping paper is dry recyclable paper."
    },
    {
        "taco_category_id": 33,
        "taco_category_name": "Normal paper",
        "taco_supercategory": "Paper",
        "ecosort_class_id": "dry_paper_newspaper",
        "primary_stream": "Dry waste",
        "mapping_status": "direct",
        "explanation": "Office stationery, printed documents, and notebooks are dry recyclable paper."
    },
    {
        "taco_category_id": 34,
        "taco_category_name": "Paper bag",
        "taco_supercategory": "Paper bag",
        "ecosort_class_id": "dry_paper_newspaper",
        "primary_stream": "Dry waste",
        "mapping_status": "direct",
        "explanation": "Clean unlaminated kraft paper shopping bags are dry recyclable paper."
    },
    {
        "taco_category_id": 35,
        "taco_category_name": "Plastified paper bag",
        "taco_supercategory": "Paper bag",
        "ecosort_class_id": "dry_plastic_mlp_non_recyclable",
        "primary_stream": "Dry waste",
        "mapping_status": "partial",
        "explanation": "Polymer-laminated paper bags are composite multi-material packaging (0 annotations currently in TACO)."
    },
    {
        "taco_category_id": 36,
        "taco_category_name": "Plastic film",
        "taco_supercategory": "Plastic bag & wrapper",
        "ecosort_class_id": "dry_plastic_ldpe_films",
        "primary_stream": "Dry waste",
        "mapping_status": "direct",
        "explanation": "Flexible Low-Density Polyethylene (LDPE, Resin Code 4) packaging film and stretch wrap belong to dry recyclable plastics."
    },
    {
        "taco_category_id": 37,
        "taco_category_name": "Six pack rings",
        "taco_supercategory": "Plastic bag & wrapper",
        "ecosort_class_id": "dry_plastic_ldpe_films",
        "primary_stream": "Dry waste",
        "mapping_status": "direct",
        "explanation": "Photodegradable LDPE can carrier rings belong to dry plastic waste."
    },
    {
        "taco_category_id": 38,
        "taco_category_name": "Garbage bag",
        "taco_supercategory": "Plastic bag & wrapper",
        "ecosort_class_id": "dry_plastic_ldpe_films",
        "primary_stream": "Dry waste",
        "mapping_status": "partial",
        "explanation": "Clean polyethylene refuse sacks are dry plastic; if containing waste, segregated by contents."
    },
    {
        "taco_category_id": 39,
        "taco_category_name": "Other plastic wrapper",
        "taco_supercategory": "Plastic bag & wrapper",
        "ecosort_class_id": "dry_plastic_mlp_non_recyclable",
        "primary_stream": "Dry waste",
        "mapping_status": "partial",
        "explanation": "Generic packaging wrappers predominantly consist of metallised BOPP or multi-layer plastics directed to refuse-derived fuel."
    },
    {
        "taco_category_id": 40,
        "taco_category_name": "Single-use carrier bag",
        "taco_supercategory": "Plastic bag & wrapper",
        "ecosort_class_id": "dry_plastic_single_use_banned",
        "primary_stream": "Dry waste",
        "mapping_status": "direct",
        "explanation": "Single-use plastic carry bags (< 120 microns) are banned articles under Plastic Waste Management Rules, 2016/2022."
    },
    {
        "taco_category_id": 41,
        "taco_category_name": "Polypropylene bag",
        "taco_supercategory": "Plastic bag & wrapper",
        "ecosort_class_id": "dry_plastic_pp_containers",
        "primary_stream": "Dry waste",
        "mapping_status": "direct",
        "explanation": "Woven polypropylene grain sacks and shopping bags are recyclable Polypropylene (PP, Resin Code 5)."
    },
    {
        "taco_category_id": 42,
        "taco_category_name": "Crisp packet",
        "taco_supercategory": "Plastic bag & wrapper",
        "ecosort_class_id": "dry_plastic_mlp_non_recyclable",
        "primary_stream": "Dry waste",
        "mapping_status": "direct",
        "explanation": "Snack pouches are non-recyclable multi-layered plastics (MLP) with aluminium vapor coating, directed to cement kiln co-processing."
    },
    {
        "taco_category_id": 43,
        "taco_category_name": "Spread tub",
        "taco_supercategory": "Plastic container",
        "ecosort_class_id": "dry_plastic_pp_containers",
        "primary_stream": "Dry waste",
        "mapping_status": "direct",
        "explanation": "Margarine, butter, and spread containers are injection-molded Polypropylene (PP, Resin Code 5)."
    },
    {
        "taco_category_id": 44,
        "taco_category_name": "Tupperware",
        "taco_supercategory": "Plastic container",
        "ecosort_class_id": "dry_plastic_pp_containers",
        "primary_stream": "Dry waste",
        "mapping_status": "direct",
        "explanation": "Reusable rigid food storage containers are recyclable Polypropylene (PP, Resin Code 5)."
    },
    {
        "taco_category_id": 45,
        "taco_category_name": "Disposable food container",
        "taco_supercategory": "Plastic container",
        "ecosort_class_id": "dry_plastic_pp_containers",
        "primary_stream": "Dry waste",
        "mapping_status": "direct",
        "explanation": "Rigid polypropylene meal containers are dry recyclable plastic if rinsed of food remnants."
    },
    {
        "taco_category_id": 46,
        "taco_category_name": "Foam food container",
        "taco_supercategory": "Plastic container",
        "ecosort_class_id": "dry_plastic_single_use_banned",
        "primary_stream": "Dry waste",
        "mapping_status": "direct",
        "explanation": "Expanded polystyrene takeaway clamshells are banned single-use plastic articles under Plastic Waste Management Rules."
    },
    {
        "taco_category_id": 47,
        "taco_category_name": "Other plastic container",
        "taco_supercategory": "Plastic container",
        "ecosort_class_id": "dry_plastic_hdpe_containers",
        "primary_stream": "Dry waste",
        "mapping_status": "partial",
        "explanation": "Miscellaneous rigid plastic containers and tubs belong to dry plastic recycling."
    },
    {
        "taco_category_id": 48,
        "taco_category_name": "Plastic glooves",
        "taco_supercategory": "Plastic glooves",
        "ecosort_class_id": None,
        "primary_stream": "Unsupported",
        "mapping_status": "ambiguous",
        "explanation": "Ambiguous between clinical gloves (statutory biomedical / Special care waste under BMWM Rules) and domestic kitchen gloves."
    },
    {
        "taco_category_id": 49,
        "taco_category_name": "Plastic utensils",
        "taco_supercategory": "Plastic utensils",
        "ecosort_class_id": "dry_plastic_single_use_banned",
        "primary_stream": "Dry waste",
        "mapping_status": "direct",
        "explanation": "Single-use plastic cutlery (forks, spoons, knives) is banned under Plastic Waste Management Rules."
    },
    {
        "taco_category_id": 50,
        "taco_category_name": "Pop tab",
        "taco_supercategory": "Pop tab",
        "ecosort_class_id": "dry_metal_aluminium_cans",
        "primary_stream": "Dry waste",
        "mapping_status": "direct",
        "explanation": "Aluminium can pull-rings are dry recyclable aluminium metal."
    },
    {
        "taco_category_id": 51,
        "taco_category_name": "Rope & strings",
        "taco_supercategory": "Rope & strings",
        "ecosort_class_id": None,
        "primary_stream": "Unsupported",
        "mapping_status": "ambiguous",
        "explanation": "Polymer ropes (nylon/PP) are dry waste, whereas natural jute/hemp cords are compostable wet waste; ambiguous without material test."
    },
    {
        "taco_category_id": 52,
        "taco_category_name": "Scrap metal",
        "taco_supercategory": "Scrap metal",
        "ecosort_class_id": "dry_metal_steel_tin_cans",
        "primary_stream": "Dry waste",
        "mapping_status": "direct",
        "explanation": "Ferrous and non-ferrous metal scraps are dry recyclable metal under SWM Rules 2026."
    },
    {
        "taco_category_id": 53,
        "taco_category_name": "Shoe",
        "taco_supercategory": "Shoe",
        "ecosort_class_id": "pending_composite_footwear",
        "primary_stream": "Dry waste",
        "mapping_status": "direct",
        "explanation": "Discarded footwear composed of composite rubber, EVA, and textiles belongs to dry combustible / co-processing streams."
    },
    {
        "taco_category_id": 54,
        "taco_category_name": "Squeezable tube",
        "taco_supercategory": "Squeezable tube",
        "ecosort_class_id": "dry_plastic_mlp_non_recyclable",
        "primary_stream": "Dry waste",
        "mapping_status": "partial",
        "explanation": "Cosmetic and toothpaste tubes are composite plastic-aluminium laminates directed to refuse-derived fuel."
    },
    {
        "taco_category_id": 55,
        "taco_category_name": "Plastic straw",
        "taco_supercategory": "Straw",
        "ecosort_class_id": "dry_plastic_single_use_banned",
        "primary_stream": "Dry waste",
        "mapping_status": "direct",
        "explanation": "Plastic drinking straws are prohibited single-use plastics under Plastic Waste Management Rules."
    },
    {
        "taco_category_id": 56,
        "taco_category_name": "Paper straw",
        "taco_supercategory": "Straw",
        "ecosort_class_id": "dry_paper_newspaper",
        "primary_stream": "Dry waste",
        "mapping_status": "direct",
        "explanation": "Bio-based cellulose paper drinking straws are dry waste or compostable."
    },
    {
        "taco_category_id": 57,
        "taco_category_name": "Styrofoam piece",
        "taco_supercategory": "Styrofoam piece",
        "ecosort_class_id": "dry_plastic_single_use_banned",
        "primary_stream": "Dry waste",
        "mapping_status": "direct",
        "explanation": "Expanded polystyrene packaging blocks and loose insulation belong to dry plastic waste."
    },
    {
        "taco_category_id": 58,
        "taco_category_name": "Unlabeled litter",
        "taco_supercategory": "Unlabeled litter",
        "ecosort_class_id": None,
        "primary_stream": "Unsupported",
        "mapping_status": "unsupported",
        "explanation": "Unclassified litter fragments cannot be mapped to any statutory stream without violating zero-hallucination principles."
    },
    {
        "taco_category_id": 59,
        "taco_category_name": "Cigarette",
        "taco_supercategory": "Cigarette",
        "ecosort_class_id": "dry_inerts_street_sweepings",
        "primary_stream": "Dry waste",
        "mapping_status": "direct",
        "explanation": "Discarded cigarette butts with cellulose acetate filters and toxic chemical residues are non-recyclable inerts/combustible dry waste under SWM Rules 2026."
    }
]

out_path = Path("ecosort-ai/backend/app/vision/dataset_taxonomy_mapping.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump({
        "dataset_name": "TACO (Trash Annotations in Context)",
        "source_url": "https://github.com/pedropro/TACO",
        "total_taco_categories": len(TACO_MAPPING),
        "mapping_summary": {
            "direct": sum(1 for m in TACO_MAPPING if m["mapping_status"] == "direct"),
            "partial": sum(1 for m in TACO_MAPPING if m["mapping_status"] == "partial"),
            "ambiguous": sum(1 for m in TACO_MAPPING if m["mapping_status"] == "ambiguous"),
            "unsupported": sum(1 for m in TACO_MAPPING if m["mapping_status"] == "unsupported"),
        },
        "stream_coverage": {
            "Dry waste": sum(1 for m in TACO_MAPPING if m["primary_stream"] == "Dry waste"),
            "Wet waste": sum(1 for m in TACO_MAPPING if m["primary_stream"] == "Wet waste"),
            "Special care waste": sum(1 for m in TACO_MAPPING if m["primary_stream"] == "Special care waste"),
            "Sanitary waste": sum(1 for m in TACO_MAPPING if m["primary_stream"] == "Sanitary waste"),
            "Unsupported": sum(1 for m in TACO_MAPPING if m["primary_stream"] == "Unsupported"),
        },
        "categories": TACO_MAPPING
    }, f, indent=2)

print("Generated dataset_taxonomy_mapping.json successfully!")
print("Mapping Summary:", json.dumps({
    "direct": sum(1 for m in TACO_MAPPING if m["mapping_status"] == "direct"),
    "partial": sum(1 for m in TACO_MAPPING if m["mapping_status"] == "partial"),
    "ambiguous": sum(1 for m in TACO_MAPPING if m["mapping_status"] == "ambiguous"),
    "unsupported": sum(1 for m in TACO_MAPPING if m["mapping_status"] == "unsupported"),
}, indent=2))
