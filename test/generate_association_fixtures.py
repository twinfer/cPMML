#!/usr/bin/env python3
"""
test/generate_association_fixtures.py

Run:
    python test/generate_association_fixtures.py

Outputs (overwriting any existing files):
    test/data/model/AssociationRules.zip            recommendation (default)
    test/data/dataset/AssociationRules.csv
    test/data/model/AssociationRulesExcl.zip        exclusiveRecommendation
    test/data/dataset/AssociationRulesExcl.csv
    test/data/model/AssociationRulesRuleAssoc.zip   ruleAssociation
    test/data/dataset/AssociationRulesRuleAssoc.csv
"""

from __future__ import annotations

import csv
import textwrap
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Repo layout
# ---------------------------------------------------------------------------
REPO_ROOT   = Path(__file__).parent.parent
MODEL_DIR   = REPO_ROOT / "test" / "data" / "model"
DATASET_DIR = REPO_ROOT / "test" / "data" / "dataset"

# ---------------------------------------------------------------------------
# Data structures (mirror C++ AssociationModel internals)
# ---------------------------------------------------------------------------

@dataclass
class Item:
    id: str
    value: str              # label / binary field name
    mapped_value: str = ""  # display substitute for value (optional)
    weight: float = 1.0

@dataclass
class Itemset:
    id: str
    item_refs: frozenset[str]   # set of Item ids
    support: float = 0.0

@dataclass
class AssociationRule:
    antecedent: str             # Itemset id
    consequent: str             # Itemset id
    support: float
    confidence: float
    lift: float = 1.0
    id: str = ""

@dataclass
class Model:
    name: str
    n_transactions: int
    min_support: float
    min_confidence: float
    items: dict[str, Item]          # item id → Item
    itemsets: dict[str, Itemset]    # itemset id → Itemset
    rules: list[AssociationRule]
    algorithm: str = "recommendation"   # recommendation | exclusiveRecommendation | ruleAssociation

# ---------------------------------------------------------------------------
# Scorer — mirrors C++ AssociationModel::score_raw / predict_raw exactly
#
# Binary encoding:  item.value == field name in basket dict.
#                   Item is active when basket[item.value] != 0.
# ---------------------------------------------------------------------------

def _active_items(model: Model, basket: dict[str, int]) -> frozenset[str]:
    """Return frozenset of item IDs whose corresponding basket field is non-zero."""
    return frozenset(
        item.id
        for item in model.items.values()
        if basket.get(item.value, 0) != 0
    )

def _consequent_str(rule: AssociationRule, model: Model) -> str:
    """
    Deterministic string for a rule's consequent.
    Multi-item consequents are sorted and comma-joined (matches C++ consequent_value()).
    """
    vals = sorted(
        model.items[iid].mapped_value or model.items[iid].value
        for iid in model.itemsets[rule.consequent].item_refs
    )
    return ",".join(vals)

def _antecedent_matches(rule: AssociationRule, active: frozenset[str],
                        model: Model) -> bool:
    return model.itemsets[rule.antecedent].item_refs.issubset(active)

def _consequent_ok(rule: AssociationRule, active: frozenset[str],
                   model: Model) -> bool:
    cons = model.itemsets[rule.consequent].item_refs
    if model.algorithm == "exclusiveRecommendation":
        return cons.isdisjoint(active)       # consequent must NOT be in basket already
    if model.algorithm == "ruleAssociation":
        return cons.issubset(active)         # consequent MUST already be in basket
    return True                              # recommendation: no constraint on consequent

def score(model: Model, basket: dict[str, int]) -> str:
    """
    Score one basket.  Returns the consequent string of the highest-confidence
    matching rule, or "" when no rules fire.

    Tie-breaking: std::ranges::minmax_element / max_element in C++ picks the
    last max in a tie; here we use max() which picks the first.  Avoid ties
    in test cases or the fixture will be fragile.
    """
    active = _active_items(model, basket)
    matching = [
        r for r in model.rules
        if _antecedent_matches(r, active, model) and _consequent_ok(r, active, model)
    ]
    if not matching:
        return ""
    best = max(matching, key=lambda r: r.confidence)
    return _consequent_str(best, model)

# ---------------------------------------------------------------------------
# PMML XML generator
# ---------------------------------------------------------------------------

def _build_pmml(model: Model) -> str:
    # Collect and sort input field names (binary: item.value == field name)
    input_fields = sorted({item.value for item in model.items.values()})

    data_fields_xml = "\n".join(
        f'    <DataField name="{f}" optype="continuous" dataType="integer"/>'
        for f in input_fields
    )
    mining_fields_xml = "\n".join(
        f'      <MiningField name="{f}" usageType="active"/>'
        for f in input_fields
    )

    # Output section: always present so output_name() returns a stable name
    output_xml = (
        "\n    <Output>\n"
        f'      <OutputField name="recommendation" feature="predictedValue"\n'
        f'                   dataType="string" optype="categorical"\n'
        f'                   algorithm="{model.algorithm}"/>\n'
        "    </Output>"
    )

    items_xml = "\n".join(
        f'    <Item id="{item.id}" value="{item.value}"'
        + (f' mappedValue="{item.mapped_value}"' if item.mapped_value else "")
        + (f' weight="{item.weight}"' if item.weight != 1.0 else "")
        + "/>"
        for item in model.items.values()
    )

    def _itemset_xml(iset: Itemset) -> str:
        refs = "\n".join(
            f'      <ItemRef itemRef="{r}"/>' for r in sorted(iset.item_refs)
        )
        sup = f' support="{iset.support}"' if iset.support else ""
        return f'    <Itemset id="{iset.id}"{sup}>\n{refs}\n    </Itemset>'

    itemsets_xml = "\n".join(_itemset_xml(s) for s in model.itemsets.values())

    def _rule_xml(rule: AssociationRule) -> str:
        parts = []
        if rule.id:
            parts.append(f'id="{rule.id}"')
        parts += [
            f'antecedent="{rule.antecedent}"',
            f'consequent="{rule.consequent}"',
            f'support="{rule.support}"',
            f'confidence="{rule.confidence}"',
        ]
        if rule.lift != 1.0:
            parts.append(f'lift="{rule.lift}"')
        return f'    <AssociationRule {" ".join(parts)}/>'

    rules_xml = "\n".join(_rule_xml(r) for r in model.rules)

    return textwrap.dedent(f"""\
        <?xml version="1.0" encoding="UTF-8"?>
        <PMML version="4.4.1" xmlns="http://www.dmg.org/PMML-4_4">
          <Header description="{model.name} — white-box fixture (generate_association_fixtures.py)"/>
          <DataDictionary numberOfFields="{len(input_fields)}">
        {data_fields_xml}
          </DataDictionary>
          <AssociationModel functionName="associationRules"
                            modelName="{model.name}"
                            numberOfTransactions="{model.n_transactions}"
                            minimumSupport="{model.min_support}"
                            minimumConfidence="{model.min_confidence}"
                            numberOfItems="{len(model.items)}"
                            numberOfItemsets="{len(model.itemsets)}"
                            numberOfRules="{len(model.rules)}">{output_xml}
            <MiningSchema>
        {mining_fields_xml}
            </MiningSchema>
        {items_xml}
        {itemsets_xml}
        {rules_xml}
          </AssociationModel>
        </PMML>
        """)

# ---------------------------------------------------------------------------
# Fixture writer
# ---------------------------------------------------------------------------

def write_fixture(model: Model,
                  test_cases: list[tuple[dict[str, int], str]]) -> None:
    """
    Compute expected outputs with the Python scorer, then write:
      test/data/model/<model.name>.zip
      test/data/dataset/<model.name>.csv
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    pmml_xml  = _build_pmml(model)
    zip_path  = MODEL_DIR   / f"{model.name}.zip"
    csv_path  = DATASET_DIR / f"{model.name}.csv"

    # Write PMML inside zip
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{model.name}.pmml", pmml_xml)

    # Input field names (sorted, same order as PMML DataDictionary)
    input_cols = sorted({item.value for item in model.items.values()})

    # Compute expected output and write CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=input_cols + ["recommendation"])
        writer.writeheader()
        for basket, description in test_cases:
            expected = score(model, basket)
            row = {col: basket.get(col, 0) for col in input_cols}
            row["recommendation"] = expected
            writer.writerow(row)
            print(f"    {description:55s} → '{expected}'")

    print(f"  wrote {zip_path.relative_to(REPO_ROOT)}")
    print(f"  wrote {csv_path.relative_to(REPO_ROOT)}")
    print()

# ---------------------------------------------------------------------------
# Model definition
#
# Grocery basket: 5 items, 3 rules
#
#   Rule 1: {bread, butter} → {jam}   conf=0.80  (highest confidence)
#   Rule 2: {bread, milk}   → {eggs}  conf=0.60
#   Rule 3: {butter, jam}   → {bread} conf=0.70
#
# Items use binary encoding: each item is a 0/1 field named after the item.
# ---------------------------------------------------------------------------

ITEMS = {
    "1": Item(id="1", value="bread"),
    "2": Item(id="2", value="butter"),
    "3": Item(id="3", value="milk"),
    "4": Item(id="4", value="jam"),
    "5": Item(id="5", value="eggs"),
}

ITEMSETS = {
    # Antecedents
    "A1": Itemset("A1", frozenset({"1", "2"}), support=0.30),  # {bread, butter}
    "A2": Itemset("A2", frozenset({"1", "3"}), support=0.20),  # {bread, milk}
    "A3": Itemset("A3", frozenset({"2", "4"}), support=0.25),  # {butter, jam}
    # Consequents
    "C1": Itemset("C1", frozenset({"4"}),       support=0.40),  # {jam}
    "C2": Itemset("C2", frozenset({"5"}),       support=0.40),  # {eggs}
    "C3": Itemset("C3", frozenset({"1"}),       support=0.35),  # {bread}
}

RULES = [
    AssociationRule("A1", "C1", support=0.30, confidence=0.80, lift=2.0, id="R1"),
    AssociationRule("A2", "C2", support=0.20, confidence=0.60, lift=1.5, id="R2"),
    AssociationRule("A3", "C3", support=0.25, confidence=0.70, lift=1.8, id="R3"),
]

def make_model(algorithm: str, name: str) -> Model:
    return Model(
        name=name,
        n_transactions=100,
        min_support=0.10,
        min_confidence=0.50,
        items=ITEMS,
        itemsets=ITEMSETS,
        rules=RULES,
        algorithm=algorithm,
    )

# ---------------------------------------------------------------------------
# Test cases
# Each entry: (basket dict, human-readable description)
# ---------------------------------------------------------------------------

RECOMMENDATION_CASES: list[tuple[dict[str, int], str]] = [
    # --- single rule fires ---
    ({"bread": 1, "butter": 1},
     "R1 fires (bread+butter → jam, conf=0.80)"),

    ({"bread": 1, "milk": 1},
     "R2 fires (bread+milk → eggs, conf=0.60)"),

    ({"butter": 1, "jam": 1},
     "R3 fires (butter+jam → bread, conf=0.70)"),

    # --- multiple rules fire, highest confidence wins ---
    ({"bread": 1, "butter": 1, "milk": 1},
     "R1+R2 fire, R1 wins (conf 0.80 > 0.60)"),

    ({"bread": 1, "butter": 1, "jam": 1},
     "R1+R3 fire, R1 wins (conf 0.80 > 0.70)"),

    ({"bread": 1, "butter": 1, "milk": 1, "jam": 1},
     "R1+R2+R3 fire, R1 wins (conf 0.80)"),

    # --- no rules fire ---
    ({"bread": 1},
     "only bread in basket, no antecedent complete → empty"),

    ({},
     "empty basket → empty"),

    # --- items already in basket but still recommended (recommendation allows it) ---
    ({"bread": 1, "butter": 1, "jam": 1},
     "R1 fires: recommends jam even though jam already in basket"),
]

EXCLUSIVE_REC_CASES: list[tuple[dict[str, int], str]] = [
    # --- consequent not in basket → included ---
    ({"bread": 1, "butter": 1},
     "R1 fires, jam not in basket → jam recommended"),

    ({"bread": 1, "milk": 1},
     "R2 fires, eggs not in basket → eggs recommended"),

    # --- consequent already in basket → excluded ---
    ({"bread": 1, "butter": 1, "jam": 1},
     "R1 fires but jam in basket → R1 excluded; R3 fires but bread in basket → excluded → empty"),

    # --- multi-rule: one excluded, one valid ---
    ({"bread": 1, "butter": 1, "milk": 1, "jam": 1},
     "R1: jam in basket→excluded; R2: eggs not in basket→valid; R3: bread in basket→excluded → eggs"),

    # --- all consequents in basket → empty ---
    ({"bread": 1, "butter": 1, "milk": 1, "jam": 1, "eggs": 1},
     "full basket, all consequents already present → empty"),

    # --- no rules fire at all ---
    ({},
     "empty basket → empty"),
]

RULE_ASSOC_CASES: list[tuple[dict[str, int], str]] = [
    # --- antecedent + consequent both in basket ---
    ({"bread": 1, "butter": 1, "jam": 1},
     "R1: antecedent+consequent in basket (jam=1) → jam; R3 also fires (conf 0.80 > 0.70)"),

    ({"butter": 1, "jam": 1, "bread": 1},
     "R1+R3 both fire (all items present), R1 wins (conf 0.80)"),

    ({"bread": 1, "butter": 1, "milk": 1, "jam": 1, "eggs": 1},
     "full basket, all rules fire, R1 wins (conf 0.80)"),

    # --- antecedent fires but consequent not in basket → excluded ---
    ({"bread": 1, "butter": 1},
     "R1 antecedent fires but jam not in basket → excluded → empty"),

    ({"bread": 1, "milk": 1},
     "R2 antecedent fires but eggs not in basket → empty"),

    # --- empty basket ---
    ({},
     "empty basket → empty"),
]

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    fixtures = [
        (make_model("recommendation",          "AssociationRules"),          RECOMMENDATION_CASES),
        (make_model("exclusiveRecommendation", "AssociationRulesExcl"),      EXCLUSIVE_REC_CASES),
        (make_model("ruleAssociation",         "AssociationRulesRuleAssoc"), RULE_ASSOC_CASES),
    ]

    for model, cases in fixtures:
        print(f"\n{'=' * 60}")
        print(f"  {model.name}  [{model.algorithm}]")
        print(f"{'=' * 60}")
        write_fixture(model, cases)


if __name__ == "__main__":
    main()
