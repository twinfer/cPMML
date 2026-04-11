#!/usr/bin/env python3
"""
test/generate_audit_binary_reg_fixture.py
fixture generator for AuditBinaryReg.

The model is a binary logistic regression (PMML RegressionModel,
functionName="classification", normalizationMethod="softmax").

Scoring:
  raw_1 = intercept_1 + Σ(numeric_coefs) + Σ(categorical_coefs)
  raw_0 = 0   (second RegressionTable has intercept=0, no predictors)
  predicted = "1" if raw_1 > raw_0  (i.e. raw_1 > 0)
            = "0" otherwise

Run:
    python test/generate_audit_binary_reg_fixture.py

Outputs:
    test/data/dataset/AuditBinaryReg.csv   (inputs + prediction column)
    (model zip already exists at test/data/model/AuditBinaryReg.zip)
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT   = Path(__file__).parent.parent
DATASET_DIR = REPO_ROOT / "test" / "data" / "dataset"

# ---------------------------------------------------------------------------
# Model coefficients — extracted verbatim from AuditBinaryReg.xml
# (PMML RegressionTable targetCategory="1")
# ---------------------------------------------------------------------------

INTERCEPT: float = -6.47942380092536

NUMERIC_COEFS: dict[str, float] = {
    "Age":        0.0299879200201045,
    "Income":     0.0000036731327315447,
    "Deductions": 0.00114746608389311,
    "Hours":      0.0312349903924695,
}

# Absent / missing categories contribute 0 (PMML default).
# Categories in the DataDictionary but absent from the table
# (e.g. Employment="Unemployed", "Volunteer") also contribute 0.
CATEGORICAL_COEFS: dict[str, dict[str, float]] = {
    "Employment": {
        "Consultant": 0,
        "Private":    0.24749499011722,
        "PSFederal":  0.530473716438337,
        "PSLocal":   -0.0207122873696658,
        "PSState":    0.212050568809577,
        "SelfEmp":    0.576815002280744,
        # "Unemployed" and "Volunteer" not in table → 0
    },
    "Education": {
        "Associate":    0,
        "Bachelor":    -0.296718468841548,
        "College":     -1.24911108355482,
        "Doctorate":    1.27763556496018,
        "HSgrad":      -1.49332772958347,
        "Master":      -0.0232344784395536,
        "Preschool":  -16.0878000781185,
        "Professional": 1.95279174931871,
        "Vocational":  -1.39352308632626,
        "Yr10":        -1.74273851770973,
        "Yr11":        -1.48068009592283,
        "Yr12":        -1.91216334389683,
        "Yr1t4":      -17.5165250616353,
        "Yr5t6":       -3.13764478754663,
        "Yr7t8":      -17.1491909769817,
        "Yr9":         -3.28906157554174,
    },
    "Marital": {
        "Absent":                0,
        "Divorced":             -0.237854919610349,
        "Married":               2.7652891885919,
        "Married-spouse-absent":-0.0422700264784461,
        "Unmarried":             0.290665055436097,
        "Widowed":               0.338979612525046,
    },
    "Occupation": {
        "Cleaner":     0,
        "Clerical":    1.23766090658888,
        "Executive":   1.8349081139858,
        "Farming":     0.228453259831202,
        "Home":       -12.1937763455989,
        "Machinist":   0.764979393114096,
        "Military":  -13.0673115906607,
        "Professional": 1.46855586721193,
        "Protective":  2.10303370845987,
        "Repair":      0.795620651090366,
        "Sales":       0.874412173270209,
        "Service":    -0.496502223605296,
        "Support":     1.05942018386932,
        "Transport":   0.178577748117911,
    },
    "Gender": {
        "Female": 0,
        "Male":   0.375114585003982,
    },
}

# ---------------------------------------------------------------------------
# Scorer — mirrors PMML softmax binary classification
# ---------------------------------------------------------------------------

@dataclass
class Sample:
    Age:        float
    Employment: str
    Education:  str
    Marital:    str
    Occupation: str
    Income:     float
    Gender:     str
    Deductions: float
    Hours:      float

def _raw_score(s: Sample) -> float:
    """Compute the raw linear score for class '1'."""
    score = INTERCEPT
    # Numeric predictors
    score += NUMERIC_COEFS["Age"]        * s.Age
    score += NUMERIC_COEFS["Income"]     * s.Income
    score += NUMERIC_COEFS["Deductions"] * s.Deductions
    score += NUMERIC_COEFS["Hours"]      * s.Hours
    # Categorical predictors (missing category → 0)
    score += CATEGORICAL_COEFS["Employment"].get(s.Employment, 0)
    score += CATEGORICAL_COEFS["Education"].get(s.Education, 0)
    score += CATEGORICAL_COEFS["Marital"].get(s.Marital, 0)
    score += CATEGORICAL_COEFS["Occupation"].get(s.Occupation, 0)
    score += CATEGORICAL_COEFS["Gender"].get(s.Gender, 0)
    return score

def predict(s: Sample) -> str:
    """
    Binary softmax: raw_0 = 0 (empty second table).
    Predicted class = '1' if raw_1 > 0, else '0'.
    """
    return "1" if _raw_score(s) > 0 else "0"

def probability_1(s: Sample) -> float:
    """P(class=1) under softmax, for reference."""
    r = _raw_score(s)
    return math.exp(r) / (math.exp(r) + 1.0)

# ---------------------------------------------------------------------------
# Test cases
# Each tuple: (Sample, description)
# Expected prediction is computed — not hardcoded.
# ---------------------------------------------------------------------------

TEST_CASES: list[tuple[Sample, str]] = [
    # --- clearly class 1 ---
    (Sample(Age=45, Employment="Private",   Education="Bachelor",    Marital="Married",
            Occupation="Executive",   Income=100000, Gender="Male",   Deductions=5000, Hours=50),
     "married executive male, high income/deductions → 1"),

    (Sample(Age=40, Employment="PSFederal", Education="Professional", Marital="Married",
            Occupation="Professional", Income=80000, Gender="Female", Deductions=2000, Hours=45),
     "married federal professional, doctorate-level edu → 1"),

    (Sample(Age=50, Employment="SelfEmp",   Education="Doctorate",   Marital="Married",
            Occupation="Protective",  Income=90000, Gender="Male",   Deductions=3000, Hours=55),
     "married self-employed protective doctorate → 1"),

    (Sample(Age=38, Employment="Private",   Education="Master",      Marital="Married",
            Occupation="Clerical",    Income=70000, Gender="Male",   Deductions=4000, Hours=48),
     "married private clerical master → 1"),

    # --- class 1 via high deductions ---
    (Sample(Age=30, Employment="Private",   Education="College",     Marital="Married",
            Occupation="Sales",       Income=40000, Gender="Male",   Deductions=8000, Hours=40),
     "married, high deductions override low education → 1"),

    # --- clearly class 0 ---
    (Sample(Age=25, Employment="Unemployed", Education="HSgrad",     Marital="Absent",
            Occupation="Service",     Income=10000, Gender="Female", Deductions=0,    Hours=20),
     "young unemployed female, low income → 0"),

    (Sample(Age=65, Employment="Volunteer",  Education="Preschool",  Marital="Divorced",
            Occupation="Home",        Income=0,     Gender="Female", Deductions=0,    Hours=10),
     "volunteer home worker, preschool education → 0"),

    (Sample(Age=22, Employment="Private",    Education="Yr9",        Marital="Absent",
            Occupation="Farming",     Income=15000, Gender="Male",   Deductions=0,    Hours=30),
     "young farmer, low education and income → 0"),

    (Sample(Age=35, Employment="PSLocal",    Education="Yr12",       Marital="Divorced",
            Occupation="Transport",   Income=30000, Gender="Female", Deductions=0,    Hours=38),
     "local govt transport worker, low education → 0"),

    # --- boundary: married boosts over the line ---
    (Sample(Age=28, Employment="Private",    Education="College",    Marital="Married",
            Occupation="Repair",      Income=20000, Gender="Female", Deductions=0,    Hours=35),
     "married college repair, modest income: married bonus tips to 1 or 0"),

    # --- categorical edge cases ---
    (Sample(Age=33, Employment="PSState",    Education="Associate",  Marital="Widowed",
            Occupation="Support",     Income=45000, Gender="Male",   Deductions=500,  Hours=40),
     "state employee, associate edu, widowed male → check"),

    (Sample(Age=55, Employment="Consultant", Education="Vocational", Marital="Absent",
            Occupation="Machinist",   Income=35000, Gender="Male",   Deductions=0,    Hours=45),
     "consultant machinist, vocational edu, absent → 0"),
]

# ---------------------------------------------------------------------------
# Fixture writer
# ---------------------------------------------------------------------------

INPUT_FIELDS = ["Age", "Employment", "Education", "Marital",
                "Occupation", "Income", "Gender", "Deductions", "Hours"]

def write_fixture() -> None:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = DATASET_DIR / "AuditBinaryReg.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=INPUT_FIELDS + ["TARGET_Adjusted"])
        writer.writeheader()
        for sample, description in TEST_CASES:
            expected = predict(sample)
            p1 = probability_1(sample)
            row = {
                "Age":        sample.Age,
                "Employment": sample.Employment,
                "Education":  sample.Education,
                "Marital":    sample.Marital,
                "Occupation": sample.Occupation,
                "Income":     sample.Income,
                "Gender":     sample.Gender,
                "Deductions": sample.Deductions,
                "Hours":      sample.Hours,
                "TARGET_Adjusted": expected,
            }
            writer.writerow(row)
            print(f"  [{expected}] P(1)={p1:.3f}  {description}")

    print(f"\n  wrote {csv_path.relative_to(REPO_ROOT)}")
    print(f"  (model zip already at test/data/model/AuditBinaryReg.zip)")

if __name__ == "__main__":
    print(f"AuditBinaryReg  [binary logistic regression, softmax]")
    print(f"{'=' * 60}")
    write_fixture()
