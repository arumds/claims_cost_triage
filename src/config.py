"""Central configuration: paths, column groups, modelling constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_DATA_IN_REPO = ROOT / "data" / "tech_data_2026.xlsx"        # default: data/ inside the project
DATA_PATH = _DATA_IN_REPO if _DATA_IN_REPO.exists() else ROOT.parent / "tech_data_2026.xlsx"  # fallback: next to project
OUTPUT_DIR = ROOT / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"

RANDOM_STATE = 42
N_FOLDS = 5
RARE_ORG_MIN_COUNT = 10          # inspection orgs with fewer quotes are grouped into "OTHER"
COST_CAP = 300_000               # values piling up at this level look like a cap/truncation
IMAGE_CAP = 300

# Columns known BEFORE / AT the time the contractor quote arrives -> allowed as features.
CATEGORICAL = [
    "inspection_org", "inspection_type", "insurance_type",
    "building_geographical_location", "building_type", "quote_event",
]
NUMERIC_PRE_QUOTE = [
    "estimated_repair_cost", "inspection_images", "building_area",
    "construction_year", "insurance_excess_amount",
]
# Features that use the quoted amount itself (used by the rejection model only).
QUOTE_FEATURES = ["contractor_cost", "log_cost_to_estimate", "quote_minus_estimate"]

# Outcome columns: decided AFTER the quote is processed -> never use as features.
LEAKY = ["insurance_company_cost", "policyholder_cost", "decision_on_quote", "manual_review"]

# Primary triage rule: review if bill > RULE_K × inspector estimate
RULE_K = 1.25
RULE_TRIGGERS = (1.10, 1.25, 1.50, 2.00)
SEVERE_OVERRUN_EUR = 20_000      # bill at least this much above the estimate = severe overrun
