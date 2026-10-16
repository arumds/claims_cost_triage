# Claims cost reduction – review triage model

## Business problem → data science problem
Only ~23% of quotes are manually reviewed. The goal is to send the **right** quotes to review:
score every incoming contractor quote by *expected rejected value* and route the top of the
ranking to a reviewer, auto-approving the rest.

Two models:

| Model | Question | Data | Method |
|---|---|---|---|
| **A – Fair-cost model** | What should this repair cost, given what the inspection saw? | all approved quotes | Quantile gradient boosting (p50 and p90) on log cost |
| **B – Rejection model** | How likely is a reviewer to reject this quote? | manually reviewed quotes only (70 rejections) | L2-regularised logistic regression |

`priority_eur = p_reject × quoted amount` ranks quotes for review. Quotes above the fair-cost p90 are flagged too.

## Setup (Python 3.10+)
macOS / Linux:
```bash
cd "/path/to/If-task/claims_ml"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name claims-ml --display-name "Python (claims-ml)"
```

## Run
```bash
python run_pipeline.py                       # reads data/tech_data_2026.xlsx
python run_pipeline.py path/to/tech_data_2026.xlsx
jupyter lab notebooks/01_data_exploration.ipynb   # choose kernel "Python (claims-ml)"
```
The pipeline runs in under a minute and writes everything to `outputs/`. The notebook reads
`data/tech_data_2026.xlsx` by default; set the `CLAIMS_DATA` env variable to point elsewhere.
Run `deactivate` to leave the venv.

## Project layout
```
notebooks/01_data_exploration.ipynb   step-by-step EDA with findings (run before the pipeline)
src/config.py     paths, feature lists, leakage list
src/data.py       load, data-quality report, cleaning, collapse to one row per quote
src/features.py   quote vs estimate features, feature matrix
src/eda.py        figures 01–04 + EDA tables
src/models.py     Model A (fair cost) and Model B (rejection), grouped out-of-fold CV
src/evaluate.py   capture curves, review-budget table, figure 05
run_pipeline.py   end-to-end run
```

## Key modelling decisions
- **Unit = quote.** Rows sharing a quote_id are coverage lines with identical contractor cost,
  decision and review flag, so they are collapsed (assumption, documented in `data.py`).
- **Grouped CV by claim_id.** Quotes from the same claim never end up in both train and test.
- **No leakage.** `insurance_company_cost`, `policyholder_cost`, `decision_on_quote` and
  `manual_review` are produced after the decision and are never used as features.
- **Rare inspection orgs** (<10 quotes) are grouped into `OTHER`.
- **Evaluation in €.** Share of rejected € caught at a given review budget, compared with
  the simple rule "review the largest quotes first".

## Results (out-of-fold, from this dataset)
- **Model A:** MAE €22k vs €41k for "quote = inspector estimate"; median abs. % error 24% vs 45%.
  Main drivers (out-of-fold permutation importance, `outputs/fair_cost_importance.csv`): insurance type (~31%),
  damage type (~24%), inspector estimate (~17%), number of inspection images (~11%), quote stage (~8%),
  inspection org (~5%). Building attributes add almost nothing.
- **Model B** (coefficients in `outputs/rejection_coefficients.csv`): ROC-AUC 0.68 vs 0.65 for "largest quote first". At a 20% review budget it
  catches 79% of rejected € vs 77% for the baseline. This is a small gain, which is expected
  with only 70 positive labels.
- **Auto-processed quotes:** 17% are above the fair-cost p90. That's about €23M of quoted
  amount above the p50 fair cost. This is a candidate saving, not a proven one: it needs an audit to confirm.
- **Inspection orgs differ a lot.** The median quote/estimate ratio ranges from 0.46 to 1.82 across orgs.

## Limitations
- Rejection labels exist **only** for reviewed quotes, and review targets large claims.
  The rejection model is therefore trained on a biased sample, and its scores on
  auto-processed quotes are extrapolation.
- There is no label for "how much was saved". Rejected/approved is a weak proxy.
- Inspector estimates are bucketed to €10k (240 zeros, 3.5% missing). Building area is 50% missing.
- Costs look capped at €300k. There are no dates, so time drift and time-based validation can't be checked.
- Being overpriced relative to the fair cost does **not** predict rejection in this data
  (negative coefficient). Rejections may be driven by coverage or policy reasons more than price.

## Next steps
1. **Random audit** of ~5% of auto-approved quotes to get unbiased labels. This is the main recommendation.
2. Record the **approved amount after review** so savings can be modelled as a regression target.
3. Add claim/quote dates, quote line items and inspection images (CV model on photos).
4. Use inspection-org level feedback as a lever separate from claim triage.
5. Calibrate p90 (coverage is currently 83%) with conformal prediction. Monitor drift and contractor behaviour after deployment.
