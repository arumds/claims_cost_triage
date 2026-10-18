# Claims cost reduction – review triage model

## Business problem → data science problem
Only ~23% of quotes are manually reviewed. The goal is to send the **right** quotes to review:
score every incoming contractor quote by *expected rejected value* and route the top of the
ranking to a reviewer, auto-approving the rest.

A transparent rule first, two models on top:

| Layer | Question | Data | Method |
|---|---|---|---|
| **Primary – rule** | Is the bill far above the inspector estimate? | every quote with an estimate | review if bill > 1.25 × estimate (`RULE_K` in `config.py`) |
| **A – Fair-cost model** | What should this repair cost, given what the inspection saw? | one quote per approved claim (final quote preferred) | Quantile gradient boosting (p50 and p90) on log cost |
| **B – Rejection model** | How likely is a reviewer to reject this quote? | manually reviewed quotes only (70 rejections) | L2-regularised logistic regression |

`priority_eur = p_reject × quoted amount` ranks quotes for review. Quotes above the fair-cost p90 are flagged too.
The rule and the models are compared head to head on the same reviewed quotes (`src/triage.py`).

## Setup (Python 3.10+)
macOS / Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run
```bash
python run_pipeline.py                       # reads data/tech_data_2026.xlsx
python run_pipeline.py path/to/tech_data_2026.xlsx
jupyter lab notebooks/01_data_exploration.ipynb   # choose kernel "Python (claims-ml)"
streamlit run app.py                              # interactive demo app
```
The pipeline runs in under a minute and writes everything to `outputs/`. The notebook reads
`data/tech_data_2026.xlsx` by default; set the `CLAIMS_DATA` env variable to point elsewhere.
Run `deactivate` to leave the venv.

## Project layout
```
notebooks/01_data_exploration.ipynb   step-by-step EDA with findings (run before the pipeline)
app.py            Streamlit demo: problem, data issues, explorer, model results, quote checker, review queue, verdict
src/config.py     paths, feature lists, leakage list
src/data.py       load, data-quality report, cleaning, collapse to one row per quote / per claim
src/features.py   quote vs estimate features, feature matrix
src/eda.py        figures 01–04 + EDA tables
src/models.py     Model A (fair cost) and Model B (rejection), grouped out-of-fold CV
src/evaluate.py   capture curves, review-budget table, figures 05 and 07
src/triage.py     primary rule (bill > k × estimate), rule table, head-to-head with the fair-cost model
run_pipeline.py   end-to-end run
```

## Demo app
`streamlit run app.py` opens a browser app with seven pages:
1. **Problem & approach**: the business → data science framing and headline numbers
2. **Data & limitations**: review rate by quote/estimate gap, selection bias, data-quality report
3. **Explore the data**: interactive estimate vs quote scatter and inspection-org comparison, with filters
4. **Model results**: model metrics, fair-cost feature importance, quote vs fair price gap, adjustable review-budget curve
5. **Quote checker**: enter inspection details and a quote, get the fair-cost range, P(reject) and a review/approve recommendation (includes the claim 2 example)
6. **Review queue**: all quotes ranked by € at risk for a chosen review capacity, compared with today's reviews, downloadable as CSV
7. **Verdict & next steps**: where ML helps and where it doesn't (yet), and approaches to explore with more time

The app reads `data/tech_data_2026.xlsx` (or `CLAIMS_DATA`); if no file is found it offers an upload button.
Models train once on first load (about 1 minute) and are then cached.

## Key modelling decisions
- **Unit = quote.** Rows sharing a quote_id have the same contractor cost, estimate, decision and
  review flag, so they are collapsed to one row. About 12% of quotes have rows with conflicting
  categories (e.g. House vs Commercial for the same building); each category is resolved by majority
  vote (ties -> first row; documented in `data.py`). The fair-cost model uses only the raw
  inspection-time columns (`CATEGORICAL` + `NUMERIC_PRE_QUOTE` in `config.py`).
- **Fair-cost model on one quote per claim.** A claim's quotes are adjusted over time, so they are
  revisions of the same repair. Model A is trained on one quote per claim, cleaned as in
  `notebooks/01_data_exploration.ipynb`: the final quote if there is one, otherwise the latest
  preliminary quote (`to_claim_level` in `data.py`). Every quote is still scored by the fold model
  that did not see its claim, so the rejection model (Model B, all quotes) gets honest fair-cost inputs.
- **Grouped CV by claim_id.** Quotes from the same claim never end up in both train and test.
- **No leakage.** `insurance_company_cost`, `policyholder_cost`, `decision_on_quote` and
  `manual_review` are produced after the decision and are never used as features.
- **Rare inspection orgs** (<10 quotes) are grouped into `OTHER`.
- **Evaluation in €.** Share of rejected € caught at a given review budget, compared with
  the simple rule "review the largest quotes first".

## Results (out-of-fold, from this dataset)
- **Model A** (2,390 claims, one quote each): MAE €22k vs €37k for "quote = inspector estimate";
  median abs. % error 24% vs 47%. The p90 band holds 83% of bills (target 90%).
  Main drivers (out-of-fold permutation importance, `outputs/fair_cost_importance.csv`): inspector estimate (~25%),
  damage type (~20%), insurance type (~16%), quote stage (~16%), number of inspection images (~15%),
  inspection org (~5%). Building attributes add almost nothing.
- **Model B** (coefficients in `outputs/rejection_coefficients.csv`): ROC-AUC 0.66 vs 0.65 for "largest quote first". At a 20% review budget it
  catches 79% of rejected € vs 77% for the baseline. This is a small gain, which is expected
  with only 70 positive labels.
- **Primary rule vs fair-cost model** (`outputs/rule_table.csv`, `rule_vs_fair_cost.csv`, `queue_overlap.csv`,
  figure 07; 762 reviewed quotes with an estimate, 67 rejected, out-of-fold):

  | Approach | ROC-AUC | rejected € caught @ 20% budget |
  |---|---|---|
  | primary: rule, bill ÷ estimate | 0.54 | 33% |
  | secondary: fair-cost gap, bill ÷ fair p50 | 0.53 | 38% |
  | secondary: fair-cost + P(reject) × bill | 0.68 | 80% |
  | reference: largest bill first | 0.67 | 78% |

  At k = 1.25 the rule flags 45% of quotes and catches 97% of severe overruns (> €20k over the estimate), but only
  21% of the quotes reviewers rejected; 9% of quotes have no usable estimate and can't be scored. The model beats the
  rule, but almost all of its gain comes from looking at the biggest bills first: the price gap alone, rule or fair-cost,
  does not predict rejection. At the same budget the two queues agree on 43% of quotes (Jaccard); the model's queue
  holds 89% of the € above fair p90 vs 76% for the rule (a yardstick that favours the model).
- **Auto-processed claims:** 17.5% are above the fair-cost p90, about €14M above the p50 fair cost
  (one quote per claim). Summed over all quotes it would be €22M, but that counts revised quotes of the
  same repair several times. This is a candidate saving, not a proven one: it needs an audit to confirm.
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
