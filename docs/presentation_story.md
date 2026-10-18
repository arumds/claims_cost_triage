# Claims cost triage: from data exploration to a fair-cost model

As of 2026-10-18. Online version: https://claude.ai/code/artifact/843689c9-7da3-47bb-b721-e9bd13c75ac1

**The story in one paragraph.** We asked where claims cost can be saved. The data says the inspector's
estimate is a biased yardstick for contractor bills, and that today's manual review does not look at the
bills that run over. A fair-cost model built only from inspection data prices each claim twice as accurately
as the estimate (typical error ±24% vs ±47%). Used to rank the review queue, it beats a
simple "bill > 1.25× estimate" rule at the same effort (80% vs 33% of rejected € caught), mainly because it
looks at the biggest bills first. It finds about €14M of bills above the fair range in claims
nobody reviewed. That is a candidate saving, not a proven one. The ask is a random audit of those claims to
turn it into a measured one.

## Question and hypotheses

The business question: **can we pay less for repairs without reviewing more claims?** The lever is the
contractor bill: approve fair bills quickly, and send bills that are too high to review.

The presentation tests five hypotheses, in this order:

| # | Hypothesis | If true, it means |
| --- | --- | --- |
| H1 | The inspector's estimate is a biased benchmark for the bill: its error depends on the type and size of the claim | A flat rule like "review if bill > 1.25× estimate" flags the wrong claims |
| H2 | Today's manual review does not target bills that run over, so money above a fair price passes unchecked | There is a pool of excess cost in unreviewed claims |
| H3 | A model using only data known when the quote arrives can predict a fair bill much better than the estimate | We can build a better benchmark without waiting for new data |
| H4 | Bills far above the fair price are the ones reviewers reject | The fair-cost gap is a proven savings signal |
| H5 | The models (fair-cost + rejection risk) beat the simple rule at the same review effort | The models are worth running on top of the rule |

H1–H3 come from the data exploration and are tested by the fair-cost model. H4 is the honest check: it
decides whether the savings are proven or still need an audit. H5 compares the models with the rule a team
could run tomorrow, "review if bill > 1.25× estimate".

## The data and how it was cleaned

The raw file has 4,698 rows covering 3,699 quotes and 2,390 claims; the analysis uses **one quote per
claim**. A claim's quotes are revisions over time, so counting them all would add up the same repair several
times (€262M across all quotes vs €146M with one per claim).

| Step | Rows left | Why |
| --- | --- | --- |
| Raw file | 4,698 | Several rows per quote (coverage lines) |
| Drop exact duplicate rows | 4,672 | 26 rows were identical copies |
| One row per quote | 3,699 | Contractor cost, estimate, decision and review flag are the same on every row of a quote |
| One quote per claim | 2,390 | Keep the final quote if there is one, else the latest preliminary quote |

What is never used as a model input, and why:

- **Insurer and policyholder cost:** they split the bill, so they are derived from what we predict.
- **Decision and review flag:** they are made after the bill arrives.

Known gaps: building area is missing for 54% of claims, construction year for 13%. There are no dates, and
no record of the amount approved after review. The cleaning is in `notebooks/01_data_exploration.ipynb`
(section 3) and `to_claim_level` in `src/data.py`.

## Evidence 1: the estimate is a biased yardstick (H1)

**H1 is supported.** The estimate's error has a direction that depends on the claim, so it cannot serve as
one benchmark for every bill.

Median contractor bill ÷ inspector estimate, one quote per claim (claims with an estimate above zero):

| Group | Bill ÷ estimate | Claims |
| --- | --- | --- |
| **By damage type** | | |
| Building & property damage | 1.61× | 218 |
| Fire & natural damage | 1.34× | 442 |
| Theft, vandalism & contents | 0.98× | 59 |
| Water damage | 0.88× | 1,186 |
| Other & services | 0.77× | 211 |
| **By size of the inspector estimate** | | |
| €10–20k | 1.75× | 373 |
| €30–50k | 1.11× | 832 |
| €60–80k | 1.08× | 436 |
| €90k+ | 0.75× | 475 |

Source: `notebooks/01_data_exploration.ipynb`, section 4.

Fire and building damage come in well above the estimate, water and "other" below it. Small estimates are too
low and large ones too high; estimates are rounded to €10k, which drives part of this. Two more sources of
bias:

- **Inspection org:** medians range from 0.88× to 1.51× among the orgs with 30 or more claims.
- **Photos:** 1.33× with 0–10 photos vs 0.96× with 41 or more.

**So what:** a flat rule like "review if the bill is over 1.25× the estimate" would flag normal fire claims and
miss overpriced water claims. The benchmark has to adjust for type, size, org and photos, which is what the
fair-cost model does.

## Evidence 2: review does not look where the money is (H2)

**H2 is supported.** Reviewers see 24% of claims, but not the ones whose bills run over, and every rejection
we have comes from that reviewed group.

| Measure (one quote per claim) | Auto-processed | Manually reviewed |
| --- | --- | --- |
| Claims | 1,819 | 571 |
| Median contractor bill | €43k | €31k |
| Median insurer cost | €25k | €130k |
| Bill ÷ estimate (median) | 1.26× | 0.71× |
| Insurer cost above the bill | 2% | 85% |
| Rejected | 0 | 35 (6%) |

What this shows:

- **Review follows insurer cost, not the bill.** Reviewed claims have 5× the insurer cost but *smaller* bills.
  Insurer cost looks like a product of the review itself, one more reason it can never be a model input.
- **Overruns pass unchecked.** Auto-processed bills sit at 1.26× their estimate; reviewed ones at 0.71×.
- **Rejection labels are biased and scarce.** Auto-processed claims are never rejected because nobody checks
  them. With 70 rejected quotes in total (35 after one quote per claim), a model trained on rejections learns
  how reviewers choose, not which bills are too high.

This is why the next step is a **price benchmark learned from every bill**, not a rejection model.

## Evidence 3: what drives the bill

**The bill is driven by what was damaged, how much, and the quote stage, not by the building.** A linear model
with one 0/1 feature per bucket explains about half the variation (cross-validated R² 0.47). Gradient boosting
on the same buckets does no better, so the effects are mostly additive.

Effect on the contractor bill, all else equal (2,374 claims; 90% intervals exclude 0):

| Driver (share of explained variation) | Bucket | Effect on the bill |
| --- | --- | --- |
| Damage and insurance type (33%) | Fire & natural damage (vs water) | +67% |
| | Natural & Environmental cover (vs water) | +48% |
| | Building & property damage (vs water) | +43% |
| | Theft, vandalism & contents (vs water) | +31% |
| Photo count, i.e. damage size (16%) | 41+ photos (vs 0–10) | +89% |
| | 21–40 photos (vs 0–10) | +33% |
| Quote stage (14%) | Final quote (vs preliminary) | −34% |
| Inspection org (2%) | Org C20B65008C5 (vs other orgs) | +32% |
| | Org C4B4BBADF30 (vs other orgs) | −34% |
| Location (1%) | Rural (vs urban) | +25% |

No real effect: building age, building area, building type, policy excess. Source:
`notebooks/02_cost_drivers.ipynb`.

Photo count measures damage size, not a cause: more photos do not make a repair dearer. The inspector estimate
is left out of this model on purpose, so the other drivers can show. It returns in the fair-cost model, where
it is the strongest single input.

## The fair-cost model (H3)

**The fair-cost model answers one question per claim: what should this repair cost, given what the inspection
saw?** It never sees the quoted amount, so the gap between bill and fair price is the signal.

| Design choice | What we did | Why |
| --- | --- | --- |
| Data | One quote per claim, 2,390 claims | Quotes are revisions; one per claim avoids counting a repair twice |
| Training rows | Approved claims only | Rejected bills should not teach the model what "fair" is |
| Inputs | Damage type, insurance type, quote stage, inspection org, location, building type; estimate, photos, area, construction year, excess | Everything known when the quote arrives |
| Left out | Insurer and policyholder cost, decision, review flag | Decided after the bill: leakage |
| Algorithm | Quantile gradient boosting on log cost | Gives a fair median (p50) and a "high but plausible" band (p90) |
| Validation | 5-fold cross-validation, folds grouped by claim | Every prediction is for a claim the model never saw |

**Is the estimate leakage?** No, as far as the data can show:

- The estimate never changes across a claim's quotes (0 of 781 multi-quote claims), so it is not updated after
  the bill.
- It does not track the bill closely: rank correlation 0.50, and only 9% of bills land within ±10% of it.
- The timing is inferred, not recorded (there are no dates). This is a question to confirm with the business.

**Does one quote per claim lose accuracy?** No. Trained on all 3,611 approved quotes or on 2,345 approved
claims, the model scores the same on held-out claims (typical error ±23% vs ±22%). All quotes of a claim share
one inspection, so the earlier quotes add little.

## Results

**H3 is supported: the fair-cost model halves the error of the inspector's estimate.** All figures are
out-of-fold, on 2,390 claims.

| Benchmark for the bill | Typical error | Mean error (MAE) |
| --- | --- | --- |
| Inspector estimate as is | ±47% | €36.9k |
| Cost-driver model (buckets, no estimate) | ±35% | €25.9k |
| **Fair-cost model** | **±24%** | **€21.9k** |

The estimate is the strongest input; building details add almost nothing. Share of the model's accuracy lost
when each input is shuffled (`outputs/fair_cost_importance.csv`):

| Input | Share of importance |
| --- | --- |
| Inspector estimate | 25% |
| Damage type | 20% |
| Insurance type | 16% |
| Quote stage | 16% |
| Photo count | 15% |
| Inspection org | 5% |
| Policy excess | 2% |
| Building attributes (age, type, location, area) | <1% |

What the benchmark finds in today's data:

- **47% of claims** are priced close to fair (0.8–1.25×): candidates for fast approval.
- **17.5% of auto-processed claims** are above the fair band (p90), with **€14.3M** above the fair median.
- **146 of the 204 claims** priced over 2× their fair cost went through without review, about **€10M** above
  fair.

One calibration note: 83% of bills fall inside the p90 band, not the targeted 90%, so the band is slightly
narrow.

## Primary rule vs the models (H5)

**H5 is supported, with a caveat.** The primary rule reviews a quote when the bill is more than 1.25× the
inspector's estimate. The secondary ranks quotes by the fair-cost and rejection models: P(reject) × bill. Both
rank the same 762 reviewed quotes that have an estimate (67 rejected, worth €9.0M), out-of-fold
(`outputs/rule_vs_fair_cost.csv`).

| Reviewing the top 20% (153 quotes) | Primary rule | Secondary model |
| --- | --- | --- |
| Rejected € caught | 33% | **80%** |
| Rejected quotes caught | 18 of 67 | **29 of 67** |
| Median bill in the queue | €78k | €247k |
| ROC-AUC | 0.54 | 0.68 |

- **Why the model wins:** rejected quotes are big bills (median €94k vs €34k for approved ones), but they are
  not unusually far above their estimate (0.85× vs 0.74×). The rule ranks by the ratio and fills its queue with
  mid-size overruns. The model ranks by € at risk and picks the big bills, where the rejected € sit.
- **The caveat:** "largest bill first" alone catches 78%. The gain comes from bill size, not from the price
  gap, which matches H4.
- **The rule on its own:** at k = 1.25 it flags 45% of quotes and catches 97% of bills more than €20k above
  their estimate, but only 21% of the quotes reviewers rejected. It cannot score the 9% of quotes with no
  usable estimate.

## Verdict on the hypotheses

**Four of five hold; H4 is why the savings are a candidate, not a proven number.**

| # | Hypothesis | Verdict | Key evidence |
| --- | --- | --- | --- |
| H1 | The estimate is a biased benchmark | Supported | Bill ÷ estimate runs from 0.75× to 1.75× by damage type and estimate size |
| H2 | Review misses bills that run over | Supported | Auto-processed bills 1.26× estimate vs 0.71× for reviewed; review follows insurer cost |
| H3 | A fair-cost model beats the estimate | Supported | Typical error ±24% vs ±47%; MAE €21.9k vs €36.9k |
| H4 | Bills above fair cost are the ones reviewers reject | Not supported | As a rejection score, the fair-cost gap reaches ROC-AUC 0.52, close to a coin flip |
| H5 | The models beat the simple rule | Supported | 80% vs 33% of rejected € at a 20% budget; "largest bill first" alone gets 78% |

What the data cannot prove yet:

- **Savings:** there is no record of the amount approved after review, so €14.3M above fair is a pool to
  examine, not money recovered.
- **Why H4 fails:** either rejections are driven by coverage or policy rather than price, or reviewers do not
  check price. The current labels cannot tell these apart.
- **The benchmark is relative:** if a whole segment is overpriced, the model learns that as normal.
- **Timing:** without dates, the order of quotes and the timing of the estimate are inferred.

The one direct sign of savings: when a quote was rejected and revised, the revised quote was a median 13% lower
(€575k across 36 claims).

## Recommendation and next steps

**Run the rule now, rank with the models on top, and prove the value with a random audit before counting any
savings.**

1. **Route by price, not by insurer cost, in layers.** Start with the transparent rule (bill > 1.25×
   estimate) as the floor. Rank the review queue by the models (P(reject) × bill), and send claims above the
   p90 band to review. Fast-approve claims close to fair (0.8–1.25×).
2. **Run a random audit** of about 5% of auto-approved claims above the band. This gives unbiased labels and a
   measured saving per flagged euro, which turns €14.3M from candidate to measured.
3. **Start recording two fields:** the date of each quote and estimate, and the amount approved after review.
   Both are needed to measure savings and check drift over time.
4. **Judge success in euros:** € caught per review, compared with the primary rule and with "largest bill
   first", which is already a strong baseline.
5. **Use the cost drivers for the narrative and partner feedback:** damage type, photos and quote stage
   explain the bill. Inspection org is a small lever, at most about €0.7M.

Questions to confirm with the business:

- [ ] Is the estimate always recorded before the contractor quote arrives?
- [ ] Is `quote_id` chronological? Why do some final quotes come before preliminary ones?
- [ ] Are quotes with different damage types in one claim revisions, or separate repairs?

## Slide outline

Eleven slides, about 15 minutes. Each headline states the finding, so the deck reads as the story on its own.

| # | Headline | Visual | Talking point |
| --- | --- | --- | --- |
| 1 | We can pay less for repairs by checking the right bills, not more bills | Title + the one-paragraph story | The ask: a random audit |
| 2 | The question and five hypotheses | Hypothesis table (H1–H5) | We test each one with data |
| 3 | 4,698 rows become 2,390 claims, one quote each | Cleaning steps table | Quotes are revisions; counting all of them doubles the money |
| 4 | The estimate is a biased yardstick | Bill ÷ estimate chart | Fire and small estimates run high, water and large estimates low |
| 5 | Review follows insurer cost, not the bill | Auto vs reviewed table | Overruns pass unchecked; rejection labels are biased |
| 6 | Damage type, damage size and quote stage drive the bill | Cost-driver effects chart | Building details do not matter |
| 7 | A fair-cost model built from inspection data | Design table | Only data known when the quote arrives; no leakage |
| 8 | It halves the error of the estimate | Accuracy table + importance chart | ±24% vs ±47% |
| 9 | The models catch 80% of rejected € vs 33% for the rule, at the same effort | Head-to-head curve + queue scatter | The win comes from looking at big bills first (H5) |
| 10 | €14M sits above the fair band in unreviewed claims, but it is not proven | Verdict table (H4 not supported) | Candidate saving; why we need the audit |
| 11 | Rule now, models on top, audit to prove it | Next steps 1–5 | Two data fields to start recording |

Backup slides: leakage check on the estimate, one quote per claim vs all quotes, rule thresholds k = 1.1–2.0,
rejection model vs "largest bill first".

## Questions to expect

| Question | Prepared answer |
| --- | --- |
| Isn't using the inspector's estimate leakage? | No. It is fixed before the quote and never changes across a claim's quotes (0 of 781). Only 9% of bills land within ±10% of it. The timing still needs confirming, since there are no dates. |
| Why one quote per claim and not every quote? | Quotes are revisions of the same repair. Summing them counts the money twice (€262M vs €146M). Accuracy is the same either way (±22% vs ±23%). |
| Why not just use the rule? | It is the floor, and it is transparent. But it catches only 33% of rejected € at a 20% budget (models: 80%), it ignores bill size, and it cannot score the 9% of quotes without an estimate. |
| Why not just predict rejections? | Only 70 rejected quotes, all from a biased review sample. The rejection model barely beats "largest bill first" (ROC-AUC 0.66 vs 0.65). |
| Is the €14M a saving? | No, it is the amount above the fair median in unreviewed claims. Only an audit can say how much of it is recoverable. |
| Why doesn't the fair-cost gap predict rejection? | Rejections may be about coverage or policy, not price, or reviewers may not check price. Today's labels cannot tell these apart, which is why the audit matters. |
| How accurate is the model on a single claim? | Typical error ±24%. Its p90 band holds 83% of bills, so it flags slightly too many; widening it is a simple fix. |
| Would a more complex model help? | Not much on this data. Gradient boosting and a linear model score the same on the same features; the gain comes from the estimate, not the algorithm. |
| Which inspection orgs should we talk to? | One org (C20B65008C5) runs about 32% above comparable claims. Org explains only 2% of cost, so this is partner feedback, not the main lever. |
| What does it take to go live? | The model runs today on existing fields (`run_pipeline.py`, app). To prove value: the audit plus two new fields, dates and approved amount. |
