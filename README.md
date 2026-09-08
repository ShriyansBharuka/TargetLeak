# targetleak

**Did your model learn, or did you leak?**

`targetleak` reads a training set and tells you which columns are secretly
carrying the answer — before you ship a model whose test score was never real.

```bash
pip install targetleak
targetleak --demo
```

> For `.parquet` files: `pip install "targetleak[parquet]"`

## Status: early, and here is exactly how early

**0.3.0. Weeks old. Swept across ~100 real public datasets and run against one
production dataset besides its own test suite.** Read the findings and apply
your own judgement; that is a safe and useful way to use it today. What I would
not do yet is wire it into a shared CI pipeline as a blocking gate before you
have run it by hand a few times and built up the `--ignore` list, because a
false positive that blocks a colleague's pull request on day one is how a check
gets deleted.

Three things worth knowing before you rely on it:

- **The first four real datasets it met each exposed a bug the synthetic tests
  missed** - pandas 3 string dtypes, NaN targets being read as the negative
  class, an integer-cardinality cliff, and a leak covering only 121 rows. A
  later sweep across ~100 more found twelve further problems, all of them
  fixed or documented in [`SWEEP_FINDINGS.md`](SWEEP_FINDINGS.md). Yours may
  still be the one that breaks it, but it is no longer the first hard one.
- **Roughly 0.6% of clean columns get flagged** (2 of 358 in the benchmark
  below). Expect a little noise and expect to suppress the odd column.
- **A clean report is not proof of absence.** It cannot tell a leak from a
  genuinely easy problem - see the benchmark, where that limitation is
  measured rather than hidden. What it *can* tell you is measured too: 98% of
  planted leaks found across four families and three split shapes.

The most useful thing you can send is a dataset shape that crashes it or
produces an obviously wrong finding.
[Open an issue](https://github.com/ShriyansBharuka/TargetLeak/issues) - you do
not need to share the data, just the column types, the rough shape, and what
it said.

```
CRITICAL target-proxy [cancellation_reason]
         alone separates the target at 1.0000 (true AUC 1.0000), 29.9 SE
         above chance. One column should not nearly solve the target - this
         is very likely computed from the answer, or recorded after it was
         known.
         EVIDENCE: 'not_given': 0% positive (n=568) | 'moved': 100% positive
         (n=302) | 'price': 100% positive (n=330)
         FIX: Establish when this column receives its value. If it is
         written at or after the moment the target becomes known, it cannot
         be an input. Drop it, or rebuild it from data available strictly
         before the prediction cutoff.

VERDICT: 3 critical leak(s). Do not trust this model's test score until they
are resolved.
```

Every finding carries the measurement it rests on, how far that stands from
chance, and what to do about it.

## The idea

A model needs *many* features to reach AUC 0.95. A leak gets there with **one**.

So any lone column that nearly solves the target is the prime suspect — it is
usually a proxy for the answer that will not exist at prediction time. That
single check finds most real leaks, and it needs no model, no training run, and
no labels beyond the ones you already have.

Every finding comes with what to do about it. Naming a leak without saying how
to fix it is a scolding, not a tool.

## Usage

```bash
targetleak data.csv  --target churned
targetleak data.parquet --target y --split is_test --group user_id
targetleak data.csv  --target y --json          # for CI
targetleak data.csv  --target y --no-fixes      # findings only
```

As a library:

```python
import targetleak

findings = targetleak.analyse(df, target="churned")
print(targetleak.report(findings))

if any(f.severity == "critical" for f in findings):
    raise SystemExit("refusing to train on a leaking dataset")
```

Exit code is `1` when anything critical is found, so this works as a gate:

```bash
targetleak data.csv --target y && python train.py
```

### In CI

```yaml
- run: pip install targetleak
- run: targetleak data/train.csv --target churned
```

A checker with no way to accept a finding can never go green, so it gets
deleted. Reviewed a column and decided it is fine? Name it:

```bash
targetleak data.csv --target y --ignore customer_tier,promo_code
```

Ignored columns are **still listed**, at info level. A silent suppression is
how the next real regression gets missed.

There is a ready-made workflow in
[`.github/workflows/leak-check.yml`](.github/workflows/leak-check.yml) that
comments the findings on the pull request and attaches the HTML report. Your
data never leaves your runner; only the findings reach the comment.

## What it checks

| Check | Catches |
|---|---|
| `target-proxy` | One column that nearly solves the target on its own |
| `pure-categories` | A repeated value whose rows all carry one target class |
| `suspicious-name` | Columns *named* like labels or forward-looking values |
| `identifier-like` | IDs the model will memorise instead of learning |
| `suspiciously-predictive` | Strong enough to be worth confirming |
| `train-test-contamination` | Rows whose features appear on both sides of your split |
| `group-overlap` | The same entity in train and test |
| `temporal-column` | Dates, where a random split trains on the future |
| `duplicate-rows` | Repeats that inflate the test score |
| `target-mostly-null` | A training set far smaller than the row count suggests |
| `constant` | Features carrying no information at all |
| `dead-on-labelled-rows` | Features that vary in the file but not on labelled rows |
| `underpowered` | Scores too high to ignore but on too little data to trust |
| `group-overlap` | An entity whose *identity* predicts the target, spanning your split |
| `widespread-separability` | So many columns are predictive that the problem is easy, not leaky |

Binary, multiclass (one-vs-rest) and continuous targets. CSV, TSV, Parquet.

Every score is also required to stand clear of the null by several standard
errors, with the bar rising as more columns are tested. Without that, 60
columns of pure noise on 20 rows produce a "critical leak" - the threshold
alone has no idea how much data it is looking at.

For a raw column that null is analytic (Hanley-McNeil). For a target-encoded
one it is **measured**, by permuting the target and re-encoding, because an
encoding built from the target is not independent of it - assuming otherwise
made a four-category noise column clear the bar 3.7% of the time against a
nominal 0.05%. The permutations run only for columns that already cleared the
score threshold, so the cost is per candidate, not per column.

## Two details that make it work

**Categoricals are scored with out-of-fold target encoding.** Encode in-fold
and *every* high-cardinality column scores ~1.0, because each category predicts
its own mean — the tool would flag every legitimate feature and be useless. A
120-city column of pure noise scores 0.50 here, as it should.

**Names are checked as well as statistics.** A 5-day forward label is future
information whether or not it correlates well with the label you are training
on. No AUC threshold can catch that; the column's name gives it away. Found on
real data where seven `label_*` columns sat in the feature matrix scoring only
0.60–0.76 — invisible to statistics, obvious from their names.

## Benchmark

Synthetic tests where the same author writes both the leak and the detector
prove nothing. This runs against data nobody here constructed:

```bash
python benchmark/run_benchmark.py
```

| dataset | rows | cols | known leaks | found | false positives |
|---|---:|---:|---|---|---:|
| titanic | 1,309 | 13 | `boat`, `body` | `boat`, `body` | 0 |
| credit-g | 1,000 | 20 | - | - | 0 |
| adult | 48,842 | 14 | - | - | 0 |
| kc1 | 2,109 | 21 | - | - | 0 |
| bank-marketing | 45,211 | 16 | - | - | 0 |
| Australian | 690 | 14 | - | - | 0 |
| dresses-sales | 500 | 12 | - | - | 0 |
| SpeedDating | 8,378 | 120 | - | - | 2 |
| churn | 5,000 | 20 | - | - | 0 |
| iris | 150 | 4 | - | - | 0 |
| wine | 178 | 13 | - | - | 0 |
| breast_cancer | 569 | 30 | - | - | 0 |
| digits | 1,797 | 64 | - | - | 0 |
| diabetes | 442 | 10 | - | - | 0 |

**Recall on documented leaks: 2/2. False positives on clean data: 2 across 358
columns in 13 datasets (0.6%).**

Two documented leaks is too few to say anything about which *kinds* of leak get
caught, and publicly documented tabular leaks with downloadable data are
scarce. So there is a second benchmark that plants them instead — the data
stays real, only the leak is injected, which gives ground truth by
construction:

```bash
python benchmark/recall.py
```

| planted leak | full strength | covering 70% | 40% | 20% |
|---|---|---|---|---|
| target proxy computed from the answer | 6/6 | 6/6 | 6/6 | 6/6 |
| reason code filled in for one class | 6/6 | 6/6 | 6/6 | 6/6 |
| measurement never taken when it happened | 6/6 | 6/6 | 6/6 | 6/6 |
| the answer plus noise | 6/6 | 6/6 | 6/6 | 3/6 |

The leaks that live in the split rather than in a column get the same
treatment. These only run when you pass `--split`, so no amount of sweeping
unlabelled data exercises them:

| planted leak | | | |
|---|---|---|---|
| rows copied from train into test | 6/6 at 100% of test rows | 6/6 at 50% | 6/6 at 10% |
| an entity whose identity predicts the target | 6/6 | 6/6 at 60% | |
| a real date column under a random split | 6/6 | | |

**129 of 132 planted leaks found (98%) across six real datasets**, with the
three misses all in the weakest row — a column separating the target at about
0.72, which is genuinely near the edge of what one column can show.

Recall on its own would reward a check that fires on everything, so each split
family has an innocent twin that has to stay quiet: an entity id spanning the
split whose identity says nothing, and a split with no shared rows. **0 false
alarms on those.** That guard is not hypothetical — `group-overlap` used to
fire on 280+ columns of KDD98 by asking whether values straddled the split, a
quantity fixed by rows-per-value that carries no information at all, and a
recall-only benchmark would have called that version perfect.

Reported per family and per strength on purpose, because one recall percentage
hides a whole family being invisible — which is exactly what this found on its
first run, twice. `reason code` was 0/6 below full strength in all six
datasets, because the check meant to catch it was gated behind the AUC
threshold whose dilution it existed to see past. Then `target proxy` was 0/6
below 70%, because a numeric column with pure sub-populations is the same leak
as a pure category and only the categorical half was ever checked.

Both gates now come from the data rather than the dtype: a group of identical
values must be improbable under the base rate **and** cover at least 5% of the
rows. The coverage floor is not decoration — without it the check fires on
4,283 of riccardo's 4,296 quantised features, where 37 rows sharing a float
value land on one class at p = 5e-23 and mean nothing at all.

Eight of those are ordinary supervised-learning sets in wide use with no
leakage anyone has reported, and unlike iris and wine they carry the mess of
real data: pandas `category` columns, heavy missingness, a 121-column frame,
imbalanced targets. Zero false positives across them is the number that
matters most here.

The two on SpeedDating are the name rule misfiring: with the target called
`match`, `expected_num_matches` looks like a sibling of it, when it is really
a survey answer collected *before* the event. Counted, not tuned away.

Titanic's `boat` and `body` are the textbook leakage example — a lifeboat
number exists only for people who got into a lifeboat, a body-recovery number
only for people who did not survive. The other five ship with scikit-learn and
are among the most-studied datasets in the field; if they leaked, it would be
famous. Every critical finding there is counted against the tool.

iris and wine used to contribute five of these, and the fix was not to tune a
threshold. On iris, `petal length` separates setosa perfectly — but that is
*one class of three*, and averaged across all three it separates the target far
less well. A leak gives away the whole target; a good feature gives away one
class. Severity on a multiclass target now follows the macro-averaged
one-vs-rest score rather than the best single class, so those become warnings
that say so. The same change took a 100-species leaf dataset from 32 critical
findings to zero, and every one of those 32 was true and none was a leak.

**The two that remain are counted, not explained away.** On SpeedDating the
name rule reads `expected_num_matches` as a sibling of the target `match`,
when it is really a survey answer collected before the event.

The benchmark also earned its place immediately: it caught a miss on `body`.
That column identifies only 121 of 1,309 passengers, so its missingness AUC is
0.575 and the ranking check walked straight past it — even though every one of
those 121 died. A column can give the answer away on a subset of rows while
looking like noise overall, and that check now exists because a dataset we did
not write exposed its absence.

## A real find in a dataset nobody warned us about

Pointed at UCI's `cylinder-bands` (540 rows, printing-press defects), it
reported seven critical findings, which looked at first like a false-positive
cluster. It was not:

| column | rows missing | target on those rows | that class overall | P under the null |
|---|---:|---|---:|---:|
| `solvent_type` | 55 | **all** `band` | 42.2% | 2.5e-21 |
| `varnish_pct` | 56 | **all** `band` | 42.2% | 1.1e-21 |
| `ink_pct` | 56 | **all** `band` | 42.2% | 1.1e-21 |
| `ESA_Amperage` | 55 | **all** `band` | 42.2% | 2.5e-21 |

Seven process measurements share the same ~55 missing rows, and every one of
those rows is a defective band, in a dataset where defects are 42% of the
total. The plain reading is that when a defect occurred the run was abandoned
and the measurements were never taken, so the *absence* of a reading carries
the outcome. Train on this with a random split and impute the gaps and you
will get a model that looks good and knows nothing.

Whether that is a leak or a legitimate signal depends on when those
measurements are recorded relative to the defect being known - which a domain
expert can answer and this tool cannot. That is the honest shape of every
finding here: it points, you decide.

## A real find

One dataset is not a validation set, and this one is the author's own project
rather than an independent trial - so read it as a worked example, not a
benchmark. It is here because the bugs were real and nobody had planted them.

Run against a 448,000-row × 69-column production training set for a live
trading system, `targetleak` reported that the target column was 79% null and
that 17 features were constant on the rows that actually carried labels —
including an entire insider-trading feature family fed by its own weekly
ingestion pipeline.

Those features varied normally across the file: 72,668 distinct values for one
of them. But of the 72,828 rows where it was non-zero, **zero** carried a label.
The label column and the enriched features covered disjoint time periods. The
model had never seen a single non-zero value from that pipeline, and its owner
had been investigating why the model showed no edge.

That is the class of bug this finds: not a mistake in the modelling, a mistake
in what reached the model.

## What it does not do

- It does not detect preprocessing leakage (a scaler fit before the split).
  That lives in your code, not your data — read your pipeline.
- A `suspiciously-predictive` finding is not proof. Some features really are
  that good. It tells you where to look.
- A discrete predictor against a continuous target is scored by how well the
  target separates each of its groups, not by correlation. |Spearman| for a
  two-group predictor is capped at 0.866 at an even split and collapses from
  there, so a flag identifying the top 2% of a revenue target used to score
  0.62 and read as noise.
- The power gate is a Bonferroni-flavoured approximation, not an exact test.
  It is there to stop small samples producing confident nonsense, and it will
  occasionally hold back a real finding on a small dataset - which is reported
  as `underpowered` rather than hidden.
- It cannot know your business. A column that is legitimate at prediction time
  in one system is a leak in another. You decide; it points.

## Development

```bash
git clone https://github.com/ShriyansBharuka/TargetLeak
cd TargetLeak
pip install -e ".[dev]"
pytest
```

## License

Apache-2.0.
