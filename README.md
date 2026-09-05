# targetleak

**Did your model learn, or did you leak?**

`targetleak` reads a training set and tells you which columns are secretly
carrying the answer — before you ship a model whose test score was never real.

```bash
pip install targetleak
targetleak --demo
```

```
CRITICAL target-proxy [refund_amount]
         alone reaches AUC 1.0000. One column should not nearly solve the
         target - this is very likely computed from the answer, or recorded
         after it was known.
         FIX: Establish when this column receives its value. If it is
         written at or after the moment the target becomes known, it cannot
         be an input. Drop it, or rebuild it from data available strictly
         before the prediction cutoff.

VERDICT: 3 critical leak(s). Do not trust this model's test score until they
are resolved.
```

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

## What it checks

| Check | Catches |
|---|---|
| `target-proxy` | One column that nearly solves the target on its own |
| `pure-categories` | Categories that partition the target exactly |
| `suspicious-name` | Columns *named* like labels or forward-looking values |
| `identifier-like` | IDs the model will memorise instead of learning |
| `suspiciously-predictive` | Strong enough to be worth confirming |
| `train-test-contamination` | Rows whose features appear on both sides of your split |
| `group-overlap` | The same entity in train and test |
| `temporal-column` | Dates, where a random split trains on the future |
| `duplicate-rows` | Repeats that inflate the test score |
| `target-mostly-null` | A training set far smaller than the row count suggests |
| `constant` | Features carrying no information on your labelled rows |

Binary and continuous targets. CSV, TSV, and Parquet.

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

## A real find

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
