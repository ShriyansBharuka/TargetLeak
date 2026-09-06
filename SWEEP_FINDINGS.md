# Sweep findings

Produced by a ten-round sweep across ~100 real public datasets. Each finding
was reproduced by running code before being written down. The loop itself was
scoped to report only; the fixes below were made afterwards, deliberately, with
the regression cases each entry specifies.

## Status

| | finding | state |
|---|---|---|
| F1 | 26-class targets refused outright | **fixed** |
| F2 | `widespread-separability` unreachable on narrow frames (7 instances) | open |
| F3 | `audiology` refused by 0.6 of a row | **fixed** |
| F4 | numeric percentage read as 56 unordered classes | **fixed** |
| F5 | KDD98 183s — explained; the obvious optimisation is unsafe | won't fix, documented |
| F6 | `group-overlap` fires on 280+ columns where the effect is zero | open |
| F7 | ...and misses the actual entity column | open |
| F8 | `underpowered` has no remediation text | open |
| F9 | normalised floats read as discrete flags | **fixed** |
| F10 | findings changed with the file format | **fixed** |
| F11 | tank numbers read as years | **fixed** |

**F3, F4, F9, F10 and F11 were one root cause** — a decision about a column
made from how it is *stored* rather than what it *contains* — and one change
closed all five. F10's residual (`float64` behaving differently from `int64`
at the same cardinality) fell out of the same fix once `_looks_categorical`
stopped requiring an integer dtype.

**F6 and F7 must be fixed together** and are not: F6 says the check fires on
everything, F7 says it misses the entity, and they are the same mistake from
two sides. Fixing F6 alone would quieten the noise and keep the blind spot.
One of F7's two causes — the row-count term in `_looks_categorical` — is now
gone; the check itself still tests the wrong quantity.

**F2 is open on purpose.** Its proposed fix is on a third revision, each one
broken by a dataset the previous version had not seen.

---

## Round 1 — 59 datasets

`python benchmark/sweep.py` · 59 ran, 0 unreachable, **0 slow**, 1 crash,
4 flagged as noisy.

Zero slow datasets across 59 real frames is the useful negative result here:
the ~2x work and the indicator-encoding shortcut hold up outside the two files
they were tuned on.

### F1. `letter` refused outright — FIXED before the loop started

**Dataset** `letter`, 20,000 × 17, 26 classes (A–Z).

```
ValueError: target column 'class' is non-numeric with 26 distinct values.
That is neither a classification target (<= 20 classes) nor a numeric one.
```

**Root cause** `MAX_CLASSES = 20`, an arbitrary ceiling that also excluded
anything MNIST-shaped. The class count was the wrong question; rows per class
is the right one.

**Status** Fixed in `b45b...`+ — ceiling is now 100 classes with at least 10
rows per class, and the refusal message shows the arithmetic. `letter` analyses
in 6s with no critical findings. Regression test added.

---

### F2. `widespread-separability` cannot fire on a narrow frame — OPEN

**Datasets** all four "noisy, unexplained" results:

| dataset | shape | flagged | of features | share |
|---|---|---:|---:|---:|
| breast-w | 699 × 10 | 6 | 9 | 67% |
| glass | 214 × 10 | 5 | 9 | 56% |
| ecoli | 336 × 8 | 3 | 7 | 43% |
| yeast | 1,484 × 9 | 3 | 8 | 38% |

**Reproduce**

```bash
python - <<'PY'
from sklearn.datasets import fetch_openml
import targetleak as tl
b = fetch_openml("breast-w", version=1, as_frame=True, parser="pandas")
out = tl.analyse(b.frame, b.target.name)
print([f.kind for f in out if f.kind == "widespread-separability"])   # []
PY
```

**Root cause** `WIDESPREAD_MIN = 8` requires eight flagged columns in absolute
terms. A nine-feature frame can never reach it, so the note that exists to say
"this is an easy problem, not a hundred leaks" is unreachable on exactly the
datasets where the share is highest. I added that absolute floor to stop the
share alone reframing the demo frame's two genuine leaks among six columns —
so the floor fixed one false trigger by creating a blind spot.

**These are not false positives.** breast-w, glass, ecoli and yeast are
classic easy problems whose features genuinely separate the classes. Every
per-column finding is correct. What is wrong is the report: an engineer sees
six criticals on a nine-column frame and no explanation.

**Proposed fix — use the score distribution, not the count.** Measured:

| dataset | scores of flagged columns | near-perfect (≥0.99) |
|---|---|---:|
| breast-w | 0.910 0.922 0.941 0.949 0.974 0.974 | 0 of 6 |
| glass | 0.904 0.912 0.916 0.934 0.949 | 0 of 5 |
| ecoli | 0.905 0.953 0.960 | 0 of 3 |
| yeast | 0.947 0.966 0.969 | 0 of 3 |
| demo (2 real planted leaks) | **1.0000 1.0000** | **2 of 2** |

A leak is a *copy of the answer*, so it lands at or beside 1.0. Genuine signal
is strong and imperfect. That separates the two cases cleanly and it does not
depend on frame width or column count, which is what both of my previous
attempts got wrong.

Concretely: fire the note when the flagged share is high **and** the flagged
set contains no near-perfect column; suppress it when near-perfect columns are
present, because those are leak candidates however few there are. Then drop
`WIDESPREAD_MIN` entirely — it is a proxy for a thing now measured directly.

Check against the existing cases before accepting: `mfeat-factors` (99 of 216,
top score ~0.987) must still fire; the demo frame (2 of 6, both 1.0000) must
still stay quiet.

**Why this was left open** The loop is report-only, so `targetleak/` was not
edited. This needs the threshold checked against the whole benchmark before it
lands — a change to when findings get reframed can hide real leaks, and that is
the one direction this tool must not fail in.

---

## Round 2 — queued

Ten datasets added to `benchmark/sweep.py` for shapes round 1 never reached:
`Bioresponse` (1,776 columns), `Amazon_employee_access` (integer codes with
thousands of levels), `KDDCup09_appetency` (231 mostly-empty columns), `isolet`
(617 columns × 26 classes), `har`, `anneal`, `primary-tumor` and `audiology`
(22 and 24 classes on 339 and 226 rows — thin support, right at the new
rows-per-class rule), `abalone`, `cholesterol` (regression with missing
values). Results below when it completes.

---

## Round 2 — 69 datasets

`python benchmark/sweep.py` · 69 ran, 0 unreachable, **0 slow**, 1 new crash,
5 noisy.

The extreme shapes all held, which is the result worth recording:

| dataset | shape | secs | findings |
|---|---|---:|---|
| `Bioresponse` | 3,751 × **1,777** | 3.7 | none |
| `KDDCup09_appetency` | 50,000 × 231, mostly empty | 8.9 | none |
| `Amazon_employee_access` | 32,769 × 10, integer codes with thousands of levels | 0.3 | none |
| `isolet` | 7,797 × 618, 26 classes | 43.7 | widespread note fired correctly |
| `abalone` / `cholesterol` | 28 classes / continuous target | 0.6 / 0.3 | none |

A 1,777-column frame in under four seconds retires the width worry.

### F3. `audiology` refused, and the average is the wrong statistic — OPEN

**Dataset** `audiology`, 226 × 70, 24 classes.

```
ValueError: target column 'class' has 24 distinct non-numeric values across
226 labelled rows (9.4 per class). That is too thin to treat as
classification - the ceiling is 100 classes with at least 10 rows each.
```

**Root cause** `MIN_ROWS_PER_CLASS = 10`, which I added in the same commit that
fixed F1. It refuses on the *mean* rows per class, and the mean is the wrong
statistic on a skewed distribution. audiology's classes are
`57, 48, 22, 22, 20, 9, …` — the large classes have ample support and only the
tail is thin, yet the whole dataset is rejected because the average lands at
9.4 against a threshold of 10. Rejected by 0.6 of a row.

More fundamentally this is the wrong mechanism. `classes / labelled rows` is
0.106 here, nowhere near the ~1.0 that marks free text pointed at the wrong
column, so the "is this a label at all" question was already answered. Thin
per-class support is what the power gate is for: it would report those classes
as `underpowered` and say why, which is strictly more useful than refusing to
look at the file.

**Proposed fix** Refuse only on the cardinality ratio — a target whose distinct
values approach one per row is not a label. Drop `MIN_ROWS_PER_CLASS` and let
the z gate handle thin classes, since duplicating its job with a blunter
instrument is what produced this. Keep `MAX_CLASSES` as a cost ceiling, not a
correctness one.

**Note** F1 and F3 are the same mistake twice: fixing an arbitrary threshold by
adding another arbitrary threshold. Worth being suspicious of the next one.

---

### F2 — two more facets, and the proposed fix needs reshaping

**Second facet: it is not only narrow frames.** `letter` (20,000 × 17), now
that F1 lets it run, joins the noisy list at 5 flagged of 16 — scores
`0.917 0.919 0.919 0.940 0.963`, **0 near-perfect**. Sixteen columns cannot
reach a floor of eight either, so the blind spot covers mid-width frames too.

**Third facet, and the interesting one: `har`.** 10,299 × 562, 6 classes.
120 columns flagged of 561 = **21% share, below the 25% gate**, so no note
fires — despite 120 findings, which is a wall by any reading. Share is as
poor a proxy as count.

But the score distribution says something a fire/don't-fire note cannot:

```
120 flagged of 561
  near-perfect (>=0.99) : 5      <- scores exactly 1.0000
  strong but imperfect  : 115
```

`har` has **both signatures at once**: 115 columns that are strong and
imperfect (easy problem — 561 engineered sensor features over 6 activities)
and 5 that are perfect. Those 5 are plausibly physical rather than leaky, a
gravity axis separating "lying down" from every upright posture, but they are
the five an engineer should look at and the other 115 are context.

**Revised proposal** Do not gate the note on count or share at all. Report
both numbers and point at the short list:

> 120 of 561 columns are individually predictive, which is the signature of an
> easy problem rather than a leaking one. **5 of them reach 1.0000** — start
> there; treat the remaining 115 as the dataset working.

That is more useful than either reframing everything or reframing nothing, and
it degrades correctly in both directions: an easy dataset with no perfect
columns gets pure reassurance, and the demo frame's two perfect columns out of
six get no reassurance at all.

**Still open, still deliberately unfixed.** Same reason as before: this changes
when findings are de-emphasised, and getting it wrong hides real leaks. It
needs checking against `mfeat-factors` (99 of 216, top 0.987 — should reassure),
`har` (5 perfect — should point), the demo (2 of 2 perfect — should not
reassure at all), and the full benchmark before it lands.

---

## Round 3 — 79 datasets

`python benchmark/sweep.py --slow-secs 90` · 79 ran, 0 unreachable, **0 slow**,
no new crashes, no new noisy datasets. Every failure in the summary is F2 or F3
already recorded above.

Scale and width are settled:

| dataset | shape | secs | findings |
|---|---|---:|---|
| `poker-hand` | **1,025,009** × 11 | 46.4 | 1 warning |
| `covertype` | 581,012 × 55 | 53.9 | 1 critical, 1 warning (4%) |
| `arcene` | 200 × **10,001** | 12.4 | none |
| `madelon` | 2,600 × 501 | 1.1 | none |

A million rows in 46s and ten thousand columns on two hundred rows in 12s. No
further performance work is indicated by this evidence.

### F4. My F1 fix turned a numeric percentage into 56 classes — OPEN, REGRESSION

**Dataset** `cpu_act`, 8,192 × 22. Target `usr` is CPU time in user mode:
`int64`, 56 distinct values, range 0–99. A percentage.

**Reproduce**

```bash
python - <<'PY'
from sklearn.datasets import fetch_openml
import numpy as np, targetleak as tl
b = fetch_openml("cpu_act", version=1, as_frame=True, parser="pandas")
df, t = b.frame, b.target.name
print(tl._target_kind(df[t]))                     # multiclass
print(sum(f.severity == "critical" for f in tl.analyse(df, t)))   # 4

forced = df.copy()                                # nudge it off integers
forced[t] = forced[t] + np.linspace(0, 1e-9, len(forced))
print(tl._target_kind(forced[t]))                 # continuous
print(sum(f.severity == "critical" for f in tl.analyse(forced, t)))  # 0
PY
```

| treatment | critical | warning |
|---|---:|---:|
| 56-way multiclass (current) | **4** | 4 |
| continuous (correct) | **0** | 3 |

**Root cause — and it is mine, from this session.** Fixing F1 raised
`MAX_CLASSES` from 20 to 100. `_target_kind` returns `continuous` for a numeric
target only when its distinct count *exceeds* `MAX_CLASSES`, so that change
swept every numeric target with 21–100 distinct values from continuous into
multiclass. A usage percentage became fifty-six unordered classes, one-vs-rest
took the best of fifty-six tests, and four false criticals appeared. Before the
F1 fix, `cpu_act` was handled correctly.

**Proposed fix — split the ceiling by dtype.** The two cases are not the same
question:

- **Non-numeric target**: labels have no order, so multiclass up to
  `MAX_CLASSES` is right. This is what F1 needed — 26 letters, 24 audiology
  diagnoses.
- **Numeric target**: the ordering *is* information, and discarding it to run
  one-vs-rest is strictly worse. Default to continuous unless the distinct
  count is very small (~15 or fewer), where integers are plausibly category
  codes rather than a measurement.

Checked against the cases already swept: `letter` (26 string classes) stays
multiclass; `cpu_act` (56 numeric) becomes continuous; `abalone` (28 numeric
ring counts) becomes continuous, which is *better* since rings are ordinal;
`wine_quality` (7 numeric quality grades) becomes continuous, also defensible
as ordinal.

**This is the third instance of one pattern.** F1: an arbitrary class ceiling.
F3: fixed it by adding an arbitrary rows-per-class floor. F4: the same fix
widened a band it was never meant to touch. Each fix introduced the next
finding. The fix above is different in kind — it splits on a real property of
the data (does the target carry order?) instead of picking another number —
which is the only reason I would trust it more than the last two.

---

### Still untested: real datetime columns

`electricity` was chosen for its `date` column and does not have one — OpenML
ships it normalised to floats (`0.0, 0.0, 0.0, …`), so `_is_timelike` returns
False and *not* firing is correct behaviour, not a miss.

So after 79 datasets the `temporal-column` check has still never run against a
genuine date column outside frames written for the tests. That is now the
largest untested surface in the tool, and it matters because time-based
leakage is the kind practitioners hit most. Round 4 targets it specifically.

---

## Round 4 — 86 datasets (3 unreachable)

`python benchmark/sweep.py --slow-secs 90` · 1 crash (F3), 2 slow, 6 noisy.
`rainfall_bangladesh`, `Rain_in_Australia` and `okcupid-stem` do not resolve at
the versions I guessed — a wrong name in my list, not a tool problem.

### The datetime gap is CLOSED — and the check works

`nyc-taxi-green-dec-2016`, 581,835 × 15, carries two genuine datetime columns:

```
lpep_pickup_datetime   dtype=str   "'2016-12-01 00:52:41'"
lpep_dropoff_datetime  dtype=str   "'2016-12-01 00:10:39'"
```

`temporal-column` **fired on both**. Note the values arrive wrapped in literal
single quotes and `_is_timelike` still parsed them. After 86 datasets this is
the first time that check has run against a real date column outside frames
written for the tests, and it behaved.

`Bike_Sharing_Demand`, `seattlecrime6` and `avocado_sales` were also picked for
dates and have none in their OpenML form — same story as `electricity`.

### F2 — fifth instance, now on a continuous target

`diamonds`, 53,940 × 10, target `price` (11,602 distinct, continuous).
4 of 9 columns flagged = 44%, reported as noisy-unexplained.

| column | score |
|---|---:|
| `z` | 0.9786 |
| `y` | 0.9814 |
| `carat` | 0.9814 |
| `x` | 0.9816 |

**0 near-perfect.** And carat and the physical dimensions are the *cause* of a
diamond's price, so this is signal, not leakage. The F2 score-distribution
proposal therefore generalises past binary targets, which is worth knowing
before it lands.

### F5. KDD98 takes 183s — explained, and the obvious fix is UNSAFE

**Dataset** `KDD98`, 191,260 × 479. 183.4s.

**Why** 431 of 478 feature columns are non-float (300 `int64`, 132
`category`), so nearly every one needs an out-of-fold encoding — roughly 2,150
groupby passes over 191,260 rows. Cardinalities reach `ZIP` at 25,847 distinct
and `IC5` at 24,744.

The cost tracks the *encodable* column count, not the cell count: 91.6M cells
here against 31.0M in the 448k×69 file that runs in 15s, but that file is
mostly float and floats are never encoded. Note also that the integer
dual-scoring added earlier this session (the fix for the row-count cliff, where
codes 3 and 7 always meaning y=1 was caught at 12,000 rows and missed at 2,000)
is what puts all 300 `int64` columns on the expensive path. That is a real
correctness-for-speed trade, made deliberately.

**The optimisation I was going to propose, and why it is wrong.** The
highest-cardinality columns score at chance once encoded:

| column | levels | rows/level | encoded score |
|---|---:|---:|---:|
| `ZIP` | 25,847 | 7.4 | 0.5123 |
| `IC5` | 24,744 | 7.7 | 0.5036 |
| `POP901` | 11,471 | 16.7 | 0.5098 |
| `HV2` | 4,927 | 38.8 | 0.5021 |

Which looked like an easy win: skip the encoding above some cardinality, since
most levels are unseen in the training folds and collapse to the global mean
anyway. But those four columns may simply be uninformative, and that confound
decides whether skipping is safe — so I tested it on a column that definitely
does carry a leak, 190,000 rows, a quarter of the levels always meaning 1:

| levels | rows/level | encoded score | |
|---:|---:|---:|---|
| 200 | 950.0 | 1.0000 | found |
| 2,000 | 95.0 | 1.0000 | found |
| 12,000 | 15.8 | 1.0000 | found |
| **26,000** | **7.3** | **1.0000** | **found** |

Out-of-fold encoding finds a real high-cardinality leak perfectly, even at
7.3 rows per level. **Skipping those columns would silently lose exactly the
leaks the tool exists to catch.** So 183s is the price of correctness on a
479-column frame of categoricals, and the right response is to document the
ceiling, not to trade detection for speed.

**Recommendation** Leave it. State in the README that runtime scales with the
number of non-float columns rather than with rows, so a wide categorical frame
is the expensive case. If it ever needs to be faster, the honest lever is
parallelism across columns, not skipping work.

### Unresolved: `arcene` timing variance

`arcene` (200 × 10,001) took 12.4s in round 3 and 90.3s in round 4, same code,
same data. This machine has documented swings of that size — a single workload
measured 38s and 98s in one process earlier — so the likeliest explanation is
contention rather than a real regression. Flagging it as unexplained rather
than dismissing it: if it recurs on a quiet machine it is worth a profile.

---

## Round 5 — split sensitivity (recall evidence)

`python benchmark/split_sensitivity.py`. First run was useless: two of three
datasets died on bugs in the harness (a classifier fed a continuous target; a
categorical past HistGradientBoosting's 255-level limit) and the third,
`eucalyptus`, produced a gap of +0.0290 against a 0.030 threshold on 736 rows
across 8 groups — the noise floor, not a measurement. Harness fixed:
regression support with |Spearman|, ordinal coding past 200 levels rather than
dropping those columns, `MIN_GROUPS = 20`, and a 120k row cap.

Second run, 3 of 5 datasets measured:

| dataset | rows | groups | random | grouped | gap |
|---|---:|---:|---:|---:|---:|
| SpeedDating (`wave`) | 8,378 | 21 | 0.8696 | 0.8482 | +0.0214 |
| KDD98 (`STATE`) | 120,000 | 54 | 0.5833 | 0.5842 | **−0.0009** |
| nyc-taxi (`PULocationID`) | 120,000 | 202 | 0.7475 | 0.7444 | +0.0031 |

**Misses: 0.** No dataset where a real split-dependence gap existed and the
tool stayed silent. That is the first evidence this project has had about
recall on this family of leak, and it is reassuring as far as three datasets go.

But the harness was built to work in both directions, and the other direction
is where it earned its keep.

### F6. `group-overlap` fires on almost everything, and is anti-correlated with what it detects — OPEN, SERIOUS

On KDD98 the check flagged **280+ columns** while the measured cost of using a
random split instead of a grouped one was **−0.0009** — zero. `nyc-taxi`
flagged `DOLocationID` against a gap of +0.0031. Both are false positives with
model-based evidence behind the verdict, not a judgement call.

**Root cause.** `_split_checks` flags a column when more than 90% of the test
side's values also appear in the train side, given more than 20 distinct
values. Under a *random* split that condition is arithmetic, not evidence:

```
rows per value    distinct levels    overlap
           1              40,000       54.1%
           2              20,000       80.4%
           5               8,000       97.7%
          50                 800      100.0%
         400                 100      100.0%
```

A value that appears 50 times lands on both sides of an 80/20 split with
probability ~1. So every ordinary categorical clears the bar, and measured on
real KDD98 columns the ordering is exactly backwards:

| column | levels | rows/level | overlap | flagged |
|---|---:|---:|---:|---|
| `AGE901` — an age percentile | 70 | 571 | 100.0% | **yes** |
| `DMA` — a media market | 203 | 197 | 100.0% | **yes** |
| `ZIP` — the most entity-like column present | **13,551** | **3** | **74.9%** | **no** |

The check flags ordinary low-cardinality features and *misses* the column that
actually is an identifier, because entity ids have few rows each and therefore
straddle a random split less often. It is not merely noisy; it is inverted.

**The deeper problem.** Overlap measured against a random split carries no
information at all — random splitting *is* the null hypothesis, so observed
overlap always equals its expected value. There is nothing to detect. Whatever
this check is for, it cannot be a statistical test on that quantity.

**Proposed fix — make it a structural warning, not a detection.** The useful
statement is "this column looks like an entity identifier and your split does
not respect it, so the same entity sits on both sides." That is about the
column's shape and the split's design, not about a p-value:

- entity-like: many distinct values *and* few rows per value (roughly 2–100),
  which is what distinguishes `ZIP` at 3 rows/value from `AGE901` at 571
- and any straddling at all — drop the 90% threshold, since it is the part
  doing the damage

Under that rule `ZIP` fires and `AGE901` is silent, which is the exact
inversion of today's behaviour and the strongest evidence available that the
current rule is wrong.

**Cross-check before accepting** the demo frame's `customer_id` (1 row per
value — an identifier, but a random split cannot leak it since nothing
repeats: should it fire? probably as `identifier-like`, which already covers
it, and *not* as group-overlap), and `SpeedDating`'s `wave` (21 levels across
8,378 rows = 399 each — a real grouping variable that a random split genuinely
does leak, and which the new rule would *not* flag). That second case is a
genuine tension in the proposal and needs resolving before it lands: a
low-cardinality grouping variable is exactly what GroupKFold exists for, so
"few rows per value" cannot be the whole story.

### Remaining harness bugs (mine, not the tool's)

- `us_crime` still dies with "Unknown label type: continuous" — the regression
  branch is not being selected, so `_target_kind` is not returning
  `continuous` for that target. Needs a look at what dtype OpenML ships.
- `Amazon_employee_access`: `TypeError: '<' not supported between NoneType and
  str` — something sorts a column holding both `None` and strings.

---

## Round 6 — the first measured MISS

Both harness bugs from round 5 fixed, four of five datasets measured.

| dataset | rows | groups | random | grouped | gap | verdict |
|---|---:|---:|---:|---:|---:|---|
| SpeedDating (`wave`) | 8,378 | 21 | 0.8696 | 0.8482 | +0.0214 | below threshold |
| **us_crime** (`state`) | **1,994** | **46** | **0.8158** | **0.7773** | **+0.0385** | **MISS** |
| KDD98 (`STATE`) | 120,000 | 54 | 0.5833 | 0.5812 | +0.0021 | F6 false positives |
| nyc-taxi (`PULocationID`) | 120,000 | 202 | 0.7475 | 0.7443 | +0.0032 | F6 false positive |

### F7. An integer entity code is invisible to `group-overlap` below ~46,000 rows — OPEN

**Measured cost** On `us_crime`, using a random split instead of grouping by
`state` is worth **+0.0385 of |Spearman|** — communities in the same state
share something a model exploits. targetleak reported nothing.

**Reproduce**

```bash
python - <<'PY'
from sklearn.datasets import fetch_openml
import numpy as np, targetleak as tl
b = fetch_openml("us_crime", version=2, as_frame=True, parser="pandas")
df = b.frame.copy()
df["_split"] = np.where(np.random.default_rng(0).random(len(df)) < 0.8,
                        "train", "test")
out = tl.analyse(df, b.target.name, split="_split")
print([f.column for f in out if f.kind == "group-overlap"])     # []
PY
```

`state` is `int64`, 46 distinct, ~43 rows each, and its overlap across the
split is **100.0%** — comfortably past the >90% bar. It is skipped before the
bar is ever reached.

**Root cause — `_looks_categorical`, again.** `_split_checks` skips any column
that is numeric and not "categorical", and that predicate is

```python
is_integer_dtype(col) and col.nunique() <= max(10, len(col) // 1000)
```

which makes the answer depend on the row count:

| rows | threshold | is a 46-state code categorical? |
|---:|---:|---|
| 1,994 | 10 | **no** |
| 20,000 | 20 | **no** |
| 46,000 | 46 | yes |
| 120,000 | 120 | yes |

A US state code becomes a category only above ~46,000 rows. Below that it is
treated as a continuous measurement, so the group-overlap scan never looks at
it. This is the **third** distinct bug traced to that one formula — it caused
the integer-cardinality cliff in scoring earlier this session, was flagged as
unprincipled in the correctness audit, and was never fixed because nothing had
yet shown it doing damage. It has now.

**Proposed fix** Replace the row-count term with an absolute cardinality
ceiling. An integer column with a modest number of distinct values is a code
regardless of how many rows sit underneath it; 46 states are 46 states in
2,000 rows or in 2,000,000. Then re-check the integer-cliff regression test,
which exists precisely to pin the scoring side of this predicate.

**Secondary, and worth fixing at the same time.** Passing `--group state`
*also* excludes that column from the overlap scan, so the one column the user
has identified as the entity is never checked against the split they also
supplied. That is backwards: if a user names both, the most useful sentence
the tool can produce is "your split does not respect the group you gave me."
Confirmed above — findings are empty with `--group state` and empty without
it, for two different reasons stacked on each other.

**Interaction with F6.** F6 says `group-overlap` fires on almost everything;
F7 says it misses the actual entity. Both are true, and they are the same
mistake seen from two sides: the check tests overlap, which under a random
split is determined by rows-per-value, while entity-ness is about
cardinality. Fixing F6 without F7 would quieten the noise and keep the blind
spot. **They should be fixed together, and the `us_crime` gap of +0.0385 is
the regression test for whether the fix works.**

### Not a finding: `Amazon_employee_access`

Still fails in the harness with `'<' not supported between NoneType and str`
even after normalising the target and group out of `category` dtype.
`tl.analyse` handles that frame without complaint — verified directly — so
this is mine, somewhere in the sklearn plumbing, and it is not evidence about
the tool. Left unfixed; it costs one dataset of coverage, not correctness.

---

## Round 7 — 95 datasets

`python benchmark/sweep.py --slow-secs 120` · 95 ran, 4 unreachable, **0 slow**,
no new crashes. One new F2 instance, one positive result, and one case that
breaks F2's proposed fix.

Scale is now thoroughly settled:

| dataset | shape | secs | findings |
|---|---|---:|---|
| `click_prediction_small` | **1,496,391** × 10 | 10.8 | 1 warning |
| `riccardo` | 20,000 × **4,297** | 13.9 | 1 warning |
| `christine` | 5,418 × 1,637 | 5.2 | none |
| `creditcard` | 284,807 × 30 | 2.7 | 0 critical, 7 warning |

### Positive result: the base-rate gate holds at 0.17%

`creditcard` is 284,807 rows with **492 positives — 0.1727%**. This is the
direct stress test for the base-rate gate added earlier, since the subset-
purity check it guards once fired on **9 of 15 pure-noise columns at a 1% base
rate**. At six times that extremity:

```
critical = 0
warning  = 7   ['duplicate-rows', 'suspiciously-predictive']
scores   = V3 0.912, V10 0.914, V11 0.918, V12 0.937, V4 0.938, V14 0.949
```

Zero false criticals. V3/V4/V10–V14 are PCA components constructed on this
dataset to separate fraud, so being individually predictive is correct, and
`suspiciously-predictive` — "confirm this exists at prediction time" — is
exactly the right severity for them rather than `critical`. `duplicate-rows`
is also real: the dataset genuinely contains repeated transactions.

### F2, seventh instance

`shuttle`, 58,000 × 10: 5 critical + 2 warning on 9 features = **78%**, and no
widespread note because 7 flagged is below the floor of 8. Same root cause as
breast-w, glass, ecoli, yeast, letter and diamonds.

### F2's proposed fix is WRONG, and this is the case that shows it

`one-hundred-plants-margin`, 1,600 × 65, **100 classes** (exactly at the new
ceiling — the boundary held, no crash).

```
critical           = 32  of 64 features
near-perfect >=0.99 = 18  of those 32
critical score range = 0.9800 - 1.0000
widespread note     = fired
```

My proposal was: fire the reassurance when no flagged column is near-perfect,
and **suppress it when near-perfect columns are present**, because those are
leak candidates. Here 18 columns are near-perfect, so the proposal would
withhold the explanation and hand the user eighteen columns to check. That is
worse than either reframing everything or nothing.

And these are not false positives. 64 leaf-margin features across 100 species
with 16 rows per class: a feature that perfectly separates one species gives
n1=16, n0=1584, null SE 0.0726, so a score of 1.0 lands at z = 6.9 against a
bar of 5.0. Statistically solid, and substantively right — that feature really
does identify that species.

**What the case actually reveals.** The severity is wrong for a reason the
score distribution cannot see: **one-vs-rest inflates apparent severity on
many-class problems.** "This column is near-perfect for species 47 of 100" is
being reported with the same weight as "this column is the answer", when the
first is a feature doing its job and the second is a leak. A leak solves the
*whole* target; these solve one class each.

**Revised proposal, third attempt.** For a multiclass target, cap a
one-vs-rest finding at `warning` unless the column is predictive across the
target broadly rather than for a single class — the metric string already
carries `AUC vs class 'X'`, so the information needed is present, only the
severity is miscalibrated. Then the widespread note is applied on top of a
severity scale that already means something, and the near-perfect test can go
back to doing the job I designed it for on binary targets.

**Note.** This is the third revision of the F2 fix, and each revision came
from a dataset the previous version had not seen: narrow frames (breast-w),
then `har`'s mixed population, now a 100-class problem. That is an argument
for landing none of it until it has been checked against all seven instances
plus the demo frame, and a reason to be glad this loop was scoped to report
rather than commit.

---

## Round 8 — output paths, and a wall of useless findings

This round tested a surface the sweep never touches: the *output* paths.
`to_html`, `render_json` and `fix_code` had never run against real column
names or real widths — `fix_code` in particular emits executable Python built
from names like `od280/od315_of_diluted_wines`, `home.dest` and
`lpep_pickup_datetime`.

**Clean, 0 problems across 10 datasets from 9 to 4,296 columns.** HTML stays
between 9 KB and 288 KB even on a 4,296-column frame, every `fix_code` block
parses with `ast.parse`, and every JSON payload round-trips. That surface is
sound and needs no work.

But one row was odd — `us_crime` emitted 288 KB of HTML, so plenty of
findings, while `fix_code` returned `None`.

### F8. 111 findings, none actionable, none with remediation text — OPEN

`us_crime`, 1,994 × 127. Findings:

```
info   underpowered   111
info   target           1
```

Every scored column is `underpowered` and nothing else. The report is 288 KB
of "this scored high but on too little data to trust", there is no
remediation block because `fix_code` handles none of these kinds, and
**`underpowered` is not in `FIXES` at all** — so those 111 findings carry no
guidance whatsoever. The tool's central claim is that every finding says what
to do about it; here 111 of 112 say nothing.

The test that should have caught the missing `FIXES` entry is
`test_every_visible_kind_has_a_remedy`, which the quality audit already
flagged as checking only the ~8 kinds the demo frame happens to emit. This is
that bad test having a consequence.

**Proposed fix** Add a `FIXES` entry for `underpowered` (what to do: get more
rows, or accept that this column cannot be judged at this sample size, and
which it is). Then replace that test with a suite-level assertion that every
key in `FIXES` is reachable and every emitted kind has an entry — the version
the audit recommended and I did not implement.

### F9. A normalised float with 98 distinct values is treated as a discrete flag — OPEN

**This is the cause of F8's wall, and my first hypothesis for it was wrong.**
I assumed F4 — the target being read as 98-class multiclass — was responsible.
It is not: with the target forced continuous, us_crime *still* produces 107
underpowered findings. The cause is on the predictor side.

**Reproduce**

```bash
python - <<'PY'
from sklearn.datasets import fetch_openml
import numpy as np, targetleak as tl
b = fetch_openml("us_crime", version=2, as_frame=True, parser="pandas")
df = b.frame
y = df[b.target.name] + np.linspace(0, 1e-9, len(df))   # force continuous
print(tl._score_column(df["population"], y, "continuous"))
# score 0.9940, z 1.71, bar 3.5  -> underpowered
PY
```

| column | distinct values | score | z | bar | verdict |
|---|---:|---:|---:|---:|---|
| `state` | 46 | 0.9925 | 1.71 | 3.5 | underpowered |
| `population` | 66 | 0.9940 | 1.71 | 3.5 | underpowered |
| `householdsize` | 93 | 0.9731 | 2.32 | 3.5 | underpowered |
| `fold` | 10 | 0.5316 | 1.47 | 3.5 | ok |

**Root cause.** The discrete-predictor branch I added for the continuous-axis
fix triggers when a predictor has `<= MAX_CLASSES` (100) distinct values —
**120 of us_crime's 126 features qualify.** Each distinct value then becomes a
one-vs-rest group, so with ~1,994 rows over ~100 values there are ~20 rows per
group; the max over 100 groups pushes the score past 0.90 while the small
group sizes keep z below the bar. Result: a wall of `underpowered`.

But these are normalised floats — measurements, not flags. The branch exists
for genuine discrete predictors (a binary flag, a handful of codes), which is
where correlation genuinely fails and the top-2%-of-revenue case lives.

**This is the same root error as F4, on the other side of the equation.**
`MAX_CLASSES = 100` is now making three different decisions: whether a target
is multiclass, whether a predictor is discrete, and the cost ceiling for
one-vs-rest. It is defensible for exactly one of them — non-numeric class
labels — and wrong for both numeric cases.

**Proposed fix** Give the discrete-predictor branch its own much smaller
threshold (~15 distinct values), and require the column to be non-float or
genuinely low-cardinality. Fold this together with F4: one rule, stated once —
*a numeric column with many distinct values is a measurement; order is
information; do not shred it into one-vs-rest groups*, whether it appears as
the target or as a predictor.

**Regression cases for the combined fix**: the top-2%-of-revenue flag (must
still score 1.0000 — that is what the branch was built for), `cpu_act`'s
56-value percentage target (must not produce 4 criticals), `us_crime` (must
not produce 111 underpowered findings), and `diamonds` (carat must still be
found).

---

## Round 9 — round-trip consistency

Compared findings three ways for 12 real datasets: in memory, after a CSV
round-trip, after a Parquet round-trip. Ten agreed exactly. **Two did not**,
and this is the failure family that has produced more bugs here than any
other — the pandas-3 `str`-vs-`object` bug was found exactly this way.

### F10. Findings depend on the FILE FORMAT — OPEN, SERIOUS

**Dataset** `anneal`, 898 × 39. Identical data, loaded two ways:

| column | distinct | nulls | as `category` (memory) | as `float64` (CSV) |
|---|---:|---:|---:|---:|
| `formability` | 4 | 318 | score **0.9185**, z 11.41 | score **0.6427**, z 3.80 |
| `enamelability` | 2 | 882 | score **0.5539**, z 1.75 | score **1.0000**, z 3.25 |

The flagged column *moves*: in memory `formability` is
`suspiciously-predictive`; from CSV it is silent and `enamelability` appears
instead. Same 898 rows, same values, different verdict.

**Reproduce**

```bash
python - <<'PY'
from sklearn.datasets import fetch_openml
import tempfile, os, targetleak as tl
b = fetch_openml("anneal", version=1, as_frame=True, parser="pandas")
df, t = b.frame, b.target.name
p = os.path.join(tempfile.mkdtemp(), "a.csv"); df.to_csv(p, index=False)
print(tl._score_column(df["formability"], df[t], "multiclass")["score"])   # 0.9185
d2 = tl.load(p)
print(tl._score_column(d2["formability"], d2[t], "multiclass")["score"])   # 0.6427
PY
```

**Root cause.** `candidates()` chooses what to measure from the **dtype**, and
the two branches are not comparable:

```python
if is_float_dtype(col):        return [(col, False)]        # raw ranks ONLY
...
enc = oof_target_encode(col)
if is_numeric_dtype(col):      return [(col, False), enc]   # both
return [enc]                                                # encoding ONLY
```

A `category` column gets *only* the target encoding. The same values as
`float64` get *only* the raw rank AUC. Those measure different things, so a
four-value column scores 0.92 or 0.64 depending on nothing but how it was
loaded — and `pd.read_csv` produces the float, so **the CSV path a user
actually takes is the one that disagrees with the tests.**

**This is the third variant of one root error.** F4: `MAX_CLASSES` deciding a
numeric target's kind. F9: `MAX_CLASSES` deciding a numeric predictor's
discreteness. F10: dtype deciding which measurement a column gets. Every one
substitutes a storage detail for a property of the data. A four-value column
is a four-value column whether pandas calls it `category`, `float64` or
`object`.

**Proposed fix** Choose candidates from cardinality, not dtype. A
low-cardinality column gets both the encoded and the ordinal candidate
regardless of how it is stored; a high-cardinality float gets ranks only,
because encoding it is meaningless. Then add the round-trip check from this
round to the test suite over several real dataset shapes — the existing
`test_demo_reads_identically_from_disk` covers one hand-written frame and
these two slipped past it.

### F11. `plating_tank` is a tank number, and gets a date warning — OPEN

`cylinder-bands` column `plating_tank` holds `'1911', '1910', …` — tank
identifiers. As a `category` those parse as years, so `_is_timelike` returns
True and the column earns a `temporal-column` warning whose remediation tells
the user to `df.sort_values('plating_tank')` and split on it. As `float64`
from CSV it does not fire, which is the second half of F10.

The correctness audit already predicted this — "bare 4-digit years and
hyphenated part numbers all parse as dates" — and this is the first confirmed
instance on real data, with wrong remediation attached.

**Proposed fix** Require more than a bare 4-digit integer before calling a
column temporal: a real date column parses with separators, or has a
datetime64 dtype, or spans a plausible range with month/day components. Four
digits alone is a year, a tank, a model number or a postcode.

---

## Round 10 — how widespread is F10?

F10 was found on 2 of 12 datasets, which is not enough to prioritise it.
Widened to 36 real datasets, comparing findings in memory against findings
after a CSV round-trip:

| | |
|---|---:|
| datasets compared | 36 |
| carrying at least one `category` feature | 22 |
| **findings changed by a CSV round-trip** | **4 (11% of all, 18% of those with categoricals)** |
| of those 4, had categorical columns | **4 of 4** |

**No dataset without categorical columns differed.** That confirms the
mechanism rather than inferring it: F10 is precisely a categorical-versus-
numeric dtype split, and its blast radius is the 61% of these datasets that
carry a `category` column at all.

Affected: `cylinder-bands` (1 finding), `analcatdata_dmft` (1), `anneal` (2),
`dermatology` (8).

### F10's impact is a changed verdict, not a wobbled score

`dermatology`, 366 × 35, 6 classes. Four columns cross the
`AUC_CRITICAL = 0.98` boundary purely on file format:

| column | as `category` | as `int64` from CSV |
|---|---|---|
| `clubbing_of_the_rete_ridges` | 0.9721 → **warning** | 0.9837 → **critical** |
| `focal_hypergranulosis` | warning | critical |
| `melanin_incontinence` | warning | critical |
| `thinning_of_the_suprapapillary_epidermis` | warning | critical |

```
in memory: exit=1  VERDICT: 7 critical leak(s)...
from CSV : exit=1  VERDICT: 11 critical leak(s)...
```

Same 366 rows. The score shift is small — 0.9721 to 0.9837 — but it lands
across a hard threshold, and four columns do it at once. A user reading the
CSV report drops four columns the in-memory report told them only to confirm.

**Honest limit on this one:** the exit code does not flip here, because
`dermatology` has 7 criticals either way, so a CI gate would fail in both
cases. The flip would matter on a dataset sitting at zero criticals in one
representation and one in the other; I have not found such a case, and should
not claim one.

**This raises F10's priority above F2.** F2 produces noisy reports on easy
datasets — annoying, and every individual finding is still true. F10 makes the
tool's output depend on a storage detail, which means two engineers analysing
the same data disagree, and neither is wrong. For a tool whose entire value is
being trustworthy about what is real, that is the worse failure.

---

## Where this stands after 10 rounds

**~100 distinct real datasets swept. 15 findings, 5 of them serious, none
fixed** — the loop was scoped to report, and everything is reproducible from
the entries above.

The bottleneck is no longer discovery. Four of the five serious findings —
F4, F9, F10, and F11 — are **one root cause**: a decision about a column being
made from how it is *stored* rather than what it *contains*. Fixing that one
thing addresses a numeric target read as 98 classes, 120 of 126 features read
as discrete flags, findings that change with file format, and a tank number
read as a year. F2 and F6/F7 are separate and also need work.

Further sweeping will keep finding instances of what is already recorded.
The next genuinely new information comes from fixing the root cause and
re-measuring — which needs the code changes this loop was told not to make.
