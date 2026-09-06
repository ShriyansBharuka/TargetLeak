#!/usr/bin/env python3
"""Robustness sweep: run targetleak at many real datasets and look for trouble.

This is not the benchmark. `run_benchmark.py` measures accuracy against
datasets whose leakage status is known. This one has no ground truth and does
not pretend to - it looks for the three failures that can be recognised
without knowing what is hidden in the data:

    crash       an exception escaping analyse()
    slow        runtime out of proportion to the frame
    noisy       so many columns flagged that the report is unusable

That is a real limit worth stating: a sweep like this cannot tell you the tool
MISSED a leak, because nobody has annotated these files. It hardens robustness
and false-positive behaviour, and says nothing about recall.

Every bug found so far came from real data of a shape the tests did not
anticipate, which is why this exists and why it prefers awkward datasets over
convenient ones - hundreds of columns, sixteen classes, half-empty frames,
sparse text features, two hundred rows.

Each dataset is also run a second time with a synthetic 80/20 split column, to
exercise the contamination and group-overlap paths. Those have the thinnest
real-world exposure of anything in the tool.

  python benchmark/sweep.py                 # the standard list
  python benchmark/sweep.py --limit 10      # first N, for a quick check
  python benchmark/sweep.py --slow-secs 30  # what counts as slow

Exits non-zero if anything crashed.
"""
import argparse
import sys
import time
import traceback
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
import targetleak as tl  # noqa: E402

NOISY_SHARE = 0.30

# (name, openml version). Chosen for awkwardness: width, class count,
# missingness, sparsity, tiny row counts, regression targets.
DATASETS = [
    ("credit-g", 1), ("adult", 2), ("kc1", 1), ("bank-marketing", 1),
    ("Australian", 4), ("cylinder-bands", 2), ("dresses-sales", 1),
    ("SpeedDating", 1), ("churn", 1), ("arrhythmia", 1), ("sick", 1),
    ("hypothyroid", 1), ("autos", 2), ("cnae-9", 1), ("hill-valley", 1),
    ("us_crime", 2), ("mfeat-factors", 1), ("analcatdata_dmft", 1),
    ("kr-vs-kp", 1), ("letter", 1), ("optdigits", 1), ("satimage", 1),
    ("segment", 1), ("splice", 1), ("vehicle", 1), ("spambase", 1),
    ("tic-tac-toe", 1), ("balance-scale", 1), ("breast-w", 1), ("car", 1),
    ("cmc", 1), ("credit-approval", 1), ("dermatology", 1), ("ecoli", 1),
    ("eucalyptus", 1), ("glass", 1), ("haberman", 1), ("ionosphere", 1),
    ("mushroom", 1), ("nursery", 1), ("pendigits", 1), ("phoneme", 1),
    ("soybean", 1), ("vote", 1), ("wilt", 1), ("yeast", 1),
    ("monks-problems-1", 1), ("jm1", 1), ("ozone-level-8hr", 1),
    ("qsar-biodeg", 1), ("steel-plates-fault", 1), ("texture", 1),
    ("cardiotocography", 1), ("PhishingWebsites", 1), ("banknote-authentication", 1),
    ("climate-model-simulation-crashes", 1), ("blood-transfusion-service-center", 1),
    ("MiceProtein", 1), ("wall-robot-navigation", 1), ("pc4", 1),

    # Round 2 of the sweep loop. Each of these stresses something nothing in
    # the list above reaches.
    ("Bioresponse", 1),              # 1,776 columns - extreme width
    ("Amazon_employee_access", 1),   # integer codes with thousands of levels
    ("KDDCup09_appetency", 1),       # 231 columns, most of them mostly empty
    ("isolet", 1),                   # 617 columns and 26 classes at once
    ("har", 1),                      # 561 columns, sensor signals
    ("anneal", 1),                   # heavy missingness across mixed dtypes
    ("primary-tumor", 1),            # 22 classes on 339 rows - thin support
    ("audiology", 1),                # 24 classes on 226 rows - thinner still
    ("abalone", 1),                  # 29 ordered classes, borderline regression
    ("cholesterol", 1),              # regression target with missing values

    # Round 3. Biggest gap after 69 datasets: not one of them had a real date
    # column, so _is_timelike and temporal-column have had almost no exposure
    # outside frames I wrote. Then scale, then regression targets.
    ("electricity", 1),              # has a genuine `date` column
    ("covertype", 3),                # 581,012 rows - scale
    ("poker-hand", 1),               # ~1M rows - more scale
    ("wine_quality", 1),             # ordered classes, borderline regression
    ("kin8nm", 1),                   # regression
    ("cpu_act", 1),                  # regression, 22 columns
    ("pol", 1),                      # regression, 49 columns
    ("house_16H", 1),                # regression, skewed target
    ("arcene", 1),                   # 200 rows x 10,001 columns - wide and tiny
    ("madelon", 1),                  # 500 columns, deliberately hard

    # Round 4. Hunting a genuine date column: `electricity` turned out to be
    # normalised to floats, so temporal-column has still never seen a real
    # one. Time-based leakage is the kind practitioners hit most often.
    ("Bike_Sharing_Demand", 2),
    ("nyc-taxi-green-dec-2016", 2),
    ("rainfall_bangladesh", 1),
    ("Rain_in_Australia", 1),
    ("avocado_sales", 1),
    ("seattlecrime6", 2),
    ("diamonds", 1),
    ("Airlines", 1),
    ("KDD98", 1),
    ("okcupid-stem", 1),

    # Round 7. Extreme imbalance is the real target here: at a 0.17% base rate
    # the null SE and the base-rate gate on subset purity get stressed harder
    # than anything swept so far. Plus the 100-class ceiling boundary and a
    # 4,297-column frame.
    ("creditcard", 1),               # 284,807 x 31, ~0.17% positive
    ("APSFailure", 1),               # 76k x 171, imbalanced AND heavy missing
    ("one-hundred-plants-margin", 1),  # exactly 100 classes - the ceiling
    ("shuttle", 1),                  # imbalanced 7-class
    ("connect-4", 1),                # 67k x 43, 3 classes
    ("volkert", 1),                  # 58k x 181, 10 classes
    ("riccardo", 1),                 # 20k x 4,297 - very wide
    ("christine", 1),                # 1,636 x 1,637 - wide with few rows
    ("jasmine", 1),                  # 2,984 x 145
    ("click_prediction_small", 1),   # imbalanced click-through
]


def fetch(name, version):
    from sklearn.datasets import fetch_openml
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        b = fetch_openml(name, version=version, as_frame=True, parser="pandas")
    target = b.target.name if b.target is not None else b.frame.columns[-1]
    return b.frame, target


def run_one(df, target, split=None):
    t = time.time()
    findings = tl.analyse(df, target, split=split)
    secs = time.time() - t
    loud = {f.column for f in findings
            if f.column and f.severity in ("critical", "warning")
            and f.kind in ("target-proxy", "suspiciously-predictive",
                           "pure-categories", "missingness-leak")}
    n_features = len(df.columns) - 1 - (1 if split else 0)
    return {
        "secs": secs,
        "critical": sum(f.severity == "critical" for f in findings),
        "warning": sum(f.severity == "warning" for f in findings),
        "share": len(loud) / max(n_features, 1),
        "widespread": any(f.kind == "widespread-separability" for f in findings),
        "kinds": sorted({f.kind for f in findings}),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, help="only the first N datasets")
    ap.add_argument("--slow-secs", type=float, default=45.0)
    a = ap.parse_args(argv)

    todo = DATASETS[:a.limit] if a.limit else DATASETS
    crashes, slow, noisy, skipped, done = [], [], [], [], 0

    print(f"{'dataset':34}{'shape':>13}{'cls':>5}{'secs':>7}{'crit':>5}"
          f"{'warn':>5}{'flag':>6}  notes")
    for name, ver in todo:
        try:
            df, target = fetch(name, ver)
        except Exception as e:
            skipped.append((name, type(e).__name__))
            print(f"{name[:33]:34}{'fetch failed':>13}  {type(e).__name__}")
            continue

        n_cls = int(df[target].nunique())
        shape = f"{len(df):,}x{len(df.columns)}"
        try:
            r = run_one(df, target)
        except Exception as e:
            crashes.append((name, f"{type(e).__name__}: {e}"))
            print(f"{name[:33]:34}{shape:>13}{n_cls:>5}   *** CRASH "
                  f"{type(e).__name__}: {str(e)[:34]}")
            traceback.print_exc(limit=3)
            continue

        # Second pass with a synthetic split, to exercise the contamination
        # and group-overlap paths against real column shapes.
        notes = []
        try:
            split_df = df.copy()
            rng = np.random.default_rng(0)
            split_df["_sweep_split"] = np.where(
                rng.random(len(df)) < 0.8, "train", "test")
            run_one(split_df, target, split="_sweep_split")
        except Exception as e:
            crashes.append((f"{name} (--split)", f"{type(e).__name__}: {e}"))
            notes.append(f"SPLIT CRASH {type(e).__name__}")
            traceback.print_exc(limit=3)

        done += 1
        if r["secs"] > a.slow_secs:
            slow.append((name, r["secs"], shape))
            notes.append("SLOW")
        if r["share"] > NOISY_SHARE and not r["widespread"]:
            noisy.append((name, r["share"], shape))
            notes.append("NOISY, unexplained")
        elif r["widespread"]:
            notes.append("widespread (named)")
        print(f"{name[:33]:34}{shape:>13}{n_cls:>5}{r['secs']:>7.1f}"
              f"{r['critical']:>5}{r['warning']:>5}{r['share']:>5.0%}  "
              f"{' '.join(notes)}")

    print()
    print("=" * 72)
    print(f"ran {done} datasets, {len(skipped)} unreachable")
    print(f"crashes            : {len(crashes)}")
    for n, e in crashes:
        print(f"    {n}: {e[:90]}")
    print(f"slow (>{a.slow_secs:.0f}s)        : {len(slow)}")
    for n, s, shape in slow:
        print(f"    {n}: {s:.1f}s on {shape}")
    print(f"noisy, unexplained : {len(noisy)}")
    for n, sh, shape in noisy:
        print(f"    {n}: {sh:.0%} of columns flagged on {shape}")
    return 1 if crashes else 0


if __name__ == "__main__":
    sys.exit(main())
