#!/usr/bin/env python3
"""Corroborate findings against what a model actually does. Recall evidence.

The sweep and the benchmark both have a hole: neither can tell you targetleak
MISSED something, because nobody has annotated these datasets. This closes part
of that hole without needing ground truth, by using a property of leakage
rather than a label.

The property: **a leak inflates a random split and collapses under a correct
one.** Entity identity leaking across a random split is why the same patient,
user or ticker appearing on both sides makes a model look good; splitting by
that entity removes the free lunch.

So for a dataset with an entity column:

    A = cross-validated score under a RANDOM split
    B = cross-validated score under a GROUPED split
    gap = A - B

A large gap means the random split was being exploited. That is a fact about
the data, established by a model, with no annotation involved. Then compare it
to what targetleak said:

    gap large,  tool warned    -> corroborated
    gap large,  tool silent    -> a MISS, and the interesting case
    gap small,  tool warned    -> possible false positive
    gap small,  tool silent    -> agreement

What this does NOT prove: a large gap with a warning does not confirm the
specific column named, only that something split-dependent is going on. And a
small gap does not clear the data - a target proxy available at prediction time
inflates both splits equally and shows no gap at all. This measures one family
of leak, which is the family the --group and --split checks are about.

  python benchmark/split_sensitivity.py
  python benchmark/split_sensitivity.py --limit 3
"""
import argparse
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
import targetleak as tl  # noqa: E402

# (openml name, version, entity column, why it is an entity)
GROUPED = [
    ("SpeedDating", 1, "wave", "one speed-dating session; participants repeat"),
    ("us_crime", 2, "state", "US state; communities share policy (regression)"),
    ("KDD98", 1, "STATE", "US state across 191k donors"),
    ("Amazon_employee_access", 1, "MGR_ID", "manager; reports share access"),
    ("nyc-taxi-green-dec-2016", 2, "PULocationID", "pickup zone"),
]

MIN_GAP = 0.03          # below this, the split choice made no real difference
FOLDS = 4
# HistGradientBoosting refuses categoricals above 255 levels. Columns past this
# are ordinal-coded rather than dropped: dropping them would remove exactly the
# high-cardinality entity-ish features whose leakage this is trying to detect.
MAX_CAT_LEVELS = 200
# A group count near the fold count makes the comparison meaningless. The first
# run measured a 736-row dataset across 8 groups and produced a gap of 0.029
# against a 0.03 threshold, which is the noise floor, not a finding.
MIN_GROUPS = 20
# Cap the work so one 581k-row dataset cannot eat the whole run.
MAX_ROWS = 120_000


def _model(X, regression):
    """A fast learner that eats mixed dtypes and NaNs without preprocessing.

    Deliberately not tuned. The question is whether the SPLIT changes the
    score, so anything that imputes or one-hot-encodes would add its own
    leakage and confound the measurement.
    """
    from sklearn.ensemble import (HistGradientBoostingClassifier,
                                  HistGradientBoostingRegressor)
    cat = [isinstance(X[c].dtype, pd.CategoricalDtype) for c in X.columns]
    cls = (HistGradientBoostingRegressor if regression
           else HistGradientBoostingClassifier)
    return cls(max_iter=60, random_state=0,
               categorical_features=cat if any(cat) else None)


def _prep(df, target, group):
    y = df[target]
    X = df.drop(columns=[c for c in (target, group) if c in df.columns])
    # HistGB wants numeric or categorical, and refuses categoricals past 255
    # levels. Anything wider is ordinal-coded rather than dropped - those are
    # the entity-like columns this exists to catch.
    for c in X.columns:
        if (pd.api.types.is_object_dtype(X[c]) or X[c].dtype == "string"
                or isinstance(X[c].dtype, pd.CategoricalDtype)):
            codes = pd.factorize(X[c], use_na_sentinel=True)[0].astype(float)
            codes[codes < 0] = np.nan
            if int(X[c].nunique(dropna=True)) > MAX_CAT_LEVELS:
                X[c] = codes                        # ordinal, treated numeric
            else:
                X[c] = pd.Series(codes, index=X.index).astype("category")
    # Normalise the target and the group out of pandas `category` dtype.
    # sklearn's GroupKFold and np.unique sort them, and a categorical carrying
    # unused or null categories raises "'<' not supported between NoneType and
    # str" - a harness failure that looks like a tool failure in the log.
    if isinstance(y.dtype, pd.CategoricalDtype):
        y = y.astype("object").astype("str")
    g = df[group]
    if not pd.api.types.is_numeric_dtype(g):
        g = pd.Series(pd.factorize(g, use_na_sentinel=False)[0], index=g.index)
    keep = y.notna().to_numpy()
    return X[keep], y[keep], g[keep]


def _score(X, y, splits, regression):
    """AUC for classification, |Spearman| for regression.

    Both are rank-based and higher-is-better, so a gap between two splits
    means the same kind of thing in either problem type.
    """
    from sklearn.metrics import roc_auc_score
    out = []
    classes = pd.Series(y).unique()
    for tr, te in splits:
        if not regression and (len(np.unique(y.iloc[te])) < 2
                               or len(np.unique(y.iloc[tr])) < 2):
            continue
        m = _model(X, regression)
        m.fit(X.iloc[tr], y.iloc[tr])
        if regression:
            pred = pd.Series(m.predict(X.iloc[te]))
            rho = pred.corr(pd.Series(np.asarray(y.iloc[te])), method="spearman")
            if pd.notna(rho):
                out.append(abs(float(rho)))
        elif len(classes) == 2:
            out.append(roc_auc_score(y.iloc[te],
                                     m.predict_proba(X.iloc[te])[:, 1]))
        else:
            p = m.predict_proba(X.iloc[te])
            out.append(roc_auc_score(y.iloc[te], p, multi_class="ovr",
                                     average="macro", labels=m.classes_))
    return float(np.mean(out)) if out else float("nan")


def evaluate(name, version, group):
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import GroupKFold, KFold
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        b = fetch_openml(name, version=version, as_frame=True, parser="pandas")
    df = b.frame
    target = b.target.name
    if group not in df.columns:
        near = [c for c in df.columns if group.lower() in c.lower()]
        return {"skip": f"no column {group!r}" + (f" (similar: {near})" if near else "")}
    if len(df) > MAX_ROWS:
        df = df.sample(MAX_ROWS, random_state=0).reset_index(drop=True)

    X, y, g = _prep(df, target, group)
    if X.empty or y.nunique() < 2:
        return {"skip": "nothing to model"}
    # Decided independently, NOT by asking tl._target_kind. The harness exists
    # to check the tool, so taking the tool's own classification of the target
    # is circular - and it bit: targetleak calls us_crime's float64 crime rate
    # (98 distinct values) "multiclass", so the harness fed a continuous target
    # to a classifier and crashed. A float target with many distinct values is
    # a regression problem regardless of what the tool under test believes.
    regression = (pd.api.types.is_float_dtype(y) and y.nunique() > 20) or         (pd.api.types.is_numeric_dtype(y) and y.nunique() > 50)

    n_groups = int(g.nunique())
    if n_groups < MIN_GROUPS:
        return {"skip": f"only {n_groups} groups - too few to measure a gap"}

    random_splits = list(KFold(n_splits=FOLDS, shuffle=True,
                               random_state=0).split(X))
    grouped_splits = list(GroupKFold(n_splits=FOLDS).split(X, y, groups=g))

    a = _score(X, y, random_splits, regression)
    bb = _score(X, y, grouped_splits, regression)

    # A random split column is passed alongside the group, on purpose. Without
    # a split, `group-overlap` cannot fire at all - and that finding is the
    # whole point of the comparison: it says the entity appears on both sides
    # of a random split, which is exactly the condition that makes the
    # random-split score above unearned. Passing `group=` alone, as this
    # script first did, could only ever surface `identifier-like`.
    probe = df.copy()
    probe["_random_split"] = np.where(
        np.random.default_rng(0).random(len(df)) < 0.8, "train", "test")
    findings = tl.analyse(probe, target, split="_random_split", group=group)
    warned = sorted({f.column for f in findings
                     if f.kind in ("group-overlap", "identifier-like",
                                   "train-test-contamination")
                     and f.severity in ("critical", "warning")})
    return {"random": a, "grouped": bb, "gap": a - bb,
            "metric": "|Spearman|" if regression else "AUC",
            "groups": n_groups, "rows": len(X), "warned": warned}


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args(argv)

    misses, ran = [], 0
    for name, ver, group, why in (GROUPED[:a.limit] if a.limit else GROUPED):
        print(f"\n{name}  (grouping on {group!r} - {why})")
        try:
            r = evaluate(name, ver, group)
        except Exception as e:
            print(f"  harness failed: {type(e).__name__}: {str(e)[:90]}")
            continue
        if "skip" in r:
            print(f"  skipped: {r['skip']}")
            continue
        ran += 1
        print(f"  {r['rows']:,} rows, {r['groups']} groups")
        print(f"  random-split  {r['metric']:11} {r['random']:.4f}")
        print(f"  grouped-split {r['metric']:11} {r['grouped']:.4f}")
        print(f"  gap                       {r['gap']:+.4f}")
        big = r["gap"] >= MIN_GAP
        if big and r["warned"]:
            print(f"  -> CORROBORATED: split-dependent, and targetleak flagged "
                  f"{r['warned']}")
        elif big:
            print(f"  -> MISS: the random split was worth {r['gap']:.3f} and "
                  "targetleak said nothing")
            misses.append((name, r["gap"]))
        elif r["warned"]:
            print(f"  -> possible false positive: flagged {r['warned']} but the "
                  "split choice barely mattered")
        else:
            print("  -> agreement: no gap, nothing flagged")

    print("\n" + "=" * 68)
    print(f"{ran} datasets measured; misses worth investigating: {len(misses)}")
    for n, gp in misses:
        print(f"    {n}: {gp:+.4f} of unearned score")
    return 0


if __name__ == "__main__":
    sys.exit(main())
