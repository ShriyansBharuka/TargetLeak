#!/usr/bin/env python3
"""Corroborate findings against what a model actually does. Recall evidence.

The sweep and the benchmark both have a hole: neither can tell you targetleak
MISSED something, because nobody has annotated these datasets. This closes part
of that hole without needing ground truth, by using a property of leakage
rather than a label.

The property: **a leak inflates a random split and collapses under a correct
one.** Entity identity leaking across a random split is why the same patient,
user or ticker appearing on both sides makes a model look good; splitting by
that entity removes the free lunch. Time works the same way - train on the
future, predict the past, and the score is fiction.

So for a dataset with an entity or time column:

    A = cross-validated score under a RANDOM split
    B = cross-validated score under a GROUPED or TIME-ORDERED split
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
    ("eucalyptus", 1, "Locality", "trial site; trees at a site share conditions"),
    ("us_crime", 2, "state", "US state; communities within one share policy"),
]

MIN_GAP = 0.03          # below this, the split choice made no real difference
FOLDS = 4


def _model(X):
    """A fast learner that eats mixed dtypes and NaNs without preprocessing.

    Deliberately not tuned. The question is whether the SPLIT changes the
    score, so anything that imputes or one-hot-encodes would add its own
    leakage and confound the measurement.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    cat = [pd.api.types.is_object_dtype(X[c])
           or isinstance(X[c].dtype, pd.CategoricalDtype) for c in X.columns]
    return HistGradientBoostingClassifier(
        max_iter=60, random_state=0,
        categorical_features=cat if any(cat) else None)


def _prep(df, target, group):
    y = df[target]
    X = df.drop(columns=[c for c in (target,) if c in df.columns])
    if group in X.columns:
        X = X.drop(columns=[group])          # the entity id is never a feature
    # HistGB wants numeric or categorical; strings become categories.
    for c in X.columns:
        if pd.api.types.is_object_dtype(X[c]) or X[c].dtype == "string":
            X[c] = X[c].astype("category")
    keep = y.notna()
    return X[keep.to_numpy()], y[keep], df.loc[keep, group]


def _score(X, y, splits):
    from sklearn.metrics import roc_auc_score
    out = []
    classes = pd.Series(y).unique()
    for tr, te in splits:
        if len(np.unique(y.iloc[te])) < 2 or len(np.unique(y.iloc[tr])) < 2:
            continue
        m = _model(X)
        m.fit(X.iloc[tr], y.iloc[tr])
        p = m.predict_proba(X.iloc[te])
        if len(classes) == 2:
            out.append(roc_auc_score(y.iloc[te], p[:, 1]))
        else:
            out.append(roc_auc_score(y.iloc[te], p, multi_class="ovr",
                                     average="macro", labels=m.classes_))
    return float(np.mean(out)) if out else float("nan")


def evaluate(name, version, group):
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import KFold, GroupKFold
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        b = fetch_openml(name, version=version, as_frame=True, parser="pandas")
    df = b.frame
    target = b.target.name
    if group not in df.columns:
        return {"skip": f"no column {group!r}"}

    X, y, g = _prep(df, target, group)
    if X.empty or y.nunique() < 2:
        return {"skip": "nothing to model"}

    random_splits = list(KFold(n_splits=FOLDS, shuffle=True,
                               random_state=0).split(X))
    n_groups = int(g.nunique())
    if n_groups < FOLDS:
        return {"skip": f"only {n_groups} groups"}
    grouped_splits = list(GroupKFold(n_splits=FOLDS).split(X, y, groups=g))

    a = _score(X, y, random_splits)
    bb = _score(X, y, grouped_splits)

    findings = tl.analyse(df, target, group=group)
    warned = sorted({f.column for f in findings
                     if f.kind in ("group-overlap", "identifier-like",
                                   "train-test-contamination")
                     and f.severity in ("critical", "warning")})
    return {"random": a, "grouped": bb, "gap": a - bb,
            "groups": n_groups, "rows": len(X), "warned": warned}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args(argv)

    misses = []
    todo = GROUPED[:a.limit] if a.limit else GROUPED
    for name, ver, group, why in todo:
        print(f"\n{name}  (grouping on {group!r} - {why})")
        try:
            r = evaluate(name, ver, group)
        except Exception as e:
            print(f"  failed: {type(e).__name__}: {str(e)[:80]}")
            continue
        if "skip" in r:
            print(f"  skipped: {r['skip']}")
            continue
        print(f"  {r['rows']:,} rows, {r['groups']} groups")
        print(f"  random-split AUC  {r['random']:.4f}")
        print(f"  grouped-split AUC {r['grouped']:.4f}")
        print(f"  gap               {r['gap']:+.4f}")
        big = r["gap"] >= MIN_GAP
        if big and r["warned"]:
            print(f"  -> CORROBORATED: split-dependent, and targetleak flagged "
                  f"{r['warned']}")
        elif big:
            print("  -> MISS: the random split was worth "
                  f"{r['gap']:.3f} AUC and targetleak said nothing")
            misses.append((name, r["gap"]))
        elif r["warned"]:
            print(f"  -> possible false positive: flagged {r['warned']} but the "
                  "split choice barely mattered")
        else:
            print("  -> agreement: no gap, nothing flagged")

    print("\n" + "=" * 68)
    print(f"misses worth investigating: {len(misses)}")
    for n, gp in misses:
        print(f"    {n}: {gp:+.4f} AUC of unearned score")
    return 0


if __name__ == "__main__":
    sys.exit(main())
