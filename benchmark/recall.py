#!/usr/bin/env python3
"""Recall: plant known leaks in real datasets and count how many are found.

`run_benchmark.py` measures recall against two documented leaks in one
dataset, which is the honest number available - publicly documented tabular
leaks with downloadable data are scarce. That is also far too few to say
anything about which KINDS of leak the tool can see.

So plant the leak instead. The data stays real - real dtypes, real
missingness, real correlations between the other columns - and only the leak
is injected, so ground truth is known by construction and there are dozens of
cases instead of two.

Reported per leak family and per strength, because a single recall percentage
hides a whole family being invisible. That is not hypothetical: the first run
of this scored 66% and showed `reason code (one class)` missed at every
strength below 1.0, across all six datasets. The check meant to catch it was
gated behind the AUC threshold whose dilution it existed to see past.

What this does NOT measure: whether the tool finds leaks nobody thought to
plant. A planted-leak benchmark can only ever confirm the families its author
imagined - so read it alongside `sweep.py`, which is the search for shapes
nobody imagined.

  python benchmark/recall.py

Exits non-zero if any leak family is invisible at full strength.
"""
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
import targetleak as tl  # noqa: E402

# Chosen for the shapes a leak has to survive, not for convenience. Six
# datasets stood behind the first "98%" and every one of them was a modest
# binary classification table - so the number said nothing about a continuous
# target, a 100-class target, a frame that is mostly missing, or 1,776 columns.
DATASETS = [
    # binary, mixed dtypes - where this started
    ("credit-g", 1), ("adult", 2), ("churn", 1), ("kc1", 1),
    ("Australian", 4), ("bank-marketing", 1),
    # imbalanced, where the base-rate gates get stressed
    ("sick", 1), ("ozone-level-8hr", 1), ("click_prediction_small", 1),
    # multiclass, including the class-count ceiling
    ("vehicle", 1), ("segment", 1), ("letter", 1),
    ("one-hundred-plants-margin", 1),
    # continuous targets - a planted leak has to be found by group separation
    # rather than by ranking, which is a different code path entirely
    ("cpu_act", 1), ("us_crime", 2), ("cholesterol", 1),
    # awkward: heavy missingness, extreme width, few rows, many categoricals
    ("anneal", 1), ("Bioresponse", 1), ("dresses-sales", 1),
    ("credit-approval", 1),
]

LOUD = ("target-proxy", "pure-categories", "missingness-leak",
        "suspiciously-predictive", "dead-on-labelled-rows")


def fetch(name, ver, cap=8000):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from sklearn.datasets import fetch_openml
        b = fetch_openml(name, version=ver, as_frame=True, parser="pandas")
    df = b.frame
    if len(df) > cap:
        df = df.sample(cap, random_state=0).reset_index(drop=True)
    return df, b.target.name


def answer(y):
    """The target as a float that a planted column can be computed from.

    This replaced `binarise`, which mapped the target to "is this the modal
    class". On a binary frame those are nearly the same thing, and on anything
    else they are not remotely: it made the planted leak encode one letter of
    26, sixteen rows of 1,600 on a 100-class target, and SIX rows on
    `cholesterol`, whose target has 152 distinct values. Then `analyse` was
    handed the real target and asked to find a leak that was not there. Every
    one of `cholesterol`'s sixteen "misses" was that, not the tool.

    So: class codes for a classification target, the value itself for a
    continuous one. A column built from this really is a copy of the answer.
    """
    if tl._target_kind(y) == "continuous":
        v = pd.to_numeric(y, errors="coerce").astype(float)
        return v.fillna(v.median())
    return pd.Series(pd.factorize(y.astype(str))[0], index=y.index).astype(float)


def n_classes(y):
    return 0 if tl._target_kind(y) == "continuous" else int(y.nunique(dropna=True))


# Each planter returns the column it made, or None when the dataset cannot host
# that leak at all - which is recorded as n/a and left out of the denominator,
# because demanding a finding that is not there measures nothing. `strength` is
# the share of rows on which the leak actually carries the answer, so 1.0 is a
# clean copy and 0.2 is a leak covering a fifth of the data.
def plant_proxy(df, a, rng, strength, y):
    """A column computed from the answer - a refund amount, a risk tier.

    For a classification target each class gets its own value, so this is a
    copy of the WHOLE target rather than of one class. For a continuous target
    it is the value itself, scaled.
    """
    mask = rng.random(len(df)) < strength
    scale = 100.0 if n_classes(y) else 1.0
    noise = rng.normal(float(np.nanmean(a) * scale), float(np.nanstd(a) * scale) + 1.0,
                       len(df))
    df["refund_amount"] = np.where(mask, a * scale, noise)
    return "refund_amount"


def plant_reason_code(df, a, rng, strength, y):
    """A category filled in from the outcome - the commonest real leak.

    One value per class, so the categories partition the target. On a
    continuous target the value is the decile, which is the same shape of
    mistake a data engineer actually makes when bucketing an outcome.
    """
    k = n_classes(y)
    if k:
        if len(df) / k < 2 * tl.MIN_CATEGORY_SUPPORT:
            return None      # too few rows per class to support any category
        key = a
    else:
        key = pd.qcut(a, 10, labels=False, duplicates="drop").astype(float)
    mask = rng.random(len(df)) < strength
    col = np.where(mask, ["reason_" + str(int(v)) for v in key], "pending")
    df["cancellation_reason"] = col
    return "cancellation_reason"


def plant_missingness(df, a, rng, strength, y):
    """A measurement that was never taken once the outcome was known.

    A NaN pattern is one bit, so it can only ever encode a binary split of the
    target - it cannot give away a 100-class label, and asking it to would be
    measuring nothing. So the leak marks one class, and the dataset has to have
    enough rows in that class for the absence to be evidence at all. On
    `one-hundred-plants-margin` that is 16 rows against a support floor of 20,
    which is the tool being right, and the old version of this counted all four
    strengths there as misses.
    """
    k = n_classes(y)
    if k:
        counts = pd.Series(a).value_counts()
        target_val = counts.index[0]
        if counts.iloc[0] * strength < 2 * tl.MIN_CATEGORY_SUPPORT:
            return None
        hit = (a == target_val).to_numpy()
    else:
        hit = (a > a.quantile(0.75)).to_numpy()
    mask = hit & (rng.random(len(df)) < strength)
    col = rng.normal(10, 3, len(df))
    col[mask] = np.nan
    df["final_reading"] = col
    return "final_reading"


def plant_noisy_proxy(df, a, rng, strength, y):
    """The answer plus enough noise to look like a real feature."""
    sd = float(np.nanstd(a)) or 1.0
    d = (1.0 + 4.0 * strength) * sd / 2.0
    df["risk_score"] = (a / sd) * d + rng.normal(0, 1, len(df))
    return "risk_score"


PLANTERS = [("target proxy (exact)", plant_proxy),
            ("reason code (per class)", plant_reason_code),
            ("missingness", plant_missingness),
            ("noisy proxy", plant_noisy_proxy)]
STRENGTHS = [1.0, 0.7, 0.4, 0.2]


# --- leaks that live in the SPLIT rather than in a column -------------------
#
# These have the thinnest real-world exposure of anything in the tool: they
# only run when the caller passes --split, so no amount of sweeping unlabelled
# datasets exercises them. Each planter builds a `_split` column itself,
# returns the finding kind it should produce, and is checked by kind because
# contamination is a statement about the frame rather than about a column.

def plant_contamination(df, a, rng, strength, tgt):
    """Rows copied from train into test - the classic duplicated-row split."""
    n = len(df)
    is_test = rng.random(n) < 0.25
    df["_split"] = np.where(is_test, "test", "train")
    train_idx = np.flatnonzero(~is_test)
    test_idx = np.flatnonzero(is_test)
    take = max(int(len(test_idx) * strength), 1)
    src = rng.choice(train_idx, size=take, replace=False)
    dst = test_idx[:take]
    feat = [c for c in df.columns if c != "_split"]
    df.loc[df.index[dst], feat] = df.loc[df.index[src], feat].to_numpy()
    return "train-test-contamination", tgt


def plant_group_overlap(df, a, rng, strength, tgt):
    """An entity whose identity predicts the target, split at random.

    The leak is not that the entity spans the split - under a random split
    everything does. It is that knowing WHICH entity tells you the answer, so
    the same entity being on both sides hands the model the target.
    """
    n = len(df)
    n_ent = max(n // 30, 20)
    ent = rng.integers(0, n_ent, n)
    # Entity identity carries the target for `strength` of the rows: rows whose
    # entity id is even are positive, the rest keep their real label.
    flip = rng.random(n) < strength
    # This one builds its own binary target on purpose, so the entity's
    # identity is the only thing that decides it for `strength` of the rows.
    base = (pd.Series(a) > pd.Series(a).median()).astype(float).to_numpy()
    y_new = np.where(flip, (ent % 2 == 0).astype(float), base)
    df["account_id"] = [f"ACC{e:05d}" for e in ent]
    # The real target goes, or it sits in the frame as a perfect proxy for the
    # one we just built and every finding is about that instead.
    del df[tgt]
    df["_leaky_y"] = y_new
    df["_split"] = np.where(rng.random(n) < 0.25, "test", "train")
    return "group-overlap", "_leaky_y"


def plant_temporal(df, a, rng, strength, tgt):
    """A real date column under a random split: training on the future."""
    n = len(df)
    days = np.sort(rng.integers(0, 900, n))
    df["signup_date"] = pd.to_datetime("2023-01-01") + pd.to_timedelta(days, "D")
    df["_split"] = np.where(rng.random(n) < 0.25, "test", "train")
    return "temporal-column", tgt


SPLIT_PLANTERS = [("train/test contamination", plant_contamination, [1.0, 0.5, 0.1]),
                  ("entity identity leak", plant_group_overlap, [1.0, 0.6]),
                  ("date under a random split", plant_temporal, [1.0])]


# --- controls: the same shapes with no leak in them -------------------------
#
# Recall on its own rewards a check that fires on everything, so each split
# family gets an innocent twin that must stay quiet. This is not a hypothetical
# guard: `group-overlap` used to fire on 280+ columns of KDD98 by testing
# whether values straddled the split, a quantity fixed by rows-per-value that
# carries no information at all. A recall-only benchmark would have called that
# version perfect.

def control_innocent_entity(df, a, rng, strength, tgt):
    """An entity id spanning the split whose identity says nothing.

    The column matters, not just the kind. An earlier version of this control
    asked only whether `group-overlap` fired anywhere in the report, and
    counted six datasets as false alarms - but the findings were on `state`,
    `CIRCULARITY` and `vedge-mean`, never on the planted id. `us_crime`'s
    `state` has a model-measured +0.0385 split gap, so flagging it is the
    check working, and the control was calling a true positive a failure.
    """
    n = len(df)
    ent = rng.integers(0, max(n // 30, 20), n)
    df["account_id"] = [f"ACC{e:05d}" for e in ent]
    df["_split"] = np.where(rng.random(n) < 0.25, "test", "train")
    return "group-overlap", tgt, None, "account_id"


def control_clean_split(df, a, rng, strength, tgt):
    """A split with no shared rows. Contamination must not be reported.

    The duplicates have to go first, and finding that out is why this control
    exists. Its first run reported a critical contamination on kc1 and looked
    like a false alarm - kc1 genuinely contains 897 duplicated rows in 2,109,
    so a random split really does put identical feature vectors on both sides
    and the finding was correct. The control was wrong, not the tool.
    """
    df = df.drop_duplicates(subset=[c for c in df.columns if c != tgt])
    df = df.reset_index(drop=True)
    df["_split"] = np.where(rng.random(len(df)) < 0.25, "test", "train")
    return "train-test-contamination", tgt, df


CONTROLS = [("innocent entity id", control_innocent_entity, [0.0]),
            ("clean split", control_clean_split, [0.0])]


def main():
    rows = []
    for name, ver in DATASETS:
        try:
            df0, tgt = fetch(name, ver)
        except Exception as e:
            print(f"  skip {name}: {type(e).__name__}")
            continue
        a = answer(df0[tgt])
        kind = tl._target_kind(df0[tgt])
        print(f"  {name} ({kind}, {n_classes(df0[tgt]) or 'continuous'})")
        for label, plant in PLANTERS:
            for s in STRENGTHS:
                rng = np.random.default_rng(0)
                df = df0.copy()
                col = plant(df, a, rng, s, df0[tgt])
                if col is None:
                    # The dataset cannot host this leak - too few rows per
                    # class for the evidence to exist. Not a miss.
                    rows.append((label, s, name, "n/a"))
                    continue
                try:
                    out = tl.analyse(df, tgt)
                except Exception as e:
                    rows.append((label, s, name, f"CRASH {type(e).__name__}"))
                    continue
                hit = [f for f in out if f.column == col and f.kind in LOUD]
                sev = ("critical" if any(f.severity == "critical" for f in hit)
                       else "warning" if hit else "MISSED")
                rows.append((label, s, name, sev))

        for label, plant, strengths in list(SPLIT_PLANTERS) + list(CONTROLS):
            for s in strengths:
                rng = np.random.default_rng(0)
                df = df0.copy()
                try:
                    # A planter may hand back a replacement frame when it has
                    # to change the row set rather than just add columns.
                    got = plant(df, a, rng, s, tgt)
                    want, use_tgt = got[0], got[1]
                    df = got[2] if len(got) > 2 and got[2] is not None else df
                    # A fourth element pins the finding to one column. Without
                    # it a control passes or fails on whether the KIND appeared
                    # anywhere, which counted a correct finding on another
                    # column as a false alarm.
                    want_col = got[3] if len(got) > 3 else None
                    out = tl.analyse(df, use_tgt, split="_split",
                                     group="account_id"
                                     if "account_id" in df.columns else None)
                except Exception as e:
                    rows.append((label, s, name, f"CRASH {type(e).__name__}"))
                    continue
                hit = [f for f in out if f.kind == want
                       and (want_col is None or f.column == want_col)]
                sev = ("critical" if any(f.severity == "critical" for f in hit)
                       else "warning" if hit else "MISSED")
                rows.append((label, s, name, sev))
        print(f"  done {name}")

    def table(planters, header, quiet_is_good=False):
        print(f"\n{header}")
        print(f"{'leak family':28}{'strength':>9}{'critical':>10}{'warning':>9}"
              f"{'MISSED':>8}{'n/a':>6}{'CRASH':>7}")
        for entry in planters:
            label = entry[0]
            for s in (entry[2] if len(entry) > 2 else STRENGTHS):
                sub = [r[3] for r in rows if r[0] == label and r[1] == s]
                if not sub:
                    continue
                c, w = sub.count("critical"), sub.count("warning")
                m, na = sub.count("MISSED"), sub.count("n/a")
                x = sum(1 for v in sub if v.startswith("CRASH"))
                if quiet_is_good:
                    flag = "   ok" if not (c or w) else "   <-- FIRED"
                else:
                    flag = "   <-- blind" if m and not (c or w) else ""
                print(f"{label:28}{s:>9.1f}{c:>10}{w:>9}{m:>8}{na:>6}{x:>7}{flag}")

    table(PLANTERS, "leaks planted in a column")
    table(SPLIT_PLANTERS, "leaks planted in the split (analyse(..., split=...))")
    table(CONTROLS, "controls - the same shapes with no leak, must stay quiet",
          quiet_is_good=True)

    control_labels = {label for label, _, _ in CONTROLS}
    # n/a leaves the denominator. A dataset that cannot host a leak was never a
    # test of whether the leak gets found, and counting it as a miss understates
    # the tool exactly as counting it as a hit would overstate it.
    planted = [r for r in rows
               if r[0] not in control_labels and r[3] != "n/a"]
    skipped = sum(1 for r in rows
                  if r[0] not in control_labels and r[3] == "n/a")
    caught = sum(1 for r in planted if r[3] in ("critical", "warning"))
    print(f"\ndetected {caught}/{len(planted)} planted leaks "
          f"({caught / max(len(planted), 1):.0%}) across {len(DATASETS)} datasets"
          + (f"; {skipped} scenarios n/a - the dataset cannot host that leak"
             if skipped else ""))
    misses = [r for r in planted if r[3] not in ("critical", "warning")]
    if misses:
        print("\nnot found:")
        for label, s, ds, why in misses:
            print(f"    {label:28} strength {s:.1f}  {ds:18} {why}")
    alarms = [r for r in rows
              if r[0] in control_labels and r[3] in ("critical", "warning")]
    print(f"false alarms on controls: {len(alarms)}")
    for label, _, ds, sev in alarms:
        print(f"    {label:28} {ds:18} {sev}")

    # A family invisible at full strength is a hole, not a threshold choice.
    # Weak plants are allowed to be missed - a leak covering 20% of rows with
    # noise over the rest genuinely is near the edge of what one column can
    # show - but a clean copy of the answer must never be silent. A crash is
    # always a failure, at any strength.
    all_families = [(e[0], e[2] if len(e) > 2 else STRENGTHS)
                    for e in list(PLANTERS) + list(SPLIT_PLANTERS)]
    blind = [label for label, ss in all_families
             if rows and all(r[3] == "MISSED"
                             for r in rows if r[0] == label and r[1] == max(ss))]
    crashed = sorted({r[0] for r in rows if r[3].startswith("CRASH")})
    if blind:
        print(f"\nINVISIBLE AT FULL STRENGTH: {blind}")
    if crashed:
        print(f"CRASHED: {crashed}")
    if alarms:
        print("CONTROL FIRED: a check that reports a leak in data with none")
    return 1 if blind or crashed or alarms else 0


if __name__ == "__main__":
    sys.exit(main())
