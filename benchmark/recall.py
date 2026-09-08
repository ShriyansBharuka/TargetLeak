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

DATASETS = [("credit-g", 1), ("adult", 2), ("churn", 1), ("kc1", 1),
            ("Australian", 4), ("bank-marketing", 1)]

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


def binarise(y):
    """A 0/1 float so leak injection is arithmetic, whatever the source dtype."""
    v = pd.Series(pd.factorize(y.astype(str))[0], index=y.index)
    return (v == v.mode()[0]).astype(float)


# Each planter returns the name of the column it made. `strength` is the share
# of rows where the leak actually carries the answer; the rest is noise, so
# 1.0 is a clean copy and 0.3 is a leak that only covers part of the data.
def plant_proxy(df, y01, rng, strength):
    """A column computed from the answer, e.g. a refund amount.

    OPEN: this family is missed below strength 0.7, and the reason is worth
    stating rather than presenting as a threshold choice. Below that, 400-odd
    rows hold exactly 0.0 or exactly 100.0 - pure sub-populations of a NUMERIC
    column - while the rest is noise around 50. That is the pure-category case
    in numeric clothing, and the purity check only runs on categoricals.

    Part of the miss is this planter's fault: overwriting the leak with NOISE
    is not how a partial proxy occurs in the wild. Real ones leave NaN on the
    uncovered rows (the missingness check catches those, at every strength
    here) or leave a zero. So do not read this row as "targetleak misses 60% of
    proxies" - read it as one unimplemented check and one artificial planter.
    """
    mask = rng.random(len(df)) < strength
    col = np.where(mask, y01 * 100.0, rng.normal(50, 30, len(df)))
    df["refund_amount"] = col
    return "refund_amount"


def plant_reason_code(df, y01, rng, strength):
    """A category only filled in for one class - the commonest real leak."""
    mask = rng.random(len(df)) < strength
    col = np.where((y01 == 1) & mask, "churn_reason_given", "not_applicable")
    other = rng.random(len(df)) < 0.25
    col = np.where((y01 == 0) & other, "no_contact", col)
    df["cancellation_reason"] = col
    return "cancellation_reason"


def plant_missingness(df, y01, rng, strength):
    """A measurement that was never taken when the outcome occurred."""
    mask = (y01 == 1) & (rng.random(len(df)) < strength)
    col = rng.normal(10, 3, len(df))
    col[mask] = np.nan
    df["final_reading"] = col
    return "final_reading"


def plant_noisy_proxy(df, y01, rng, strength):
    """The answer plus enough noise to look like a real feature."""
    d = 1.0 + 4.0 * strength          # separation in SDs
    df["risk_score"] = y01 * d + rng.normal(0, 1, len(df))
    return "risk_score"


PLANTERS = [("target proxy (exact)", plant_proxy),
            ("reason code (one class)", plant_reason_code),
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

def plant_contamination(df, y01, rng, strength, tgt):
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


def plant_group_overlap(df, y01, rng, strength, tgt):
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
    y_new = np.where(flip, (ent % 2 == 0).astype(float), y01)
    df["account_id"] = [f"ACC{e:05d}" for e in ent]
    # The real target goes, or it sits in the frame as a perfect proxy for the
    # one we just built and every finding is about that instead.
    del df[tgt]
    df["_leaky_y"] = y_new
    df["_split"] = np.where(rng.random(n) < 0.25, "test", "train")
    return "group-overlap", "_leaky_y"


def plant_temporal(df, y01, rng, strength, tgt):
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

def control_innocent_entity(df, y01, rng, strength, tgt):
    """An entity id spanning the split whose identity says nothing."""
    n = len(df)
    ent = rng.integers(0, max(n // 30, 20), n)
    df["account_id"] = [f"ACC{e:05d}" for e in ent]
    df["_split"] = np.where(rng.random(n) < 0.25, "test", "train")
    return "group-overlap", tgt


def control_clean_split(df, y01, rng, strength, tgt):
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
        y01 = binarise(df0[tgt])
        for label, plant in PLANTERS:
            for s in STRENGTHS:
                rng = np.random.default_rng(0)
                df = df0.copy()
                col = plant(df, y01.to_numpy(), rng, s)
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
                    got = plant(df, y01.to_numpy(), rng, s, tgt)
                    want, use_tgt = got[0], got[1]
                    df = got[2] if len(got) > 2 else df
                    out = tl.analyse(df, use_tgt, split="_split",
                                     group="account_id"
                                     if "account_id" in df.columns else None)
                except Exception as e:
                    rows.append((label, s, name, f"CRASH {type(e).__name__}"))
                    continue
                hit = [f for f in out if f.kind == want]
                sev = ("critical" if any(f.severity == "critical" for f in hit)
                       else "warning" if hit else "MISSED")
                rows.append((label, s, name, sev))
        print(f"  done {name}")

    def table(planters, header, quiet_is_good=False):
        print(f"\n{header}")
        print(f"{'leak family':28}{'strength':>9}{'critical':>10}{'warning':>9}"
              f"{'MISSED':>8}{'CRASH':>7}")
        for entry in planters:
            label = entry[0]
            for s in (entry[2] if len(entry) > 2 else STRENGTHS):
                sub = [r[3] for r in rows if r[0] == label and r[1] == s]
                if not sub:
                    continue
                c, w = sub.count("critical"), sub.count("warning")
                m = sub.count("MISSED")
                x = sum(1 for v in sub if v.startswith("CRASH"))
                if quiet_is_good:
                    flag = "   ok" if not (c or w) else "   <-- FIRED"
                else:
                    flag = "   <-- blind" if m and not (c or w) else ""
                print(f"{label:28}{s:>9.1f}{c:>10}{w:>9}{m:>8}{x:>7}{flag}")

    table(PLANTERS, "leaks planted in a column")
    table(SPLIT_PLANTERS, "leaks planted in the split (analyse(..., split=...))")
    table(CONTROLS, "controls - the same shapes with no leak, must stay quiet",
          quiet_is_good=True)

    control_labels = {label for label, _, _ in CONTROLS}
    planted = [r for r in rows if r[0] not in control_labels]
    caught = sum(1 for r in planted if r[3] in ("critical", "warning"))
    print(f"\ndetected {caught}/{len(planted)} planted leaks "
          f"({caught / max(len(planted), 1):.0%}) across {len(DATASETS)} datasets")
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
