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
        print(f"  done {name}")

    print()
    print(f"{'leak family':26}{'strength':>9}{'critical':>10}{'warning':>9}"
          f"{'MISSED':>8}")
    for label, _ in PLANTERS:
        for s in STRENGTHS:
            sub = [r[3] for r in rows if r[0] == label and r[1] == s]
            if not sub:
                continue
            c = sub.count("critical")
            w = sub.count("warning")
            m = sub.count("MISSED")
            flag = "   <-- blind" if m and not (c or w) else ""
            print(f"{label:26}{s:>9.1f}{c:>10}{w:>9}{m:>8}{flag}")
    caught = sum(1 for r in rows if r[3] in ("critical", "warning"))
    print(f"\ndetected {caught}/{len(rows)} planted leaks "
          f"({caught / max(len(rows), 1):.0%}) across {len(DATASETS)} datasets")
    misses = [r for r in rows if r[3] == "MISSED"]
    if misses:
        print("\nmisses:")
        for label, s, ds, _ in misses:
            print(f"    {label:26} strength {s:.1f}  {ds}")

    # A family invisible at full strength is a hole, not a threshold choice.
    # Weak plants are allowed to be missed - a leak covering 20% of rows with
    # noise over the rest genuinely is near the edge of what one column can
    # show - but a clean copy of the answer must never be silent.
    blind = [label for label, _ in PLANTERS
             if rows and all(r[3] == "MISSED"
                             for r in rows if r[0] == label and r[1] == 1.0)]
    if blind:
        print(f"\nINVISIBLE AT FULL STRENGTH: {blind}")
    return 1 if blind else 0


if __name__ == "__main__":
    sys.exit(main())
