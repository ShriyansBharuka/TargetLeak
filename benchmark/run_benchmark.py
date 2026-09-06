#!/usr/bin/env python3
"""Measure targetleak against datasets whose leakage status is known already.

Synthetic tests where the same author writes the leak and the detector prove
nothing about a detector. This runs against data nobody here constructed:

POSITIVES - documented leaks
    Titanic (OpenML id 40945) carries `boat` and `body`. A lifeboat number
    exists only for people who got into a lifeboat; a body-recovery number
    exists only for people who did not survive. Both are recorded after the
    outcome, both are the standard worked example of leakage, and neither was
    planted by us. Requires network.

NEGATIVES - the harder half
    iris, wine, breast_cancer, digits, diabetes ship with scikit-learn and are
    among the most-studied datasets in the field. If they contained leakage it
    would be famous. Every critical finding here is a false positive, and a
    detector's false-positive rate is what decides whether anyone keeps it
    installed.

    They are also a fair test of the tool's blind spot: iris and wine are
    genuinely easy, and a single feature really does almost separate a class.
    That is indistinguishable from a leak by measurement alone. The benchmark
    reports it as a false positive rather than explaining it away.

  python benchmark/run_benchmark.py            # table to stdout
  python benchmark/run_benchmark.py --markdown # README-ready
"""
import argparse
import sys
import warnings

import pandas as pd

sys.path.insert(0, ".")
import targetleak as tl  # noqa: E402

LEAK_KINDS = {"target-proxy", "pure-categories", "missingness-leak",
              "suspicious-name", "train-test-contamination"}


class Case:
    def __init__(self, name, target, leaks=(), ids=(), note="", needs_net=False):
        self.name = name
        self.target = target
        self.leaks = set(leaks)      # documented leakage, must be found
        self.ids = set(ids)          # identifiers: flagging them is correct
        self.note = note
        self.needs_net = needs_net
        self.frame = None


def _sk(loader, target="target"):
    from sklearn import datasets
    bunch = getattr(datasets, loader)(as_frame=True)
    df = bunch.frame.copy()
    if target not in df.columns:
        df[target] = bunch.target
    return df


def build_cases(with_network=True):
    cases = []

    titanic = Case(
        "titanic", "survived",
        leaks={"boat", "body"},
        ids={"name", "ticket", "cabin", "home.dest"},
        note="documented leaks: lifeboat and body-recovery numbers",
        needs_net=True)
    if with_network:
        try:
            from sklearn.datasets import fetch_openml
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                frame = fetch_openml("titanic", version=1, as_frame=True,
                                     parser="pandas").frame
            titanic.frame = frame
            cases.append(titanic)
        except Exception as e:  # offline: say so rather than silently skip
            print(f"  (skipping titanic: {type(e).__name__})", file=sys.stderr)

    # Real datasets, fetched not constructed. Eight of them are ordinary
    # supervised-learning sets in wide use with no leakage anyone has
    # reported, so a critical finding on any of them counts against us -
    # and unlike iris and wine they carry the messiness of real data:
    # pandas `category` dtype, heavy missingness, 121-column frames,
    # imbalanced targets. This is the negative set that actually matters.
    real = [
        ("credit-g", 1, "categorical-heavy, 1k rows"),
        ("adult", 2, "48k rows, mixed dtypes"),
        ("kc1", 1, "software metrics, all float"),
        ("bank-marketing", 1, "45k rows, 10 category columns"),
        ("Australian", 4, "small and categorical"),
        ("dresses-sales", 1, "500 rows, 12 category columns, missingness"),
        ("SpeedDating", 1, "121 columns"),
        ("churn", 1, "imbalanced binary target"),
    ]
    if with_network:
        for name, ver, note in real:
            c = Case(name, None, note=note, needs_net=True)
            try:
                from sklearn.datasets import fetch_openml
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    b = fetch_openml(name, version=ver, as_frame=True,
                                     parser="pandas")
                c.frame, c.target = b.frame, b.target.name
                cases.append(c)
            except Exception as e:
                print(f"  (skipping {name}: {type(e).__name__})", file=sys.stderr)

    clean = [
        ("iris", "load_iris", "genuinely easy: one feature nearly splits a class"),
        ("wine", "load_wine", "genuinely easy, 13 features, 3 classes"),
        ("breast_cancer", "load_breast_cancer", "no id column in the sklearn copy"),
        ("digits", "load_digits", "64 pixels, several always blank"),
        ("diabetes", "load_diabetes", "continuous target"),
    ]
    for name, loader, note in clean:
        c = Case(name, "target", note=note)
        try:
            c.frame = _sk(loader)
            cases.append(c)
        except Exception as e:
            print(f"  (skipping {name}: {type(e).__name__})", file=sys.stderr)
    return cases


def evaluate(case):
    findings = tl.analyse(case.frame, case.target)
    flagged = {f.column for f in findings
               if f.kind in LEAK_KINDS and f.severity in ("critical", "warning")}
    critical = {f.column for f in findings if f.severity == "critical"}
    ided = {f.column for f in findings if f.kind == "identifier-like"}

    found = case.leaks & flagged
    missed = case.leaks - flagged
    # An identifier reported as an identifier is a correct call, not a miss.
    excused = case.leaks | case.ids
    false_pos = {c for c in critical if c and c not in excused}
    return {
        "case": case, "n": len(case.frame),
        "cols": len(case.frame.columns) - 1,
        "found": found, "missed": missed,
        "false_pos": false_pos, "ids": ided & case.ids,
        "n_critical": len(critical),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--markdown", action="store_true")
    ap.add_argument("--offline", action="store_true",
                    help="skip datasets that need a download")
    a = ap.parse_args(argv)

    cases = build_cases(with_network=not a.offline)
    if not cases:
        # Without this, sklearn missing meant "0 false positives across 0
        # columns" and exit 0 - a green benchmark measured on nothing, and
        # this is the evidence the README cites.
        print("no datasets loaded - install scikit-learn "
              "(pip install '.[benchmark]')", file=sys.stderr)
        return 2
    results = [evaluate(c) for c in cases]

    pos = [r for r in results if r["case"].leaks]
    neg = [r for r in results if not r["case"].leaks]
    tp = sum(len(r["found"]) for r in pos)
    fn = sum(len(r["missed"]) for r in pos)
    fp_clean = sum(len(r["false_pos"]) for r in neg)
    clean_cols = sum(r["cols"] for r in neg)

    if a.markdown:
        print("| dataset | rows | cols | known leaks | found | false positives |")
        print("|---|---:|---:|---|---|---:|")
        for r in results:
            leaks = ", ".join(f"`{c}`" for c in sorted(r["case"].leaks)) or "-"
            found = ", ".join(f"`{c}`" for c in sorted(r["found"])) or "-"
            print(f"| {r['case'].name} | {r['n']:,} | {r['cols']} | {leaks} | "
                  f"{found} | {len(r['false_pos'])} |")
    else:
        for r in results:
            c = r["case"]
            print(f"\n{c.name}  ({r['n']:,} rows x {r['cols']} cols)  -- {c.note}")
            if c.leaks:
                print(f"  documented leaks : {sorted(c.leaks)}")
                print(f"  found            : {sorted(r['found']) or 'NONE'}")
                if r["missed"]:
                    print(f"  MISSED           : {sorted(r['missed'])}")
            if r["ids"]:
                print(f"  identifiers named: {sorted(r['ids'])}")
            verdict = sorted(r["false_pos"]) or "none"
            print(f"  false positives  : {verdict}")

    print()
    print("=" * 62)
    if pos:
        print(f"recall on documented leaks : {tp}/{tp + fn}")
    print(f"false positives on clean data: {fp_clean} "
          f"across {clean_cols} columns in {len(neg)} datasets")
    return 0


if __name__ == "__main__":
    sys.exit(main())
