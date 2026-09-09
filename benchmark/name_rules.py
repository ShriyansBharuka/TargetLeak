#!/usr/bin/env python3
"""False-positive control for the name-based rules, against real schemas.

The name rule is the only check that fires on a string rather than on data, so
no amount of sweeping or planting leaks exercises it. That gap has already cost
twice, and both times a human reading output caught it rather than a test:

  * predicting `band_type` on cylinder-bands flagged `paper_type`, `ink_type`
    and `press_type` for sharing the word "type" - five warnings printed beside
    seven real findings
  * predicting `tip_amount` on nyc-taxi flagged `store_and_fwd_flag` as
    CRITICAL, matching "fwd" inside a telecom term of art

Recall benchmarks cannot find either. A rule that fires on everything scores
perfectly on recall. So this measures the other direction: run the name rule
over every (target, column) pair in ~100 real datasets, where the overwhelming
majority of columns are ordinary features, and look at everything it says.

Roughly 24,000 pairs, of which a handful should fire. Each firing is listed in
EXPECTED below with a hand-checked verdict, so a change that introduces a new
one fails instead of being noticed months later by a reader.

  python benchmark/name_rules.py

Exits non-zero on any firing not in EXPECTED, or any EXPECTED one that stopped
firing.
"""
import sys
import warnings

sys.path.insert(0, ".")
sys.path.insert(0, "benchmark")
import targetleak as tl  # noqa: E402
from sweep import DATASETS  # noqa: E402

# Every (dataset, target, column) the rule currently fires on, with why it is
# allowed to. Reviewed one at a time against what the column actually means.
EXPECTED = {
    ("SpeedDating", "match", "expected_num_matches"):
        "FALSE POSITIVE, counted in run_benchmark.py. The column name contains "
        "the target's whole name, which is usually a sibling label - here it is "
        "a survey answer collected before the event. Not tuned away: no rule "
        "reading only names can tell these apart.",
    ("SpeedDating", "match", "d_expected_num_matches"):
        "FALSE POSITIVE, same column binned. Counted separately because a user "
        "sees two findings.",
    ("PhishingWebsites", "Result", "SSLfinal_State"):
        "DEFENSIBLE. Matches 'final_' and asks, at warning severity, whether a "
        "column called 'final state' is known at prediction time. For SSL state "
        "the answer is yes, and confirming that is a five-second job - which is "
        "what a warning is for.",
}


def main():
    fires, checked, done, skipped = {}, 0, 0, 0
    for name, ver in DATASETS:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                from sklearn.datasets import fetch_openml
                b = fetch_openml(name, version=ver, as_frame=True,
                                 parser="pandas")
        except Exception:
            skipped += 1
            continue
        target = b.target.name if b.target is not None else b.frame.columns[-1]
        done += 1
        for c in b.frame.columns:
            if c == target:
                continue
            checked += 1
            hit = tl._suspicious_name(c, target)
            if hit:
                fires[(name, str(target), str(c))] = hit

    print(f"{done} datasets, {skipped} unreachable")
    print(f"{checked:,} real (target, column) pairs checked")
    print(f"name rule fired on {len(fires)} "
          f"({len(fires) / max(checked, 1):.3%})\n")

    for key, (sev, why) in sorted(fires.items()):
        mark = "known" if key in EXPECTED else "*** NEW"
        print(f"  {mark:8} {key[0][:20]:21} target={key[1][:16]:17} "
              f"col={key[2][:26]:27} {sev:8} {why[:52]}")

    new = sorted(k for k in fires if k not in EXPECTED)
    gone = sorted(k for k in EXPECTED if k not in fires)
    print()
    if new:
        print(f"NEW FIRINGS: {len(new)} - review each one, then either fix the "
              "rule or record it in EXPECTED with a verdict")
        for k in new:
            print(f"    {k}")
    if gone:
        print(f"NO LONGER FIRING: {len(gone)} - if that was deliberate, delete "
              "the EXPECTED entry; if not, a real tell was just silenced")
        for k in gone:
            print(f"    {k}")
    if not new and not gone:
        print("every firing accounted for")
    return 1 if new or gone else 0


if __name__ == "__main__":
    sys.exit(main())
