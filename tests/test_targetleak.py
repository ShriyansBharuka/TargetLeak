"""Tests for targetleak.

Two of these exist because the bug they cover actually shipped and was caught
on real data, not because a coverage target asked for them:

- test_unlabelled_rows_are_not_the_negative_class
- test_dates_survive_a_csv_round_trip

Both passed in-memory while the tool was broken on real input. Read their
docstrings before relaxing either.
"""
import json

import numpy as np
import pandas as pd
import pytest

import targetleak as tl


@pytest.fixture
def demo():
    return tl.demo_frame()


# --- the core promise: find planted leaks, stay quiet about honest features ---

def test_finds_numeric_target_proxy(demo):
    crit = {f.column for f in tl.analyse(demo, "churned") if f.severity == "critical"}
    assert "refund_amount" in crit


def test_finds_categorical_target_proxy(demo):
    crit = {f.column for f in tl.analyse(demo, "churned") if f.severity == "critical"}
    assert "cancellation_reason" in crit


def test_finds_identifier_and_date_columns(demo):
    kinds = {(f.kind, f.column) for f in tl.analyse(demo, "churned")}
    assert ("identifier-like", "customer_id") in kinds
    assert ("temporal-column", "signup_date") in kinds


def test_does_not_flag_an_honestly_strong_feature(demo):
    """The half that makes it a product. A detector that flags everything is
    worthless, so `signal` (AUC ~0.79) must stay out of the findings."""
    loud = {f.column for f in tl.analyse(demo, "churned")
            if f.severity in ("critical", "warning")}
    assert "signal" not in loud
    assert "noise" not in loud


def test_score_separates_honest_from_leaky(demo):
    honest = tl._score_column(demo["signal"], demo["churned"], "binary")
    leaky = tl._score_column(demo["refund_amount"], demo["churned"], "binary")
    assert 0.70 < honest["score"] < tl.AUC_WARN, honest
    assert leaky["score"] > tl.AUC_CRITICAL, leaky


def test_float_features_are_not_mistaken_for_identifiers():
    """Continuous floats are ~100% unique. An earlier uniqueness-only rule
    flagged every numeric column as an ID and skipped scoring it entirely."""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 800)
    df = pd.DataFrame({"amount": y * 50.0 + rng.normal(0, 0.3, 800), "y": y})
    out = tl.analyse(df, "y")
    assert not any(f.kind == "identifier-like" for f in out)
    assert "amount" in {f.column for f in out if f.severity == "critical"}


# --- out-of-fold encoding ----------------------------------------------------

def test_high_cardinality_innocent_column_does_not_leak():
    """In-fold target encoding scores any high-cardinality column ~1.0, because
    each category predicts its own mean. Out-of-fold is what prevents that."""
    rng = np.random.default_rng(1)
    n = 1000
    df = pd.DataFrame({"city": rng.choice([f"city{i}" for i in range(120)], n),
                       "y": rng.integers(0, 2, n)})
    assert tl._score_column(df["city"], df["y"], "binary")["score"] < tl.AUC_WARN


# --- name heuristics ---------------------------------------------------------

def test_names_catch_leaks_statistics_miss():
    """A 5-day forward label is future information even when it correlates only
    mildly with the target, so no AUC threshold can catch it. Found on real
    data where seven label_* columns scored just 0.60-0.76."""
    rng = np.random.default_rng(2)
    n = 600
    df = pd.DataFrame({
        "label_cs_5d": rng.normal(size=n),
        "future_return": rng.normal(size=n),
        "post_call_notes": rng.normal(size=n),
        "honest": rng.normal(size=n),
        "y": rng.integers(0, 2, n),
    })
    by_col = {f.column: f for f in tl.analyse(df, "y") if f.kind == "suspicious-name"}
    assert by_col["label_cs_5d"].severity == "critical"
    assert by_col["future_return"].severity == "critical"
    assert by_col["post_call_notes"].severity == "warning"
    assert "honest" not in by_col


@pytest.mark.parametrize("column,target,expected", [
    # Sibling labels: the column carries the target's whole name.
    ("label_cs_5d", "label_cs", "critical"),
    ("regional_sales_target", "sales_target", "critical"),
    # The project's label prefix stays evidence even though the target shares
    # it - getting this backwards silenced six real labels on live data.
    ("label_sector_residual", "label_cs", "critical"),
    ("Target", "label_cs", "critical"),
    # ...but a target that IS the generic word tells you nothing about others.
    ("customer_target_group", "target", None),
    ("region", "target", None),
    ("customer_target_group", "churned", "critical"),  # here 'target' is a tell
    ("revenue_last_year", "revenue", None),           # a legitimate lag
    ("sales_rolling_28d", "sales", None),             # ...and a rolling window
    ("churn_reason", "churned", "warning"),           # shares a stem
    ("plain_feature", "churned", None),
])
def test_suspicious_name_matching(column, target, expected):
    got = tl._suspicious_name(column, target)
    assert (got[0] if got else None) == expected


# --- null targets: the bug that only real data exposed -----------------------

def test_target_mostly_null_is_reported():
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"f": rng.normal(size=600),
                       "t": [1, 0] * 25 + [None] * 550})
    assert any(f.kind == "target-mostly-null" for f in tl.analyse(df, "t"))


def test_unlabelled_rows_are_not_the_negative_class():
    """Comparing a NaN target to a class value yields False, which silently
    relabels every unlabelled row as the negative class. A column that predicts
    *which rows have labels* then scores like a catastrophic leak.

    Seen on a real 448k-row dataset: a ticker column hit AUC 0.93 purely
    because only 26 of 127 tickers were labelled at all. `cohort` below is
    pure noise among the labelled rows and must stay silent.
    """
    rng = np.random.default_rng(4)
    m = 4000
    t = np.full(m, np.nan)
    t[:400] = rng.integers(0, 2, 400)
    df = pd.DataFrame({
        "cohort": np.where(np.arange(m) < 400, "labelled", "unlabelled"),
        "junk": rng.normal(size=m),
        "t": t,
    })
    out = tl.analyse(df, "t")
    # Scoped to leakage kinds on purpose: `cohort` legitimately earns a
    # dead-on-labelled-rows warning (it is constant among labelled rows). What
    # must never happen is it being scored as a *leak*.
    leaky = {f.column for f in out
             if f.kind in ("target-proxy", "suspiciously-predictive",
                           "pure-categories", "missingness-leak")}
    assert "cohort" not in leaky, f"scored labelled-ness as leakage: {leaky}"
    assert any(f.kind == "target-mostly-null" for f in out)


def test_too_few_labelled_rows_raises():
    df = pd.DataFrame({"f": range(100), "t": [1, 0] + [None] * 98})
    with pytest.raises(ValueError, match="target value"):
        tl.analyse(df, "t")


def test_dead_on_labelled_rows_outranks_a_plain_constant():
    """A column that varies in the file but is constant on every labelled row
    is a broken join, not a dead column. On real data this was 17 features -
    an entire ingestion pipeline - and it was filed as a bare info note."""
    rng = np.random.default_rng(21)
    n = 600
    t = np.full(n, np.nan)
    t[:300] = rng.integers(0, 2, 300)
    df = pd.DataFrame({
        "insider_flow": np.concatenate([np.zeros(300), rng.normal(size=300)]),
        "always_empty": np.zeros(n),
        "ok": rng.normal(size=n),
        "t": t,
    })
    by_kind = {f.kind: f for f in tl.analyse(df, "t")}
    assert by_kind["dead-on-labelled-rows"].column == "insider_flow"
    assert by_kind["dead-on-labelled-rows"].severity == "warning"
    assert by_kind["constant"].column == "always_empty"
    assert by_kind["constant"].severity == "info"


# --- bugs found by auditing the audited code ---------------------------------

def test_null_se_uses_only_the_rows_actually_scored():
    """`_auc` drops rows where the score is NaN, but the null SE was taken
    from the full target. The two disagreed by exactly the column's
    missingness, always in the anti-conservative direction: six observed
    values among a thousand rows were reported as 27 SE above chance."""
    rng = np.random.default_rng(0)
    n = 1000
    y = pd.Series(rng.integers(0, 2, n))
    col = pd.Series(np.nan, index=range(n))
    pos = np.flatnonzero(y.to_numpy() == 1)[:3]
    neg = np.flatnonzero(y.to_numpy() == 0)[:3]
    col[list(pos) + list(neg)] = [9.0, 8, 7, 3, 2, 1]   # perfect on 6 rows
    m = tl._score_column(col, y, "binary")
    assert m["score"] == 1.0, "should still separate perfectly"
    assert m["z"] < 3.5, f"six rows cannot be 3.5 SE of evidence (z={m['z']})"
    assert not [f for f in tl.analyse(pd.DataFrame({"c": col, "y": y}), "y")
                if f.severity == "critical"]


def test_subset_purity_must_beat_the_base_rate():
    """'All 44 of these rows are negative' is not evidence when 99% of every
    row is negative - P = 0.99^44 = 0.64. Without the test this fired on 9 of
    15 pure-noise columns."""
    rng = np.random.default_rng(5)
    n = 3000
    y = (rng.random(n) < 0.01).astype(int)          # 1% positive
    data = {"y": y}
    for i in range(15):
        c = rng.normal(size=n)
        c[rng.choice(n, 45, replace=False)] = np.nan   # MCAR, unrelated to y
        data[f"sensor_{i}"] = c
    hits = [f for f in tl.analyse(pd.DataFrame(data), "y")
            if f.kind == "missingness-leak"]
    assert not hits, [h.column for h in hits]


def test_subset_purity_still_catches_a_real_one():
    """The same gate must not blunt the Titanic-shaped leak it was built for."""
    rng = np.random.default_rng(9)
    n = 1300
    y = (rng.random(n) > 0.38).astype(int)
    body = np.full(n, np.nan)
    body[np.flatnonzero(y == 0)[:121]] = rng.normal(size=121)
    hits = [f for f in tl.analyse(
        pd.DataFrame({"body": body, "f": rng.normal(size=n), "y": y}), "y")
        if f.kind == "missingness-leak"]
    assert len(hits) == 1 and hits[0].column == "body"


@pytest.mark.parametrize("n", [2000, 9000, 12000])
def test_integer_coded_categories_have_no_row_count_cliff(n):
    """Codes 3 and 7 always mean y=1 - a non-monotonic leak that rank AUC
    cannot see. Whether the column was target-encoded used to depend on
    `nunique <= max(10, len//1000)`, so the SAME leak was caught at 12,000
    rows and missed at 2,000, and caught as strings but missed as the int64
    that read_csv actually hands you."""
    rng = np.random.default_rng(0)
    code = rng.integers(0, 12, n)
    y = np.isin(code, [3, 7]).astype(int)
    df = pd.DataFrame({"reason_code": code, "noise": rng.normal(size=n), "y": y})
    assert "reason_code" in {f.column for f in tl.analyse(df, "y")
                             if f.severity == "critical"}


def test_integer_and_string_codings_agree():
    rng = np.random.default_rng(0)
    n = 2000
    code = rng.integers(0, 12, n)
    y = np.isin(code, [3, 7]).astype(int)
    as_int = tl.analyse(pd.DataFrame({"c": code, "y": y}), "y")
    as_str = tl.analyse(pd.DataFrame({"c": [f"R{v:02d}" for v in code],
                                      "y": y}), "y")
    sev = lambda out: {f.severity for f in out if f.column == "c"}  # noqa: E731
    assert sev(as_int) == sev(as_str)


def test_datetime_ordering_with_the_target_is_scored():
    """Timestamps are not is_numeric_dtype, so they were target-encoded - and
    every value is distinct, so the encoding returns the global mean and a
    date that ranks perfectly with the target scored 0.5."""
    n = 3000
    y = (np.arange(n) > n * 0.6).astype(int)
    df = pd.DataFrame({"resolved_at": pd.date_range("2024-01-01", periods=n,
                                                    freq="h"), "y": y})
    assert tl._score_column(df["resolved_at"], df["y"], "binary")["score"] > 0.9


def test_duplicate_column_names_raise_rather_than_report_clean():
    """df[c] returns a DataFrame, every check raises, and the per-column
    handler swallowed it as info - so a duplicated perfect leak came back
    'no leakage detected' with exit code 0."""
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 600)
    for cols in (["refund", "refund", "y"], ["a", "y", "y"]):
        df = pd.DataFrame(np.c_[y * 5.0, y * 5.0, y], columns=cols)
        with pytest.raises(ValueError, match="duplicate column name"):
            tl.analyse(df, "y")


def test_unhashable_cells_after_the_first_row():
    """The hashability probe looked only at row 0, so a frame whose first row
    is scalar and whose later rows hold lists passed it and then died."""
    rng = np.random.default_rng(2)
    n = 300
    df = pd.DataFrame({"emb": [[1.0, 2.0]] * n, "f": rng.normal(size=n),
                       "y": rng.integers(0, 2, n)})
    df.loc[0, "emb"] = "scalar"
    assert {f.kind for f in tl.analyse(df, "y")} >= {"unscoreable"}


def test_reported_auc_does_not_depend_on_row_order():
    """The positive class was `unique()[-1]` - order of appearance - while the
    evidence table used sorted order. Shuffling the rows flipped the reported
    AUC between 1.0000 and 0.0000, and the two halves of one report
    contradicted each other."""
    rng = np.random.default_rng(3)
    seen = set()
    for first in (0, 1):
        y = np.array([first] + [1 - first] * 299 + [first] * 300)
        df = pd.DataFrame({"x": y * 100.0 + rng.normal(0, .3, 600), "y": y})
        f = next(q for q in tl.analyse(df, "y") if q.kind == "target-proxy")
        seen.add(round(f.data["auc"], 6))
    assert len(seen) == 1, f"AUC changed with row order: {seen}"


@pytest.mark.parametrize("train,val", [
    (float("nan"), 0.5), (0.9, float("nan")),
    (float("inf"), 0.5), (1.4, 0.6), (0.9, -3.0),
])
def test_diagnose_rejects_impossible_scores(train, val):
    """min(1.0, nan) is 1.0 in Python, so a NaN score produced a confident
    diagnosis - and in opposite directions depending on which one was NaN."""
    with pytest.raises(ValueError):
        tl.diagnose(train, val)


# --- statistical power: the flaw a reviewer would attack first ---------------

@pytest.mark.parametrize("n", [20, 30, 50])
def test_small_samples_do_not_produce_leaks(n):
    """60 columns of pure noise on a handful of rows. A fixed 0.90 threshold
    ignores sample size, so a coin landing badly reads as a leak: at n=20 this
    reported a critical finding on random data. Nothing here may be critical
    or warning."""
    rng = np.random.default_rng(7)
    df = pd.DataFrame({f"f{i}": rng.normal(size=n) for i in range(60)})
    df["y"] = rng.integers(0, 2, n)
    loud = [f for f in tl.analyse(df, "y")
            if f.severity in ("critical", "warning")
            and f.kind in ("target-proxy", "suspiciously-predictive",
                           "pure-categories", "missingness-leak")]
    assert not loud, [f"{f.kind}:{f.column}" for f in loud]


def test_underpowered_findings_are_still_shown():
    """Rare-event data: 4 positives in 40 rows. A column that separates them
    perfectly still only stands 3.25 SE off the null, so it is reported and
    explained rather than called a leak. Silence would be its own failure -
    the user needs to see the column and know why it was not escalated."""
    rng = np.random.default_rng(0)
    y = np.array([1] * 4 + [0] * 36)
    df = pd.DataFrame({"perfect_on_four": y + rng.normal(0, .01, 40), "y": y})
    out = tl.analyse(df, "y")
    under = [f for f in out if f.kind == "underpowered"]
    assert under, [f"{f.kind}:{f.column}" for f in out]
    assert under[0].severity == "info"
    assert "SE" in under[0].detail
    assert not [f for f in out if f.severity == "critical"]


def test_the_same_column_is_critical_once_there_is_enough_data():
    """The identical pattern at 10x the rows must escalate - the gate is about
    power, not about being timid."""
    rng = np.random.default_rng(0)
    y = np.array([1] * 40 + [0] * 360)
    df = pd.DataFrame({"perfect": y + rng.normal(0, .01, 400), "y": y})
    assert "perfect" in {f.column for f in tl.analyse(df, "y")
                         if f.severity == "critical"}


def test_real_leak_still_caught_when_powered():
    """The power gate must not blunt the actual product."""
    rng = np.random.default_rng(0)
    n = 2000
    y = rng.integers(0, 2, n)
    df = pd.DataFrame({"refund": y * 100.0 + rng.normal(0, .3, n), "y": y})
    assert "refund" in {f.column for f in tl.analyse(df, "y")
                        if f.severity == "critical"}


def test_null_se_shrinks_with_sample_size():
    assert tl._null_se(10, 10) > tl._null_se(1000, 1000)
    assert tl._z_min(5) < tl._z_min(500)


def test_reported_auc_is_the_real_auc():
    """max(auc, 1-auc) was reported as "AUC", so a perfectly inverted feature
    claimed AUC 1.0000 when its true AUC was 0.0000 - a number the user could
    not reconcile with their own metrics."""
    y = np.array([0, 0, 0, 1, 1, 1])
    inverted = pd.Series([9.0, 8, 7, 3, 2, 1])
    assert tl._auc(inverted, y) == 0.0
    assert tl._separation(tl._auc(inverted, y)) == 1.0


def test_inverted_leak_is_reported_as_inverted():
    rng = np.random.default_rng(1)
    n = 1500
    y = rng.integers(0, 2, n)
    df = pd.DataFrame({"backwards": -y * 80.0 + rng.normal(0, .4, n), "y": y})
    f = next(x for x in tl.analyse(df, "y") if x.kind == "target-proxy")
    assert "inverted" in f.detail, f.detail


# --- multiclass targets: previously a silent wrong answer --------------------

def test_multiclass_target_is_handled_not_mislabelled():
    """A 4-class nominal target used to fall through to the continuous branch,
    where a Spearman correlation against class codes is meaningless."""
    rng = np.random.default_rng(3)
    n = 1200
    y = rng.integers(0, 4, n)
    df = pd.DataFrame({"leaky": y * 10.0 + rng.normal(0, .2, n),
                       "noise": rng.normal(size=n), "y": y})
    out = tl.analyse(df, "y")
    assert any("multiclass" in f.detail for f in out if f.kind == "target")
    crit = {f.column for f in out if f.severity == "critical"}
    assert "leaky" in crit and "noise" not in crit


def test_multiclass_one_vs_rest_names_the_class():
    rng = np.random.default_rng(4)
    n = 900
    y = rng.integers(0, 3, n)
    df = pd.DataFrame({"tells_class_2": (y == 2).astype(float)
                       + rng.normal(0, .01, n), "y": y})
    f = next(x for x in tl.analyse(df, "y") if x.kind == "target-proxy")
    assert "class" in f.data["metric"]
    # numpy scalars repr as "np.int64(2)"; that must not reach a user.
    assert "np." not in f.data["metric"], f.data["metric"]
    assert "np." not in f.detail, f.detail


def test_unsupported_target_raises_instead_of_guessing():
    df = pd.DataFrame({"x": range(100),
                       "t": [f"free text {i}" for i in range(100)]})
    with pytest.raises(ValueError, match="distinct values"):
        tl.analyse(df, "t")


def test_single_valued_target_raises():
    df = pd.DataFrame({"x": range(50), "t": [1] * 50})
    with pytest.raises(ValueError, match="distinct values"):
        tl.analyse(df, "t")


# --- ignore list: without it the check can never go green in CI -------------

def test_ignored_column_does_not_fail_the_run(demo):
    before = {f.column for f in tl.analyse(demo, "churned")
              if f.severity == "critical"}
    assert "refund_amount" in before
    after = tl.analyse(demo, "churned",
                       ignore=["refund_amount", "cancellation_reason"])
    assert not [f for f in after if f.severity == "critical"]


def test_ignored_column_is_still_listed(demo):
    """A silent suppression hides the next regression. It must stay visible."""
    out = tl.analyse(demo, "churned", ignore=["refund_amount"])
    noted = {f.column for f in out if f.kind == "ignored"}
    assert "refund_amount" in noted


def test_stale_ignore_entry_is_reported(demo):
    out = tl.analyse(demo, "churned", ignore=["no_such_column"])
    assert any(f.kind == "stale-ignore" for f in out)


def test_cli_ignore_flag_flips_the_exit_code():
    assert tl.main(["--demo"]) == 1
    assert tl.main(["--demo", "--ignore",
                    "refund_amount,cancellation_reason"]) == 0


# --- missingness -------------------------------------------------------------

def test_missingness_can_be_the_leak():
    """A field only populated for one class leaks through its NaN pattern even
    when the values themselves are innocuous. Imputation does not save you."""
    rng = np.random.default_rng(6)
    n = 800
    y = rng.integers(0, 2, n)
    refund_date = np.where(y == 1, 20240101.0, np.nan)  # only churners refund
    df = pd.DataFrame({"refund_date": refund_date,
                       "tenure": rng.normal(size=n), "y": y})
    out = tl.analyse(df, "y")
    hits = [f for f in out if f.kind == "missingness-leak"]
    assert hits and hits[0].column == "refund_date", [f.kind for f in out]
    assert hits[0].severity == "critical"


def test_leak_that_only_covers_a_subset_of_rows():
    """The Titanic `body` column: a body-recovery number exists for 121 of
    1,309 passengers, every one of whom died. Its missingness AUC is 0.575 -
    ranking metrics are blind to a column that is perfectly predictive on a
    small subset, and the benchmark caught this as a miss on a documented leak.
    """
    rng = np.random.default_rng(31)
    n = 1300
    y = (rng.random(n) > 0.38).astype(int)          # ~62% class 1
    body = np.full(n, np.nan)
    dead = np.flatnonzero(y == 0)[:121]
    body[dead] = rng.normal(size=len(dead))          # only ever for class 0
    df = pd.DataFrame({"body": body, "fare": rng.normal(size=n), "y": y})

    # The overall ranking signal really is weak - that is the whole point.
    weak = tl._score_column(pd.Series(body).isna().astype(int), df["y"], "binary")
    assert weak["score"] < tl.AUC_WARN, weak

    hits = [f for f in tl.analyse(df, "y") if f.kind == "missingness-leak"]
    assert hits and hits[0].column == "body"
    assert hits[0].severity == "critical"
    assert "always" in hits[0].detail


def test_subset_rule_needs_real_support():
    """Three rows that happen to share a target value are not a leak."""
    rng = np.random.default_rng(32)
    n = 500
    y = rng.integers(0, 2, n)
    col = np.full(n, np.nan)
    col[np.flatnonzero(y == 1)[:3]] = 1.0
    df = pd.DataFrame({"sparse": col, "ok": rng.normal(size=n), "y": y})
    assert not [f for f in tl.analyse(df, "y") if f.kind == "missingness-leak"]


def test_harmless_missingness_is_not_flagged():
    rng = np.random.default_rng(7)
    n = 800
    x = rng.normal(size=n)
    x[rng.choice(n, 200, replace=False)] = np.nan  # missing at random
    df = pd.DataFrame({"x": x, "y": rng.integers(0, 2, n)})
    assert not any(f.kind == "missingness-leak" for f in tl.analyse(df, "y"))


def test_demo_reads_identically_from_disk(tmp_path, demo):
    """pandas turns the literal string 'n/a' into NaN, so a category named
    that vanishes on a CSV round-trip and the demo reported 53% purity from
    disk against 100% in memory."""
    p = tmp_path / "demo.csv"
    demo.to_csv(p, index=False)
    mem = {(f.kind, f.column, f.detail) for f in tl.analyse(demo, "churned")}
    disk = {(f.kind, f.column, f.detail)
            for f in tl.analyse(tl.load(p), "churned")}
    purity = [d for k, c, d in mem if k == "pure-categories"]
    assert purity and purity[0].startswith("100%"), purity
    assert mem == disk, f"disk and memory disagree:\n{mem ^ disk}"


# --- dtype handling ----------------------------------------------------------

def test_dates_survive_a_csv_round_trip(tmp_path, demo):
    """pandas 3 reads dates back as dtype 'str', not 'object'. An earlier
    _is_timelike tested `dtype != object`, so every date in a real CSV was
    missed while the in-memory test passed on a datetime64 column."""
    p = tmp_path / "rt.csv"
    demo.to_csv(p, index=False)
    kinds = {(f.kind, f.column) for f in tl.analyse(tl.load(p), "churned")}
    assert ("temporal-column", "signup_date") in kinds
    assert ("identifier-like", "signup_date") not in kinds
    assert "refund_amount" in {f.column for f in tl.analyse(tl.load(p), "churned")
                               if f.severity == "critical"}


def test_continuous_target():
    df = pd.DataFrame({"x": np.arange(300.0), "t": np.arange(300.0) * 2 + 1})
    assert any(f.severity == "critical" for f in tl.analyse(df, "t"))


# --- split contamination -----------------------------------------------------

def test_detects_train_test_contamination(demo):
    dup = pd.concat([demo.head(60), demo], ignore_index=True)
    dup["split"] = ["test"] * 60 + ["train"] * len(demo)
    out = tl.analyse(dup, "churned", split="split")
    assert any(f.kind == "train-test-contamination" for f in out)


# --- reporting contract ------------------------------------------------------

def test_every_visible_kind_has_a_remedy(demo):
    """A finding without a fix is a scolding. Fails when a kind is added
    without remediation text."""
    kinds = {f.kind for f in tl.analyse(demo, "churned")}
    assert {k for k in kinds if k not in tl.FIXES} <= {"target"}


def test_report_wraps_and_deduplicates_fixes(demo):
    text = tl.report(tl.analyse(demo, "churned"))
    assert "FIX:" in text
    assert text.count("FIX: The categories partition") == 1, "fix repeated per column"
    assert all(len(ln) <= 90 for ln in text.splitlines()), "unwrapped text"
    assert "FIX:" not in tl.report(tl.analyse(demo, "churned"), show_fixes=False)


def test_findings_are_json_serialisable(demo):
    out = tl.analyse(demo, "churned")
    payload = json.loads(json.dumps([f.as_dict() for f in out]))
    assert set(payload[0]) == {"severity", "kind", "column", "detail",
                               "evidence", "data", "fix"}


def test_evidence_text_is_derived_from_the_data(demo):
    """The sentence and the chart must never disagree, so both come from one
    structured dict rather than being formatted independently."""
    f = next(x for x in tl.analyse(demo, "churned")
             if x.column == "refund_amount" and x.kind == "target-proxy")
    assert f.data["kind"] == "by_class"
    assert len(f.data["groups"]) == 2
    for g in f.data["groups"]:
        assert tl._fmt(g["mean"]) in f.evidence


# --- evidence: what users said they could not figure out ---------------------

def test_numeric_leak_shows_per_class_numbers(demo):
    """'AUC 1.0000' is an assertion. Per-class means are something the user can
    look at and judge, which matters because only they know the column."""
    f = next(x for x in tl.analyse(demo, "churned")
             if x.kind == "target-proxy" and x.column == "refund_amount")
    assert f.evidence and "target=0" in f.evidence and "target=1" in f.evidence
    assert "mean" in f.evidence and "n=" in f.evidence


def test_pure_categories_name_the_categories(demo):
    f = next(x for x in tl.analyse(demo, "churned") if x.kind == "pure-categories")
    assert f.evidence
    assert "not_given" in f.evidence and "always" in f.evidence


def test_missingness_evidence_is_a_crosstab():
    rng = np.random.default_rng(8)
    n = 800
    y = rng.integers(0, 2, n)
    df = pd.DataFrame({"refund_date": np.where(y == 1, 20240101.0, np.nan),
                       "tenure": rng.normal(size=n), "y": y})
    f = next(x for x in tl.analyse(df, "y") if x.kind == "missingness-leak")
    assert f.evidence and "missing:" in f.evidence and "present:" in f.evidence


def test_evidence_never_breaks_the_report():
    """Evidence is a nicety. A weird column must not take the whole run down."""
    df = pd.DataFrame({"weird": [{"a": 1}, {"b": 2}] * 50,
                       "ok": list(range(100)),
                       "y": [0, 1] * 50})
    tl.report(tl.analyse(df, "y"))  # must not raise


# --- generated fix code ------------------------------------------------------

def test_fix_code_names_the_real_columns(demo):
    out = tl.analyse(demo, "churned")
    code = tl.fix_code(out, target="churned")
    assert "'refund_amount'" in code and "'cancellation_reason'" in code
    assert "df.drop(columns=LEAKING)" in code
    assert "'signal'" not in code, "must not tell users to drop a good feature"


def test_fix_code_is_valid_python(demo):
    """Generated code that does not parse is worse than no generated code."""
    import ast
    code = tl.fix_code(tl.analyse(demo, "churned"), target="churned")
    body = "\n".join(ln for ln in code.splitlines()
                     if not ln.startswith("# ---"))
    ast.parse(body)


def test_fix_code_suggests_grouped_split_for_ids(demo):
    code = tl.fix_code(tl.analyse(demo, "churned"), target="churned")
    assert "StratifiedGroupKFold" in code
    assert "'customer_id'" in code


def test_fix_code_suggests_time_split_for_dates(demo):
    code = tl.fix_code(tl.analyse(demo, "churned"), target="churned")
    assert "quantile(0.8)" in code and "'signup_date'" in code


def test_fix_code_is_none_when_nothing_to_fix():
    rng = np.random.default_rng(9)
    y = rng.integers(0, 2, 500)
    df = pd.DataFrame({"a": y * 0.6 + rng.normal(0, 1, 500),
                       "b": rng.normal(0, 1, 500), "y": y})
    assert tl.fix_code(tl.analyse(df, "y"), target="y") is None


def test_report_can_omit_code(demo):
    out = tl.analyse(demo, "churned")
    assert "LEAKING" in tl.report(out, target="churned")
    assert "LEAKING" not in tl.report(out, target="churned", show_code=False)


def test_cli_json_includes_fix_code(capsys):
    tl.main(["--demo", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["fix_code"] and "LEAKING" in payload["fix_code"]


def test_unknown_target_raises(demo):
    with pytest.raises(ValueError, match="not in data"):
        tl.analyse(demo, "no_such_column")


# --- diagnose: over/underfitting, which need scores rather than data ---------

@pytest.mark.parametrize("train,val,expected", [
    (0.99, 0.98, "too-good-to-be-true"),  # both near perfect -> suspect a leak
    (0.98, 0.71, "overfitting"),          # big gap
    (0.58, 0.57, "underfitting"),         # both near chance
    (0.71, 0.78, "inverted-split"),       # val beats train
    (0.84, 0.79, "healthy"),              # ordinary, believable
])
def test_diagnose_names_the_failure_mode(train, val, expected):
    kinds = {f.kind for f in tl.diagnose(train, val)}
    assert expected in kinds, f"{train}/{val} -> {kinds}"


def test_diagnose_normalises_against_the_baseline():
    """90% accuracy is excellent at 50% chance and useless at 90% chance.
    Without a baseline the same numbers would get opposite diagnoses."""
    easy = {f.kind for f in tl.diagnose(0.90, 0.89, baseline=0.5)}
    imbalanced = {f.kind for f in tl.diagnose(0.90, 0.89, baseline=0.90)}
    assert "underfitting" not in easy
    assert "underfitting" in imbalanced


def test_diagnose_critical_modes_have_remedies():
    for train, val in [(0.99, 0.98), (0.98, 0.71), (0.55, 0.54)]:
        for f in tl.diagnose(train, val):
            if f.severity in ("critical", "warning"):
                assert f.fix, f"{f.kind} has no remediation"


def test_diagnose_rejects_impossible_baseline():
    with pytest.raises(ValueError, match="baseline"):
        tl.diagnose(0.9, 0.8, baseline=1.5)


def test_diagnose_report_renders(capsys):
    text = tl.report(tl.diagnose(0.98, 0.71))
    assert "overfitting" in text and "FIX:" in text
    assert all(len(ln) <= 90 for ln in text.splitlines())


def test_verdict_does_not_call_overfitting_a_leak():
    """Overfitting is not leakage. The verdict wording must follow what was
    actually found, not assume the data path."""
    assert "leak(s)" not in tl.report(tl.diagnose(0.98, 0.71))
    assert "issue(s)" in tl.report(tl.diagnose(0.98, 0.71))


def test_verdict_still_says_leak_for_real_leaks(demo):
    assert "leak(s)" in tl.report(tl.analyse(demo, "churned"))


def test_cli_diagnose_mode(capsys):
    assert tl.main(["--train", "0.98", "--val", "0.71"]) == 1
    assert "overfitting" in capsys.readouterr().out


def test_cli_diagnose_healthy_exits_zero(capsys):
    assert tl.main(["--train", "0.84", "--val", "0.79"]) == 0


def test_cli_train_and_val_must_pair():
    with pytest.raises(SystemExit):
        tl.main(["--train", "0.9"])


# --- HTML report -------------------------------------------------------------

def test_html_has_no_scripts_or_cdn_code(demo):
    """The report must not execute anything or depend on a code CDN. It gets
    emailed, opened from file://, and read on locked-down machines."""
    doc = tl.to_html(tl.analyse(demo, "churned"), target="churned")
    assert doc.startswith("<!doctype html>") and doc.rstrip().endswith("</html>")
    for forbidden in ("<script", "http://", "cdnjs", "jsdelivr", "unpkg",
                      "@import", "onclick", "onload"):
        assert forbidden not in doc.lower(), f"must not contain {forbidden}"


def test_html_fonts_degrade_when_offline(demo):
    """Typefaces load from Google Fonts, which is the one external request the
    report makes. That is only acceptable because every family declares a real
    local fallback - offline it renders in a local mono/serif, never in a
    default that wrecks the layout."""
    doc = tl.to_html(tl.analyse(demo, "churned"), target="churned")
    externals = {u for u in ("fonts.googleapis.com", "fonts.gstatic.com")
                 if u in doc}
    assert externals == {"fonts.googleapis.com", "fonts.gstatic.com"}
    assert "cdn" not in doc.lower().replace("fonts.googleapis.com", "") \
        .replace("fonts.gstatic.com", ""), "no code CDN"
    for stack in ("Consolas,monospace", "Georgia,serif"):
        assert stack in doc.replace(" ", ""), f"missing fallback: {stack}"


def test_html_contains_findings_and_charts(demo):
    doc = tl.to_html(tl.analyse(demo, "churned"), target="churned")
    assert "refund_amount" in doc and "cancellation_reason" in doc
    assert "<svg" in doc, "evidence should be charted"
    assert "LEAKING" in doc, "remediation code should be embedded"
    assert "critical" in doc


def test_html_escapes_hostile_column_names():
    """Column names come from user data and land in HTML. A frame with a
    script tag in a column name must not produce executable markup."""
    rng = np.random.default_rng(11)
    y = rng.integers(0, 2, 300)
    df = pd.DataFrame({"<script>alert(1)</script>": y * 9.0 + rng.normal(0, .1, 300),
                       "y": y})
    doc = tl.to_html(tl.analyse(df, "y"), target="y")
    assert "<script>alert" not in doc
    assert "&lt;script&gt;" in doc


def test_html_written_to_disk(tmp_path, demo):
    p = tmp_path / "r.html"
    tl.to_html(tl.analyse(demo, "churned"), target="churned", path=p)
    assert p.exists() and p.stat().st_size > 4000


def test_html_handles_clean_data_and_diagnose():
    rng = np.random.default_rng(12)
    y = rng.integers(0, 2, 400)
    clean = pd.DataFrame({"a": y * 0.5 + rng.normal(0, 1, 400), "y": y})
    assert "Nothing detected" in tl.to_html(tl.analyse(clean, "y"), target="y")
    assert "<svg" in tl.to_html(tl.diagnose(0.98, 0.71))


def test_html_shows_the_measurement_against_its_reference_range(demo):
    """The report's central device: a score is meaningless alone, so every
    scored finding is drawn against the band a healthy column falls in."""
    doc = tl.to_html(tl.analyse(demo, "churned"), target="churned")
    assert "EXPECTED" in doc and "PATHOLOGICAL" in doc
    assert "AUC 1.0000" in doc, "the measured value should be labelled"
    f = next(x for x in tl.analyse(demo, "churned") if x.kind == "target-proxy")
    assert f.data["band"] == [0.5, tl.AUC_WARN, tl.AUC_CRITICAL, 1.0]
    assert f.data["score"] >= tl.AUC_CRITICAL


def test_cli_html_flag(tmp_path, capsys):
    p = tmp_path / "out.html"
    tl.main(["--demo", "--html", str(p)])
    assert p.exists()
    assert "HTML report written" in capsys.readouterr().out


# --- CLI ---------------------------------------------------------------------

def test_cli_demo_exits_nonzero_on_leaks(capsys):
    assert tl.main(["--demo"]) == 1
    assert "VERDICT" in capsys.readouterr().out


def test_cli_json_output(capsys):
    assert tl.main(["--demo", "--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["critical"] >= 1
    assert payload["target"] == "churned"


def test_cli_clean_data_exits_zero(tmp_path, capsys):
    rng = np.random.default_rng(5)
    y = rng.integers(0, 2, 500)
    p = tmp_path / "clean.csv"
    pd.DataFrame({"a": y * 0.6 + rng.normal(0, 1, 500),
                  "b": rng.normal(0, 1, 500), "y": y}).to_csv(p, index=False)
    assert tl.main([str(p), "--target", "y"]) == 0
    assert "no leakage detected" in capsys.readouterr().out


def test_cli_requires_target_or_demo():
    with pytest.raises(SystemExit):
        tl.main(["somefile.csv"])


# --- the benchmark is the README's evidence, so it gets tests too ------------

def _load_benchmark():
    """The benchmark needs scikit-learn. It is in the dev extra so CI runs it,
    but skip rather than fail on a minimal install."""
    pytest.importorskip("sklearn", reason="benchmark needs scikit-learn")
    import importlib.util
    import pathlib
    spec = importlib.util.spec_from_file_location(
        "targetleak_benchmark",
        pathlib.Path(__file__).resolve().parents[1] / "benchmark"
        / "run_benchmark.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_benchmark_runs_offline_and_reports_the_clean_datasets(capsys):
    import importlib.util
    import pathlib
    bench = _load_benchmark()
    assert bench.main(["--offline"]) == 0
    out = capsys.readouterr().out
    assert "false positives on clean data" in out
    # The five sklearn datasets must actually have been measured.
    for name in ("iris", "wine", "breast_cancer", "digits", "diabetes"):
        assert name in out


def test_benchmark_leak_kinds_match_the_report():
    """The benchmark's recall metric and the report's verdict wording read
    from two separate copies of this set. Adding a leak kind to one and not
    the other silently degrades both."""
    bench = _load_benchmark()
    assert bench.LEAK_KINDS <= set(tl.FIXES), bench.LEAK_KINDS - set(tl.FIXES)
