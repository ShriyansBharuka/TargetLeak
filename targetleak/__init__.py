#!/usr/bin/env python3
"""Find data leakage in a training set before it fools you.

  targetleak --demo
  targetleak data.csv --target churned
  targetleak data.csv --target churned --split is_test --group user_id
  targetleak data.parquet --target y --json

Answers one question: "did my model learn, or did I leak?"

The core check is single-feature predictive power. A model needs many features
to reach AUC 0.95; a leak gets there with one. So any lone column that nearly
solves the target is the prime suspect -- it is usually a proxy for the answer
that will not exist at prediction time.

Categoricals are scored with OUT-OF-FOLD target encoding. This matters: encode
in-fold and every high-cardinality column scores ~1.0 (each category predicts
its own mean), so the tool would flag every legitimate feature and be useless.
"""
import argparse
import json
import sys

import numpy as np
import pandas as pd

__version__ = "0.1.0"
__all__ = ["analyse", "diagnose", "report", "to_html", "fix_code", "Finding",
           "FIXES", "demo_frame", "load", "main"]

# A real multi-feature model rarely clears 0.98 on one column. A leak does.
AUC_CRITICAL = 0.98
AUC_WARN = 0.90
# Above this share of unique values a column is an identifier, not a feature.
ID_UNIQUE_RATIO = 0.95
# Categories rarer than this are ignored when judging purity -- a category with
# one row is trivially "pure" and means nothing.
MIN_CATEGORY_SUPPORT = 20
FOLDS = 5


# Finding a leak is half the job. Naming the leak without saying what to do
# about it is a scolding, not a tool -- so every kind carries its remedy.
FIXES = {
    "target-proxy":
        "Establish when this column receives its value. If it is written at or "
        "after the moment the target becomes known, it cannot be an input. "
        "Drop it, or rebuild it from data available strictly before the "
        "prediction cutoff.",
    "pure-categories":
        "The categories partition the target, which usually means the column "
        "was derived from the outcome - a reason code only filled in for one "
        "class, for example. Drop it, or collapse it to categories that exist "
        "before the outcome does.",
    "suspicious-name":
        "If it is a label, drop it from the feature matrix. If it genuinely is "
        "a feature, rename it: the current name will mislead every future "
        "reader and every automated check, including this one.",
    "identifier-like":
        "Drop it from the feature matrix. If the entity matters, encode its "
        "properties (tenure, plan, region) rather than its identity. Keep the "
        "ID itself for grouping your splits.",
    "missingness-leak":
        "Find out what populates this column. If it is only written for one "
        "class - a refund date, a cancellation reason - then its absence is "
        "the answer, and imputing the values will not help. Drop it, or "
        "replace it with a flag that is genuinely known before the outcome.",
    "suspiciously-predictive":
        "Not proof of a leak. Confirm the value exists, with this same "
        "distribution, at the moment you predict. If it only arrives later, "
        "drop it or lag it.",
    "duplicate-rows":
        "Deduplicate before splitting, never after. If the repeats are "
        "legitimate, split by entity so all copies land on the same side.",
    "train-test-contamination":
        "Treat the current test score as unmeasured. Deduplicate across the "
        "whole set before splitting, then split by entity or group rather "
        "than by row.",
    "group-overlap":
        "Split on this column instead of at random - GroupKFold or "
        "StratifiedGroupKFold - so no entity appears on both sides.",
    "temporal-column":
        "Split by time, not at random: train on the past, test on the future. "
        "Then verify each feature is computed only from data available before "
        "its own row's timestamp.",
    "target-mostly-null":
        "Quote the labelled count as your training size. Then check whether "
        "labelled rows differ systematically from unlabelled ones - if they "
        "do, the model only works on that slice, whatever the test score says.",
    "constant":
        "Carries no information as-is. Worth investigating rather than just "
        "dropping: a feature that varies in the file but not on your labelled "
        "rows usually means an upstream job is not populating it for the rows "
        "you actually train on, which is a data bug, not a dead column.",
    "unscoreable":
        "Flatten or encode the column if it matters - one value per cell. "
        "Otherwise ignore this: the column was skipped, nothing else was.",
    "too-good-to-be-true":
        "Do not celebrate yet. Run the data checks on the training file "
        "before you trust this: a score this high on a real problem is more "
        "often a leak than a breakthrough. If the data checks come back clean, "
        "hold out a slice from a different time period and score that.",
    "overfitting":
        "The model memorised the training set. In rough order of what to try: "
        "early stopping on a validation set, then regularisation (weight decay, "
        "dropout, higher min_samples_leaf / lower max_depth), then fewer "
        "parameters or fewer features, then more data. Also confirm you are not "
        "tuning hyperparameters on the same split you report - that inflates "
        "the validation score itself and hides the gap.",
    "underfitting":
        "The model has not learned what is there - or there is nothing there. "
        "Check which before adding capacity: shuffle the target and retrain. "
        "If the shuffled score matches your real one, the features carry no "
        "signal and a bigger model will not help. If the real score is clearly "
        "better, then add capacity, engineer features, or train longer.",
    "inverted-split":
        "Validation beating training usually means a broken split, an easier "
        "validation slice, or something leaking into validation only. Check "
        "class balance and date ranges on both sides first. It is also normal "
        "with heavy dropout or augmentation, since those penalise the training "
        "score only - rule that out before hunting further.",
}

# Diagnosis thresholds, on skill normalised against the chance baseline, so one
# set of numbers covers AUC (baseline 0.5), accuracy on imbalanced data
# (baseline = majority rate) and R^2 (baseline 0).
SKILL_NEAR_PERFECT = 0.90
SKILL_LOW = 0.30
SKILL_GAP_LARGE = 0.20


class Finding:
    __slots__ = ("severity", "kind", "column", "detail", "data")

    def __init__(self, severity, kind, column, detail, data=None):
        self.severity = severity  # critical | warning | info
        self.kind = kind
        self.column = column
        self.detail = detail
        # Concrete numbers behind the claim. "AUC 1.0000" is an assertion;
        # "class 0 mean 0.02, class 1 mean 100.01" is something you can look
        # at and judge for yourself -- which is what users actually need,
        # because only they know whether the column is legitimate. Held as
        # structured data so the terminal line, the JSON and the HTML chart
        # are all rendered from the same numbers.
        self.data = data

    @property
    def evidence(self):
        return _ev_text(self.data)

    @property
    def fix(self):
        return FIXES.get(self.kind)

    def as_dict(self):
        return {"severity": self.severity, "kind": self.kind,
                "column": self.column, "detail": self.detail,
                "evidence": self.evidence, "data": self.data,
                "fix": self.fix}

    def __repr__(self):
        where = f" [{self.column}]" if self.column else ""
        return f"{self.severity.upper():8} {self.kind}{where}: {self.detail}"


def _auc(scores, y):
    """Rank AUC (Mann-Whitney U), tie-corrected, direction-agnostic.

    Returns max(auc, 1-auc): a feature that predicts the target perfectly
    backwards is exactly as leaky as one that predicts it forwards.
    """
    s = pd.Series(scores)
    ok = s.notna()
    s, yy = s[ok], np.asarray(y)[ok.to_numpy()]
    n1 = int((yy == 1).sum())
    n0 = int((yy == 0).sum())
    if n1 == 0 or n0 == 0 or len(s) == 0:
        return 0.5
    ranks = s.rank(method="average").to_numpy()
    auc = (ranks[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    return float(max(auc, 1.0 - auc))


def _oof_target_encode(col, y, folds=FOLDS, seed=0):
    """Target-encode a categorical using only out-of-fold means.

    In-fold encoding leaks the target into its own score, which would make
    every high-cardinality column look catastrophic. Out-of-fold is the whole
    reason this tool can tell a real leak from a legitimate feature.
    """
    y = pd.Series(np.asarray(y), index=col.index).astype(float)
    rng = np.random.default_rng(seed)
    fold = pd.Series(rng.permutation(len(col)) % folds, index=col.index)
    out = pd.Series(np.nan, index=col.index, dtype=float)
    for f in range(folds):
        tr, te = fold != f, fold == f
        means = y[tr].groupby(col[tr], observed=True).mean()
        out[te] = col[te].map(means).astype(float)
    return out.fillna(y.mean())


def _is_binary(y):
    return pd.Series(y).dropna().nunique() == 2


def _score_column(col, y, binary):
    """Single-column predictive power in [0.5, 1.0]. None if unscoreable."""
    if col.nunique(dropna=True) < 2:
        return None
    numeric = pd.api.types.is_numeric_dtype(col) and not _looks_categorical(col)
    if binary:
        yb = (pd.Series(y) == pd.Series(y).dropna().unique()[-1]).astype(int).to_numpy()
        scores = col if numeric else _oof_target_encode(col.astype("object"), yb)
        return _auc(scores, yb)
    # Continuous target: |Spearman| rescaled onto the same [0.5, 1] axis so one
    # threshold covers both problem types.
    s = col if numeric else _oof_target_encode(col.astype("object"), y)
    rho = pd.Series(s).corr(pd.Series(np.asarray(y), index=col.index),
                            method="spearman")
    return None if pd.isna(rho) else 0.5 + abs(float(rho)) / 2


def _looks_categorical(col):
    """Integer codes with few distinct values behave like categories, not scales."""
    return (pd.api.types.is_integer_dtype(col)
            and col.nunique(dropna=True) <= max(10, len(col) // 1000))


# Names that betray a column even when its correlation is mild. A 5-day forward
# label is future information whether or not it scores well against the target,
# so statistics alone cannot catch it -- but whoever named it knew what it was.
_LABEL_NAMES = ("label", "target", "outcome", "y_true", "ytrue", "ground_truth")
_FUTURE_NAMES = ("future", "fwd", "forward", "ahead", "t_plus", "tplus",
                 "lead_", "_lead", "nextday", "next_day")
_AFTER_NAMES = ("next_", "after_", "post_", "_post", "resolved", "final_")
_GENERIC_TARGETS = {"target", "label", "y", "outcome", "class", "result", "value"}


def _suspicious_name(name, target=None):
    """(severity, reason) for a column whose NAME implies it is not a feature."""
    n = str(name).lower()
    if any(k in n for k in _LABEL_NAMES):
        return ("critical", "the name says this is a label, not a feature. Another "
                            "label is still future information even when it "
                            "correlates only mildly with the one you are training on.")
    if any(k in n for k in _FUTURE_NAMES):
        return ("critical", "the name implies a forward-looking value. If it is "
                            "measured after the prediction moment it cannot be an input.")
    if any(k in n for k in _AFTER_NAMES):
        return ("warning", "the name implies something recorded after the event. "
                           "Confirm it is known at prediction time.")
    if target:
        t = str(target).lower()
        t_tokens = {p for p in t.replace("-", "_").split("_") if len(p) >= 4}
        n_tokens = {p for p in n.replace("-", "_").split("_") if len(p) >= 4}
        # Matched both ways on purpose: a target 'churned' and a column
        # 'churn_reason' share a stem, and neither name contains the other.
        for tok in t_tokens - _GENERIC_TARGETS:
            if tok in n:
                return ("warning", f"shares the name {tok!r} with the target column. "
                                   "Often a variant or descendant of the answer.")
        for tok in n_tokens - _GENERIC_TARGETS:
            if tok in t:
                return ("warning", f"its name {tok!r} appears inside the target "
                                   "column's name. Often a variant of the answer.")
    return None


def _looks_like_id(col):
    """Identifiers are strings or integer codes -- never continuous floats.

    A float measurement is ~100% unique by nature, so a bare uniqueness test
    labels every real numeric feature an ID and skips the columns that matter.
    """
    n = col.nunique(dropna=True)
    if n <= 50 or len(col) == 0 or n / len(col) <= ID_UNIQUE_RATIO:
        return False
    if pd.api.types.is_float_dtype(col) or _is_timelike(col):
        return False  # a timestamp is unique per row but it is not an ID
    return True


def _fmt(v):
    """Compact number formatting - long floats make evidence unreadable."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)[:24]
    if f == int(f) and abs(f) < 1e15:
        return f"{int(f):,}"
    return f"{f:,.4g}"


# Evidence is built as structured data first, and the sentence is derived from
# it. One source of truth serves the terminal, the JSON output and the HTML
# charts -- three renderings that must never disagree about the numbers.

def _ev_text(data):
    """Render an evidence dict as one readable line."""
    if not data:
        return None
    g = data.get("groups") or []
    k = data["kind"]
    if k == "by_class":
        return " | ".join(f"{x['label']}: mean {_fmt(x['mean'])}, "
                          f"median {_fmt(x['median'])} (n={x['n']:,})" for x in g)
    if k == "cat_rates":
        return " | ".join(f"{x['label']!r}: {x['rate']:.0%} positive "
                          f"(n={x['n']:,})" for x in g)
    if k == "missing":
        if data.get("binary"):
            return " | ".join(f"{x['label']}: {x['rate']:.0%} positive "
                              f"(n={x['n']:,})" for x in g)
        return " | ".join(f"{x['label']}: mean target {_fmt(x['rate'])} "
                          f"(n={x['n']:,})" for x in g)
    if k == "pure_cats":
        return " | ".join(f"{x['label']!r} -> always {_fmt(x['value'])} "
                          f"(n={x['n']:,})" for x in g)
    if k == "deciles":
        return " | ".join(f"{x['label']}: mean {_fmt(x['mean'])}" for x in g)
    if k == "scores":
        return (f"train {data['train']:.4f}, validation {data['val']:.4f}"
                + (f", chance {data['baseline']:.4f}" if "baseline" in data else ""))
    return None


def _evidence(col, y, binary, max_groups=4):
    """The pattern behind a finding, so the user can judge it themselves."""
    try:
        if binary:
            classes = sorted(pd.Series(y).dropna().unique())
            if len(classes) != 2:
                return None
            if pd.api.types.is_numeric_dtype(col) and not _looks_categorical(col):
                groups = []
                for cls in classes:
                    v = col[np.asarray(y) == cls].dropna()
                    if v.empty:
                        continue
                    groups.append({"label": f"target={_fmt(cls)}",
                                   "mean": float(v.mean()),
                                   "median": float(v.median()), "n": int(len(v))})
                return {"kind": "by_class", "groups": groups} if groups else None
            # Categorical: the categories with the most extreme target rates.
            rate = pd.Series(np.asarray(y), index=col.index).groupby(
                col.astype("object"), observed=True).agg(["mean", "count"])
            rate = rate.sort_values("mean")
            picked = pd.concat([rate.head(max_groups // 2),
                                rate.tail(max_groups // 2)])
            picked = picked[~picked.index.duplicated()]
            groups = [{"label": str(k)[:20], "rate": float(r["mean"]),
                       "n": int(r["count"])} for k, r in picked.iterrows()]
            return {"kind": "cat_rates", "groups": groups} if groups else None
        yy = pd.Series(np.asarray(y), index=col.index)
        if pd.api.types.is_numeric_dtype(col):
            top = col[yy >= yy.quantile(0.9)].dropna()
            bot = col[yy <= yy.quantile(0.1)].dropna()
            if len(top) and len(bot):
                return {"kind": "deciles", "groups": [
                    {"label": "bottom-decile target", "mean": float(bot.mean()),
                     "n": int(len(bot))},
                    {"label": "top-decile target", "mean": float(top.mean()),
                     "n": int(len(top))}]}
    except Exception:
        return None  # evidence is a nicety; never let it break the report
    return None


def _evidence_missing(col, y, binary):
    """Crosstab of missing-vs-present against the target."""
    try:
        yy = pd.Series(np.asarray(y), index=col.index)
        miss = col.isna()
        groups = [{"label": "missing" if m else "present",
                   "rate": float(yy[miss == m].mean()),
                   "n": int((miss == m).sum())} for m in (True, False)]
        return {"kind": "missing", "binary": bool(binary), "groups": groups}
    except Exception:
        return None


def _evidence_pure(col, y, limit=5):
    """Name the pure categories and their sizes."""
    try:
        yy = pd.Series(np.asarray(y), index=col.index)
        g = yy.groupby(col.astype("object"), observed=True).agg(
            ["count", "nunique", "first"])
        pure = g[(g["count"] >= MIN_CATEGORY_SUPPORT) & (g["nunique"] == 1)]
        pure = pure.sort_values("count", ascending=False).head(limit)
        groups = [{"label": str(k)[:20], "value": float(r["first"]),
                   "n": int(r["count"])} for k, r in pure.iterrows()]
        return {"kind": "pure_cats", "groups": groups} if groups else None
    except Exception:
        return None


def _category_purity(col, y):
    """Largest share of rows sitting in perfectly-pure, well-supported categories."""
    yb = pd.Series(np.asarray(y), index=col.index)
    g = yb.groupby(col.astype("object"), observed=True).agg(["count", "nunique"])
    pure = g[(g["count"] >= MIN_CATEGORY_SUPPORT) & (g["nunique"] == 1)]
    return float(pure["count"].sum() / len(col)) if len(col) else 0.0


def analyse(df, target, split=None, group=None):
    """Return a list of Findings, worst first."""
    if target not in df.columns:
        raise ValueError(f"target column {target!r} not in data: {list(df.columns)[:12]}")
    y = df[target]
    if y.isna().all():
        raise ValueError(f"target column {target!r} is entirely null")
    binary = _is_binary(y)
    findings = []
    skip = {target} | ({split} if split else set())
    features = [c for c in df.columns if c not in skip]
    if not features:
        raise ValueError("no feature columns left after excluding target/split")

    findings.append(Finding("info", "target", target,
                            f"{'binary' if binary else 'continuous'}, "
                            f"{len(df):,} rows, {len(features)} features"))

    null_share = float(y.isna().mean())
    if null_share > 0.05:
        findings.append(Finding(
            "warning", "target-mostly-null", target,
            f"{null_share:.0%} of rows have no target ({int(y.notna().sum()):,} "
            f"usable of {len(df):,}). Your real training set is that smaller "
            "number - quote it, and check the rows that survive are not a "
            "biased slice."))

    # Unlabelled rows must be dropped before ANY scoring. Comparing a NaN
    # target to a class value yields False, which quietly relabels every
    # unlabelled row as the negative class -- so a column that predicts
    # *which rows have labels* scores like a catastrophic leak. Seen on real
    # data: a ticker column hit AUC 0.93 purely because only 26 of 127
    # tickers were labelled at all.
    labelled = y.notna()
    if int(labelled.sum()) < 20:
        raise ValueError(
            f"only {int(labelled.sum())} rows have a target value - too few to "
            "assess. Check you named the right column.")
    df = df.loc[labelled]
    y = y.loc[labelled]

    # --- names that give a column away regardless of its score -----------
    for c in features:
        hit = _suspicious_name(c, target)
        if hit:
            findings.append(Finding(hit[0], "suspicious-name", c, hit[1]))

    # --- per-column predictive power -------------------------------------
    for c in features:
        # One hostile column must not take the whole run down. Real frames
        # carry list- and dict-valued columns (embeddings, JSON metadata)
        # that raise on nunique(); a tool that dies on them gets uninstalled.
        try:
            findings.extend(_column_findings(c, df[c], y, binary))
        except Exception as e:
            findings.append(Finding(
                "info", "unscoreable", c,
                f"could not analyse this column ({type(e).__name__}) - skipped. "
                "Values that are lists, dicts or other unhashable objects "
                "cannot be scored; flatten the column if it matters."))

    # --- dataset-level checks --------------------------------------------
    dup = int(df.duplicated().sum()) if _hashable(df) else 0
    if dup:
        findings.append(Finding(
            "warning", "duplicate-rows", None,
            f"{dup:,} fully duplicated rows ({dup / len(df):.1%}). Under a random "
            "split these land on both sides and inflate test scores."))

    if split is not None:
        findings.extend(_split_checks(df, target, split, features, group))

    for c in features:
        try:
            if _is_timelike(df[c]):
                findings.append(Finding(
                    "warning", "temporal-column", c,
                    "looks like a date/time. If you split randomly rather than by "
                    "time, the model trains on the future to predict the past."))
        except Exception:
            pass

    order = {"critical": 0, "warning": 1, "info": 2}
    findings.sort(key=lambda f: order[f.severity])
    return findings


def _hashable(df):
    """Cheap probe: duplicated() raises on unhashable cell values."""
    try:
        df.head(1).duplicated()
        return True
    except Exception:
        return False


def _column_findings(c, col, y, binary):
    """Every check for one column. Raises freely; the caller isolates it."""
    findings = []
    n_unique = col.nunique(dropna=True)
    if _looks_like_id(col):
        findings.append(Finding(
            "warning", "identifier-like", c,
            f"{n_unique:,} distinct values in {len(col):,} rows "
            f"({n_unique / len(col):.0%} unique) - looks like an ID. "
            "Drop it: models memorise IDs instead of learning."))
        return findings

    # Whether a value is missing can carry the answer even when the values
    # themselves are innocent -- a refund date or cancellation reason only
    # gets filled in for one class. Checked before the constant test on
    # purpose: a column that is constant where present and NaN elsewhere
    # leaks entirely through its NaN pattern.
    na_share = float(col.isna().mean())
    if 0.01 < na_share < 0.99:
        na_score = _score_column(col.isna().astype(int), y, binary)
        if na_score is not None and na_score >= AUC_WARN:
            sev = "critical" if na_score >= AUC_CRITICAL else "warning"
            findings.append(Finding(
                sev, "missingness-leak", c,
                f"whether this column is missing predicts the target at "
                f"{na_score:.4f} ({na_share:.0%} missing). The NaN pattern "
                "carries the answer even if the values do not.",
                _evidence_missing(col, y, binary)))

    score = _score_column(col, y, binary)
    if score is None:
        findings.append(Finding("info", "constant", c,
                                "single value or unscoreable - no information"))
        return findings
    metric = "AUC" if binary else "|Spearman|-scaled"

    def scored(ev):
        """Carry the measured score and its reference band alongside the
        evidence, so a report can present it the way a lab value is read:
        against the range a healthy column would fall in."""
        return {**(ev or {"kind": "score_only"}), "score": float(score),
                "metric": metric, "band": [0.5, AUC_WARN, AUC_CRITICAL, 1.0]}

    if score >= AUC_CRITICAL:
        findings.append(Finding(
            "critical", "target-proxy", c,
            f"alone reaches {metric} {score:.4f}. One column should not "
            "nearly solve the target - this is very likely computed from "
            "the answer, or recorded after it was known.",
            scored(_evidence(col, y, binary))))
    elif score >= AUC_WARN:
        findings.append(Finding(
            "warning", "suspiciously-predictive", c,
            f"alone reaches {metric} {score:.4f}. Plausible for a genuinely "
            "strong feature, but confirm it exists at prediction time.",
            scored(_evidence(col, y, binary))))

    if not pd.api.types.is_numeric_dtype(col) or _looks_categorical(col):
        purity = _category_purity(col.astype("object"), y)
        if purity >= 0.5 and score is not None and score >= AUC_WARN:
            findings.append(Finding(
                "critical", "pure-categories", c,
                f"{purity:.0%} of rows fall in categories with exactly one "
                "target value. The category encodes the answer.",
                _evidence_pure(col.astype("object"), y)))

    return findings


def _is_timelike(col):
    if pd.api.types.is_datetime64_any_dtype(col):
        return True
    # NB: pandas 3 gives string columns dtype 'str', not 'object'. Testing
    # `dtype != object` here silently skipped every date loaded from a CSV.
    if pd.api.types.is_numeric_dtype(col) or pd.api.types.is_bool_dtype(col):
        return False
    sample = col.dropna().astype(str).head(50)
    if sample.empty:
        return False
    try:  # pandas 3 warns rather than raises on unparseable input
        parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    except Exception:
        return False
    return bool(parsed.notna().mean() > 0.9)


def _split_checks(df, target, split, features, group):
    """Contamination between the two sides of a user-provided split."""
    out = []
    parts = df[split].dropna().unique()
    if len(parts) != 2:
        out.append(Finding("info", "split", split,
                           f"{len(parts)} distinct values - expected 2, skipping "
                           "cross-split checks"))
        return out
    a, b = df[df[split] == parts[0]], df[df[split] == parts[1]]
    feat = [c for c in features if c != group]

    if feat:
        ka = set(map(tuple, a[feat].astype("object").fillna("\0").to_numpy()))
        kb = list(map(tuple, b[feat].astype("object").fillna("\0").to_numpy()))
        shared = sum(1 for r in kb if r in ka)
        if shared:
            out.append(Finding(
                "critical", "train-test-contamination", None,
                f"{shared:,} rows in {parts[1]!r} have feature values identical to "
                f"rows in {parts[0]!r} ({shared / max(len(b), 1):.1%} of that side). "
                "The model has already seen its own test set."))

    for c in feat:
        col = df[c]
        n_unique = col.nunique(dropna=True)
        if not (3 <= n_unique <= len(df) * 0.5):
            continue
        if pd.api.types.is_numeric_dtype(col) and not _looks_categorical(col):
            continue
        sa, sb = set(a[c].dropna()), set(b[c].dropna())
        if not sb:
            continue
        overlap = len(sa & sb) / len(sb)
        if overlap > 0.9 and n_unique > 20:
            out.append(Finding(
                "warning", "group-overlap", c,
                f"{overlap:.0%} of this column's values appear on both sides of "
                "the split. If these are entities (user, patient, ticker), the "
                "model memorises them - split by this column instead."))
    return out


def _wrap(text, width=76, indent=" " * 9):
    out, line = [], indent
    for word in text.split():
        if len(line) + len(word) + 1 > width and line.strip():
            out.append(line)
            line = indent
        line += ("" if line == indent else " ") + word
    if line.strip():
        out.append(line)
    return "\n".join(out)


def diagnose(train, val, baseline=0.5, metric="AUC"):
    """Which failure mode are you in? Takes scores, not data.

    Leakage lives in the data and `analyse` finds it there. Over- and
    underfitting are properties of a *model* - no dataframe can reveal them,
    you need a training score and a validation score. This reads the two
    together and names the mode, because the same symptom ("my numbers look
    wrong") has three different causes and three opposite remedies.
    """
    if not 0 <= baseline < 1:
        raise ValueError(f"baseline must be in [0, 1), got {baseline}")

    def skill(s):
        """Fraction of the available headroom above chance that was captured."""
        return max(0.0, min(1.0, (float(s) - baseline) / (1.0 - baseline)))

    ts, vs = skill(train), skill(val)
    gap = ts - vs
    findings = [Finding(
        "info", "scores", None,
        f"train {metric} {train:.4f}, validation {metric} {val:.4f} "
        f"(chance {baseline:.4f}) - captured {ts:.0%} of the headroom on train, "
        f"{vs:.0%} on validation, a gap of {gap:+.0%}")]

    if vs > ts + 0.05:
        findings.append(Finding(
            "warning", "inverted-split", None,
            f"validation scores higher than training by {-gap:.0%}. Training is "
            "the set the model was fitted on, so it should not be the harder one.",
            {"kind": "scores", "train": train, "val": val,
             "baseline": baseline}))
    elif ts >= SKILL_NEAR_PERFECT and vs >= SKILL_NEAR_PERFECT:
        findings.append(Finding(
            "critical", "too-good-to-be-true", None,
            f"both scores capture over {SKILL_NEAR_PERFECT:.0%} of the available "
            "headroom. Genuine problems are rarely this easy, and a leak raises "
            "training and validation together - which is exactly why it survives "
            "cross-validation and then collapses in production.",
            {"kind": "scores", "train": train, "val": val,
             "baseline": baseline}))
    elif gap >= SKILL_GAP_LARGE:
        findings.append(Finding(
            "critical", "overfitting", None,
            f"training beats validation by {gap:.0%} of the headroom. The model "
            "fitted noise specific to the training rows.",
            {"kind": "scores", "train": train, "val": val,
             "baseline": baseline}))
    elif ts <= SKILL_LOW:
        findings.append(Finding(
            "critical", "underfitting", None,
            f"both scores sit within {ts:.0%} of chance. Either the model is too "
            "weak for the pattern, or there is no pattern to find.",
            {"kind": "scores", "train": train, "val": val,
             "baseline": baseline}))
    else:
        findings.append(Finding(
            "info", "healthy", None,
            f"a {gap:+.0%} headroom gap with {vs:.0%} captured on validation is "
            "an ordinary, believable result. Nothing here suggests a leak, "
            "overfitting, or underfitting."))

    order = {"critical": 0, "warning": 1, "info": 2}
    findings.sort(key=lambda f: order[f.severity])
    return findings


def fix_code(findings, target=None):
    """Generate remediation code naming the user's actual columns.

    A paragraph of advice is advice; a block with your own column names in it
    is a fix. This is the difference between knowing you have a leak and
    being able to do something about it in the next thirty seconds.
    """
    drop, regroup, retime, investigate = [], [], [], []
    for f in findings:
        if not f.column:
            continue
        if f.kind in ("target-proxy", "pure-categories", "missingness-leak"):
            drop.append(f.column)
        elif f.kind == "suspicious-name" and f.severity == "critical":
            drop.append(f.column)
        elif f.kind == "identifier-like":
            regroup.append(f.column)
        elif f.kind == "group-overlap":
            regroup.append(f.column)
        elif f.kind == "temporal-column":
            retime.append(f.column)
        elif f.kind in ("suspiciously-predictive", "constant"):
            investigate.append(f.column)

    def uniq(seq):
        return list(dict.fromkeys(seq))

    drop, regroup = uniq(drop), uniq(regroup)
    retime, investigate = uniq(retime), uniq(investigate)
    if not any((drop, regroup, retime, investigate)):
        return None

    out = ["# --- targetleak remediation " + "-" * 46]
    if drop:
        out += ["# Confirmed or near-certain leaks. Verify each one is not",
                "# legitimately available at prediction time before deleting.",
                "LEAKING = [",
                *[f"    {c!r}," for c in drop],
                "]",
                "df = df.drop(columns=LEAKING)",
                ""]
    if investigate:
        out += ["# Not proven. Check each is populated before the outcome, and",
                "# that constant columns are not an upstream pipeline failure.",
                f"SUSPECT = {investigate!r}",
                "print(df[SUSPECT].describe(include='all').T)",
                ""]
    if regroup:
        g = regroup[0]
        t = target or "<target>"
        out += ["# Split by entity so no individual appears on both sides.",
                "from sklearn.model_selection import StratifiedGroupKFold",
                "",
                "cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)",
                f"groups = df[{g!r}]",
                f"X = df.drop(columns=[{t!r}, {g!r}])",
                f"for tr, te in cv.split(X, df[{t!r}], groups=groups):",
                "    ...  # train on tr, evaluate on te",
                ""]
        if len(regroup) > 1:
            out += [f"# Other entity-like columns to consider grouping on: "
                    f"{regroup[1:]!r}", ""]
    if retime:
        t = retime[0]
        out += ["# Train on the past, test on the future. A random split here",
                "# lets the model see the answer before it is asked.",
                f"df = df.sort_values({t!r})",
                f"cutoff = df[{t!r}].quantile(0.8)",
                f"train = df[df[{t!r}] <  cutoff]",
                f"test  = df[df[{t!r}] >= cutoff]",
                ""]
    out.append("# Then re-run: targetleak <file> --target " + repr(target or "<target>"))
    return "\n".join(out)


def report(findings, show_fixes=True, target=None, show_code=True):
    crit = [f for f in findings if f.severity == "critical"]
    warn = [f for f in findings if f.severity == "warning"]
    lines = []
    seen_fix = set()
    for f in findings:
        where = f" [{f.column}]" if f.column else ""
        lines.append(f"{f.severity.upper():8} {f.kind}{where}")
        lines.append(_wrap(f.detail))
        if f.evidence:
            lines.append(_wrap("EVIDENCE: " + f.evidence))
        # One remedy per kind, not per column: seven leaky label columns share
        # one fix, and repeating it eleven times buries the findings.
        if show_fixes and f.fix and f.kind not in seen_fix:
            seen_fix.add(f.kind)
            lines.append(_wrap("FIX: " + f.fix))
        lines.append("")
    if lines and lines[-1] == "":
        lines.pop()
    lines.append("")
    # `diagnose` findings are not leaks, so the wording follows what was
    # actually found rather than assuming the data path.
    leak_kinds = {"target-proxy", "pure-categories", "missingness-leak",
                  "suspicious-name", "train-test-contamination"}
    noun = "leak" if any(f.kind in leak_kinds for f in crit) else "issue"
    if crit:
        verdict = (f"VERDICT: {len(crit)} critical {noun}(s). Do not trust this "
                   "model's test score until they are resolved.")
    elif warn:
        verdict = f"VERDICT: nothing certain, {len(warn)} thing(s) to confirm."
    elif any(f.kind == "healthy" for f in findings):
        verdict = ("VERDICT: scores look believable. This checks the numbers, "
                   "not the data - run the data checks too.")
    else:
        verdict = "VERDICT: no leakage detected by these checks."
    lines.append(_wrap(verdict, indent=""))
    if show_code:
        code = fix_code(findings, target)
        if code:
            lines += ["", code]
    return "\n".join(lines)


def demo_frame():
    """A frame with known leaks AND one honestly-strong feature.

    Public so `targetleak --demo` can show what a report looks like with no
    data of your own. The clean feature is the important half: a detector that
    flags everything is worthless, so the tests assert it is NOT flagged.
    """
    rng = np.random.default_rng(0)
    n = 1200
    y = rng.integers(0, 2, n)
    signal = y * 1.15 + rng.normal(0, 1, n)        # honest: AUC ~= 0.79
    return pd.DataFrame({
        "customer_id": [f"C{i:06d}" for i in range(n)],          # identifier
        "signal": signal,                                         # legitimate
        "noise": rng.normal(0, 1, n),                             # nothing
        # Not "n/a" for the negative class: pandas reads that literal string
        # as NaN, so the category would silently vanish from a CSV round-trip
        # and the demo would report different numbers from disk than in memory.
        "cancellation_reason": np.where(                          # target proxy
            y == 1, rng.choice(["price", "moved"], n), "not_given"),
        "refund_amount": y * 100.0 + rng.normal(0, 0.4, n),       # target proxy
        "signup_date": pd.date_range("2024-01-01", periods=n, freq="h"),
        "churned": y,
    })


def to_html(findings, target=None, source=None, path=None):
    """Render findings as a self-contained HTML report.

    Returns the HTML string, and writes it to `path` when given. The report
    embeds everything -- no CDN, no JS library -- so it opens offline, prints,
    and survives being attached to an email or dropped in a chat, which is how
    a finding actually reaches whoever owns the pipeline that caused it.
    """
    from . import _html
    doc = _html.render(findings, target=target, source=source,
                       fix_code_text=fix_code(findings, target),
                       version=__version__)
    if path:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(doc)
    return doc


def load(path):
    """Read a table by extension. Parquet keeps dtypes; CSV loses them."""
    low = str(path).lower()
    if low.endswith((".parquet", ".pq")):
        return pd.read_parquet(path)
    if low.endswith((".tsv", ".tab")):
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="targetleak",
        description="Find data leakage in a training set before it fools you.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="exit code 1 means at least one critical finding, so "
               "`targetleak data.csv --target y && python train.py` will not "
               "train on a leaking dataset.")
    ap.add_argument("data", nargs="?", help="CSV, TSV or Parquet file")
    ap.add_argument("--target", help="name of the target column")
    ap.add_argument("--split", help="column marking train/test rows")
    ap.add_argument("--group", help="entity column to exclude from row matching")
    ap.add_argument("--json", action="store_true",
                    help="machine-readable output, for CI")
    ap.add_argument("--no-fixes", action="store_true",
                    help="findings only, without remediation guidance")
    ap.add_argument("--no-code", action="store_true",
                    help="omit the generated remediation code block")
    ap.add_argument("--html", metavar="FILE",
                    help="also write a self-contained HTML report with charts")
    ap.add_argument("--demo", action="store_true",
                    help="run on a built-in frame with known leaks")
    diag = ap.add_argument_group(
        "diagnosis from scores",
        "Over- and underfitting are properties of a model, not a dataset, so "
        "these take scores rather than a file.")
    diag.add_argument("--train", type=float, metavar="SCORE",
                      help="training score")
    diag.add_argument("--val", type=float, metavar="SCORE",
                      help="validation score")
    diag.add_argument("--baseline", type=float, default=0.5, metavar="SCORE",
                      help="chance level for the metric (default 0.5 for AUC; "
                           "use the majority-class rate for accuracy, 0 for R2)")
    diag.add_argument("--metric", default="AUC", help="metric name, for wording")
    ap.add_argument("--version", action="version",
                    version=f"targetleak {__version__}")
    a = ap.parse_args(argv)

    if (a.train is None) != (a.val is None):
        ap.error("--train and --val must be given together")

    if a.train is not None:
        result = diagnose(a.train, a.val, a.baseline, a.metric)
        target = None
    elif a.demo:
        result = analyse(demo_frame(), "churned", a.split, a.group)
        target = "churned"
    elif a.data and a.target:
        result = analyse(load(a.data), a.target, a.split, a.group)
        target = a.target
    else:
        ap.error("give DATA and --target, or --demo, or --train and --val")
    if a.json:
        from . import _html
        print(_html.render_json(result, target, fix_code(result, target)))
    else:
        print(report(result, show_fixes=not a.no_fixes, target=target,
                     show_code=not a.no_code))

    if a.html:
        to_html(result, target=target, source=a.data, path=a.html)
        print(f"\nHTML report written to {a.html}")
    return 1 if any(x.severity == "critical" for x in result) else 0


if __name__ == "__main__":
    sys.exit(main())
