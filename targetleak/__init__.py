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

__version__ = "0.1.1"
__all__ = ["analyse", "diagnose", "report", "to_html", "fix_code", "Finding",
           "FIXES", "demo_frame", "load", "main"]

# A real multi-feature model rarely clears 0.98 on one column. A leak does.
AUC_CRITICAL = 0.98
AUC_WARN = 0.90
# An absolute threshold alone is not enough: on 20 rows a pure-noise column
# clears 0.90 by luck, and with 60 columns tested that happens routinely. Every
# score must therefore also stand off the null by Z_MIN standard errors, where
# the null SE comes from the Hanley-McNeil variance of AUC under no effect.
# Z_MIN grows with the number of columns as a Bonferroni approximation - crude,
# but it is the difference between reporting noise and not.
Z_TABLE = ((10, 3.5), (100, 4.0), (1000, 4.5))
Z_MAX = 5.0
# Ceiling on how many LABELS a non-numeric target may have. 26-class letter
# recognition is a mainstream problem and was once refused outright.
MAX_CLASSES = 100
# A NUMERIC column with only a handful of distinct values is plausibly coded
# classes; beyond that its ORDER is information, and shredding it into
# one-vs-rest groups throws away the thing that makes it a number. This is the
# distinction MAX_CLASSES was being asked to make and could not - it applies
# to targets (a 56-value CPU percentage was read as 56 unordered classes) and
# to predictors alike (120 of us_crime's 126 normalised floats were read as
# discrete flags, producing 111 findings and no usable verdict).
NUMERIC_CLASS_MAX = 15
# Above this share of distinct values a target is not a label at all - it is
# free text or an identifier pointed at the wrong column. Replaces a
# rows-per-class floor that refused 24-class audiology by 0.6 of a row,
# because a mean is the wrong statistic on a skewed class distribution.
TARGET_CARDINALITY_MAX = 0.5
# Above this share of unique values a column is an identifier, not a feature.
ID_UNIQUE_RATIO = 0.95
# Categories rarer than this are ignored when judging purity -- a category with
# one row is trivially "pure" and means nothing.
MIN_CATEGORY_SUPPORT = 20
FOLDS = 5
# Permutations used to measure the null for target-encoded columns, where the
# analytic SE does not apply. Only spent on columns that already cleared the
# score threshold, so this is per-candidate, not per-column.
PERMUTATIONS = 60
# Permutation calibration lowers the encoded z by roughly a fifth, so when the
# analytic z is already several times the bar the measured one cannot change
# the verdict. Skipping it there took a 452-row x 280-column frame from 32s to
# a couple of seconds - the cost was per candidate, and wide frames have many.
CALIBRATE_BELOW = 3.0
# The dataset is easy rather than leaky above this share of columns being
# individually predictive - AND this many of them in absolute terms. The share
# alone misfires on narrow frames: two leaks among six columns is 33% and is
# the perfectly ordinary case this note must not reframe. The note exists to
# defuse a wall of a hundred findings, and two is not a wall.
WIDESPREAD_SHARE = 0.25
WIDESPREAD_MIN = 8


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
        "Carries no information. Safe to drop, but check first that it is "
        "meant to be populated at all - a feature that is empty everywhere is "
        "often a pipeline that was never wired up.",
    "dead-on-labelled-rows":
        "Find out why the two do not overlap. Usually the labels cover one "
        "period and the feature job backfilled another, so joining them leaves "
        "a constant. Until that is fixed the feature contributes nothing, and "
        "any conclusion that 'the signal is not there' was measured without "
        "it.",
    "widespread-separability":
        "Do not work through these one by one. When this many columns are each "
        "predictive on their own, the usual explanation is that the problem is "
        "genuinely easy - engineered features for image or signal "
        "classification look exactly like this - not that the data is riddled "
        "with leaks. Leakage concentrates: one or two columns carrying the "
        "answer, not a hundred. Check the two or three strongest by hand, "
        "confirm they exist at prediction time, and ignore the tail.",
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
    """True rank AUC (Mann-Whitney U), tie-corrected, in [0, 1].

    Returns the AUC as measured, NOT max(auc, 1-auc). An earlier version
    folded the two together, so a perfectly inverted feature was reported as
    "AUC 1.0000" when its actual AUC was 0.0000 - a number the user could not
    reconcile with their own metrics. Direction is handled by the caller via
    `_separation`, which is what thresholds are applied to.
    """
    s = pd.Series(scores)
    ok = s.notna()
    s, yy = s[ok], np.asarray(y)[ok.to_numpy()]
    n1 = int((yy == 1).sum())
    n0 = int((yy == 0).sum())
    if n1 == 0 or n0 == 0 or len(s) == 0:
        return 0.5
    ranks = s.rank(method="average").to_numpy()
    return float((ranks[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _separation(auc):
    """Distance from chance, folded onto [0.5, 1]. A feature that predicts the
    target perfectly backwards is exactly as leaky as one that predicts it
    forwards, so thresholds run on this rather than on the raw AUC."""
    return float(max(auc, 1.0 - auc))


def _null_se(n1, n0):
    """SE of AUC under the null hypothesis of no association.

    Hanley-McNeil with AUC=0.5: sqrt((n1 + n0 + 1) / (12 * n1 * n0)). This is
    what makes the difference between "0.93 on 5,000 rows" (a finding) and
    "0.93 on 18 rows" (a coin landing badly).
    """
    if n1 < 1 or n0 < 1:
        return float("inf")
    return float(np.sqrt((n1 + n0 + 1.0) / (12.0 * n1 * n0)))


def _z_min(n_features):
    """Bonferroni-flavoured z floor: more columns tested, higher the bar."""
    for limit, z in Z_TABLE:
        if n_features <= limit:
            return z
    return Z_MAX


def _empirical_z(observed, resample, k=PERMUTATIONS, floor=1e-9):
    """Standard errors above a null estimated by permutation.

    The Hanley-McNeil SE assumes the score vector was fixed before the target
    was seen. That is true of a raw column and false of a target-encoded one:
    out-of-fold encoding removes the within-fold leak but not the between-fold
    one, so the encoded column stays correlated with the target even under the
    null. Measured on 600 null draws at n=1000, the analytic z was badly
    over-dispersed - a four-category column of pure noise cleared the 3.5 bar
    3.7% of the time against a nominal 0.05%, roughly 70x.

    So for encoded columns the null is measured instead of assumed: shuffle
    the target, re-encode against the shuffled copy, re-score, and read the
    observed value against that distribution. Only run for columns that have
    already cleared the score threshold, which is a handful per dataset, so
    the cost is bounded by the number of real candidates rather than by the
    width of the frame.
    """
    sims = np.array([resample(i) for i in range(k)], dtype=float)
    sims = sims[np.isfinite(sims)]
    if len(sims) < 5:
        return 0.0
    sd = float(sims.std(ddof=1))
    return float((observed - sims.mean()) / max(sd, floor))


def _as_content(col):
    """Represent a column by what it CONTAINS, not by how it is stored.

    This is one fix for five findings, all of which were the same mistake:
    a decision about a column being made from its dtype.

    OpenML ships `anneal.formability` as a `category` holding the strings
    '1'..'4'; `read_csv` gives the identical data as float64. Every decision
    downstream branched on that difference - which candidates to score,
    whether the column counted as discrete, whether it looked like a date - so
    the same file produced different findings depending on how it was loaded.
    Measured across 36 real datasets that changed the findings on 11% of them,
    and on `dermatology` it moved the verdict from 7 critical leaks to 11.

    It also fixes a false positive: `cylinder-bands.plating_tank` holds tank
    numbers '1910', '1911'. As strings those parse as years, so the column
    earned a `temporal-column` warning telling the user to sort and split by
    it. As numbers they are numbers.

    Only converted when EVERY present value is numeric. A mixed column keeps
    its own representation, because then the strings mean something.
    """
    if (pd.api.types.is_numeric_dtype(col)
            or pd.api.types.is_datetime64_any_dtype(col)
            or pd.api.types.is_bool_dtype(col)):
        return col
    try:
        as_num = pd.to_numeric(col, errors="coerce")
    except (TypeError, ValueError):
        return col
    present = int(col.notna().sum())
    if present and int(as_num.notna().sum()) == present:
        return as_num
    return col


def _normalise(df):
    """Apply `_as_content` across a frame without mutating the caller's."""
    changed = {}
    for c in df.columns:
        try:
            fixed = _as_content(df[c])
        except Exception:
            continue
        if fixed is not df[c]:
            changed[c] = fixed
    if not changed:
        return df
    out = df.copy(deep=False)      # shares data; copy-on-write per column
    for c, v in changed.items():
        out[c] = v
    return out


def _target_kind(y):
    """binary | multiclass | continuous | degenerate | unsupported.

    Multiclass used to fall through to the continuous branch, where a Spearman
    correlation was computed against nominal class codes - a meaningless
    number reported with full confidence. Silent wrong answers are worse than
    refusing, so each kind is now named and handled explicitly.
    """
    s = pd.Series(y).dropna()
    n = int(s.nunique())
    if n < 2:
        return "degenerate"
    if n == 2:
        return "binary"
    if pd.api.types.is_numeric_dtype(s):
        # Order is information. Only a handful of distinct values reads as codes.
        return "multiclass" if n <= NUMERIC_CLASS_MAX else "continuous"
    if n <= MAX_CLASSES and n / len(s) < TARGET_CARDINALITY_MAX:
        return "multiclass"
    return "unsupported"


def _oof_target_encode(col, y, folds=FOLDS, seed=0):
    """Target-encode a categorical using only out-of-fold means.

    In-fold encoding leaks the target into its own score, which would make
    every high-cardinality column look catastrophic. Out-of-fold is the whole
    reason this tool can tell a real leak from a legitimate feature.

    Factorised once up front, so the fold loop groups integer codes instead of
    Python objects. pandas' own group_mean still does the summing, so the
    floats are bit-identical to grouping the values themselves.
    """
    yv = np.asarray(y, dtype=float)
    rng = np.random.default_rng(seed)
    fold = rng.permutation(len(col)) % folds
    codes, _ = pd.factorize(col, use_na_sentinel=True)
    ncat = int(codes.max()) + 1 if len(codes) else 0
    out = np.full(len(col), np.nan)
    ys = pd.Series(yv)
    known = codes >= 0

    # A one-vs-rest target is a 0/1 indicator, and a multiclass frame asks for
    # one encoding per class per column - 856 columns x 9 classes x 5 folds was
    # 38,520 groupby calls and 58% of the runtime on a real dataset.
    #
    # For an indicator the fold loop collapses to arithmetic: subtract each
    # fold's sums from the totals. That is normally unsafe here, because
    # pandas' group_mean uses Kahan compensated summation and np.bincount does
    # not, so a continuous target drifts by ~1e-16. An indicator cannot drift -
    # its group sums are whole numbers held exactly in float64 - so the
    # shortcut is bit-exact for exactly this case, and the continuous case
    # keeps the groupby path below.
    finite = yv[np.isfinite(yv)]
    if ncat and len(finite) and np.array_equal(finite, finite.astype(bool)):
        kc, ky = codes[known], yv[known]
        kf = fold[known]
        tot_sum = np.bincount(kc, weights=ky, minlength=ncat)
        tot_cnt = np.bincount(kc, minlength=ncat).astype(float)
        for f in range(folds):
            sel = kf == f
            f_sum = np.bincount(kc[sel], weights=ky[sel], minlength=ncat)
            f_cnt = np.bincount(kc[sel], minlength=ncat).astype(float)
            n_tr = tot_cnt - f_cnt
            with np.errstate(invalid="ignore", divide="ignore"):
                lut = np.where(n_tr > 0, (tot_sum - f_sum) / n_tr, np.nan)
            te = known & (fold == f)
            out[te] = lut[codes[te]]
        return pd.Series(out, index=col.index).fillna(ys.mean())

    for f in range(folds if ncat else 0):
        tr = known & (fold != f)
        means = ys[tr].groupby(codes[tr], sort=False).mean()
        lut = np.full(ncat, np.nan)
        lut[means.index.to_numpy()] = means.to_numpy()
        te = known & (fold == f)
        out[te] = lut[codes[te]]
    return pd.Series(out, index=col.index).fillna(ys.mean())


def _score_column(col, y, kind, n_features=1, n_unique=None):
    """Measure one column against the target.

    Returns None when unscoreable, else a dict:
      score    separation in [0.5, 1], what thresholds are applied to
      auc      the true AUC where one exists, so the number is reconcilable
      metric   what to call it in the report
      z        standard errors above the null - guards against small samples
      z_min    the bar `z` had to clear
    """
    if (col.nunique(dropna=True) if n_unique is None else n_unique) < 2:
        return None
    zmin = _z_min(n_features)

    def candidates(target_vec):
        """Score vectors worth trying for this column.

        An integer column is ambiguous - a measurement, where order carries
        the signal, or a code, where it does not. Rank AUC sees only the
        first and target encoding only the second, and choosing between them
        by row count meant the SAME leak was caught at 12,000 rows and missed
        at 2,000, and caught as strings but missed as the integers read_csv
        actually gives you. Try both and keep the stronger: a tree model
        would happily use either.

        Each entry is (vector, built_from_the_target). The flag matters
        because an encoding built from the target invalidates the analytic
        null SE, so the winner has to say how it was made.
        """
        if pd.api.types.is_float_dtype(col):
            return [(col, False)]             # encoding a float is meaningless
        if pd.api.types.is_datetime64_any_dtype(col):
            # Order is the whole signal in a timestamp, and every value is
            # distinct, so target encoding returns the global mean and scores
            # 0.5 - a date that ranks perfectly with the target read as noise.
            return [(col.astype("int64"), False)]
        enc = (_oof_target_encode(col, target_vec), True)
        if pd.api.types.is_numeric_dtype(col):
            return [(col, False), enc]
        return [enc]

    def measure(target_ind):
        """Best AUC over the candidate encodings, with the null SE computed
        on the rows that were actually scored.

        The SE must come from the scored subset. Taking it from the full
        target inflated z by the column's missingness - a column with six
        observed values among a thousand rows was reported as "27 SE above
        chance", which is the opposite of what the power gate exists for.
        """
        best_auc, best_sep, best_se = 0.5, 0.5, float("inf")
        best_encoded = False
        for scores, encoded in candidates(target_ind):
            ok = pd.Series(scores).notna().to_numpy()
            used = np.asarray(target_ind)[ok]
            auc = _auc(scores, target_ind)
            sep = _separation(auc)
            if sep >= best_sep:
                best_auc, best_sep, best_encoded = auc, sep, encoded
                best_se = _null_se(int((used == 1).sum()),
                                   int((used == 0).sum()))
        z = (best_sep - 0.5) / best_se if best_se not in (0, float("inf")) else 0.0

        if best_encoded and best_sep >= AUC_WARN and z < CALIBRATE_BELOW * zmin:
            # The analytic SE assumes the scores were fixed before the target
            # was seen, which is false for an encoding built from it. Measure
            # the null instead - only for a column that already cleared the
            # score bar, so this is spent per candidate, not per column.
            ind = np.asarray(target_ind)

            def null_score(i):
                perm = np.random.default_rng(i).permutation(ind)
                return _separation(_auc(_oof_target_encode(col, perm), perm))

            z = _empirical_z(best_sep, null_score)
        return best_auc, best_sep, z

    if kind == "binary":
        # sorted(), not unique(): unique() is order-of-appearance, so the
        # positive class - and therefore the reported AUC and its "inverted"
        # flag - changed when the rows were shuffled, while the evidence table
        # below already used sorted order. One report contradicted itself.
        classes = sorted(pd.Series(y).dropna().unique())
        yb = (pd.Series(y) == classes[-1]).astype(int).to_numpy()
        auc, sep, z = measure(yb)
        return {"score": sep, "auc": auc, "metric": "AUC",
                "z": z, "z_min": zmin}

    if kind == "multiclass":
        # One-vs-rest keeps the best of K classes, so this column consumes K
        # tests rather than one and the Bonferroni count has to say so. Under
        # the null the max score climbs with K - 0.517 at 2 classes, 0.589 at
        # 16 - which is real inflation even though it did not reach the score
        # gate at the sizes measured.
        n_cls = int(pd.Series(y).dropna().nunique())
        zmin = _z_min(max(n_features, 1) * max(n_cls, 1))
        best, best_auc, best_z, best_cls = 0.5, None, 0.0, None
        for cls in sorted(pd.Series(y).dropna().unique(), key=repr):
            ind = (pd.Series(y) == cls).astype(int).to_numpy()
            auc, sep, z = measure(ind)
            if sep > best:
                best, best_auc, best_z, best_cls = sep, auc, z, cls
        label = _plain(best_cls)
        return {"score": best, "auc": best_auc,
                "metric": f"AUC vs class {label!r}",
                "z": best_z, "z_min": zmin}

    if kind == "continuous":
        yy = pd.Series(np.asarray(y), index=col.index)

        # A DISCRETE predictor against a continuous target must not be scored
        # with a correlation. |Spearman| for a two-group predictor is capped at
        # sqrt(3)/2 = 0.866 at an even split and collapses as the split skews,
        # so a flag that perfectly identifies the top 2% of a revenue target
        # scored 0.62 - indistinguishable from noise, while the identical data
        # against a binary target scored AUC 1.0000. Whole classes of leak
        # (binary flags, low-cardinality codes, missingness indicators) were
        # getting a free pass purely because the target was a number.
        #
        # So swap the roles: ask how well the target separates each of the
        # predictor's groups from the rest. That is an ordinary AUC, it reaches
        # 1.0 exactly when a group occupies one end of the target's range, and
        # it lands on the same [0.5, 1] axis as every other check. The groups
        # are not derived from y, so the Hanley-McNeil null SE is valid here.
        n_vals = int(col.nunique(dropna=True)) if n_unique is None else n_unique
        if n_vals <= NUMERIC_CLASS_MAX:
            best, best_z, best_lab = 0.5, 0.0, None
            for val in sorted(col.dropna().unique(), key=repr):
                ind = (col == val).to_numpy().astype(int)
                sep = _separation(_auc(yy, ind))
                if sep > best:
                    se = _null_se(int(ind.sum()), int((ind == 0).sum()))
                    best, best_lab = sep, val
                    best_z = (sep - 0.5) / se if se else 0.0
            if best_lab is None:
                return None
            return {"score": best, "auc": None,
                    "metric": f"target separation for {_plain(best_lab)!r}",
                    "z": best_z, "z_min": zmin}

        best_r, n, best_encoded = None, 0, False
        for s, encoded in candidates(np.asarray(y)):
            rho = pd.Series(s).corr(yy, method="spearman")
            if pd.isna(rho):
                continue
            if best_r is None or abs(float(rho)) > abs(best_r):
                best_r, best_encoded = float(rho), encoded
                n = int(min(pd.Series(s).notna().sum(), yy.notna().sum()))
        if best_r is None:
            return None
        # Fisher z for Spearman, mapped onto the same [0.5, 1] axis as AUC so
        # one set of thresholds covers both problem types. The 1.06 is
        # Spearman's variance inflation over Pearson's 1/(n-3).
        r = min(abs(best_r), 0.999999)
        score = 0.5 + r / 2
        z = (0.5 * np.log((1 + r) / (1 - r)) * np.sqrt(max(n - 3, 1) / 1.06)
             if n > 3 else 0.0)
        if best_encoded and score >= AUC_WARN and z < CALIBRATE_BELOW * zmin:
            # Same reason as the binary path: Fisher-z assumes the predictor
            # was not built from the target.
            yv = np.asarray(y)

            def null_rho(i):
                perm = np.random.default_rng(i).permutation(yv)
                enc = _oof_target_encode(col, perm)
                rr = pd.Series(enc).corr(pd.Series(perm, index=col.index),
                                         method="spearman")
                return 0.5 + min(abs(float(rr)), 0.999999) / 2 \
                    if pd.notna(rr) else np.nan

            z = _empirical_z(score, null_rho)
        return {"score": score, "auc": None,
                "metric": "|Spearman| (scaled)", "z": float(z), "z_min": zmin}

    return None


def _looks_categorical(col):
    """Few distinct values behaves like a category, whatever the dtype.

    Both halves of the previous rule - `is_integer_dtype(col) and nunique <=
    max(10, len(col) // 1000)` - were storage or size standing in for content:

    - The integer requirement made 1.0/2.0/3.0 behave differently from
      1/2/3, which is the residue of the dtype bug this commit is about.
    - The row-count term made the answer depend on how many rows sat
      underneath. A 46-value state code counted as a category only above
      ~46,000 rows; below that it was a "measurement" and the group-overlap
      scan skipped it, which is why grouping us_crime by state cost 0.0385 of
      real score with nothing reported (F7, still open - that check needs its
      own fix, and this only removes one of its two causes).
    """
    if not pd.api.types.is_numeric_dtype(col):
        return False
    return col.nunique(dropna=True) <= NUMERIC_CLASS_MAX


# Names that betray a column even when its correlation is mild. A 5-day forward
# label is future information whether or not it scores well against the target,
# so statistics alone cannot catch it -- but whoever named it knew what it was.
_LABEL_NAMES = ("label", "target", "outcome", "y_true", "ytrue", "ground_truth")
_FUTURE_NAMES = ("future", "fwd", "forward", "ahead", "t_plus", "tplus",
                 "lead_", "_lead", "nextday", "next_day")
_AFTER_NAMES = ("next_", "after_", "post_", "_post", "resolved", "final_")
_GENERIC_TARGETS = {"target", "label", "y", "outcome", "class", "result", "value"}
# Names that mark a column as a PAST value of something. A lagged target is a
# standard, legitimate feature; flagging it produces exactly the false positive
# that gets a checker switched off.
_LAG_NAMES = ("_lag", "lag_", "_prev", "prev_", "_prior", "prior_", "_last",
              "last_", "_ytd", "_l1y", "_1y", "_7d", "_28d", "_30d", "_90d",
              "trailing", "rolling", "_to_date", "historic", "_ago",
              "_yesterday", "_lastyear", "_last_year", "_past")


def _suspicious_name(name, target=None):
    """(severity, reason) for a column whose NAME implies it is not a feature."""
    n = str(name).lower()
    t = str(target).lower() if target else ""
    # A lagged copy of the target is one of the most common legitimate
    # features there is. `revenue_last_year` when predicting `revenue` is not
    # a leak, and telling someone to drop it is worse than saying nothing.
    if any(k in n for k in _LAG_NAMES):
        return None
    # A column carrying the target's whole name is a sibling of it: predicting
    # `label_cs` with `label_cs_5d` in the features is the same label at a
    # different horizon. Checked before the token rules below, because those
    # deliberately ignore any word the target itself contains - which would
    # otherwise silence exactly this case.
    if len(t) >= 5 and t not in _GENERIC_TARGETS and t in n and n != t:
        return ("critical", f"its name contains the target's own name {t!r}, so "
                            "it is most likely the same quantity at a different "
                            "horizon, aggregation or lag. Those are labels, not "
                            "features.")
    # A label-ish word stops being evidence only when the target IS that word.
    # Predicting a column called `target` says nothing about `customer_target_
    # group`. But predicting `label_cs` means this project prefixes its labels
    # with "label", so `label_sector_residual` is more suspicious, not less -
    # an earlier version had this backwards and went quiet on six real labels.
    generic = t in _GENERIC_TARGETS
    label_hits = [k for k in _LABEL_NAMES
                  if k in n and not (generic and k in t)]
    if label_hits:
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


def _looks_like_id(col, n_unique=None):
    """Identifiers are strings or integer codes -- never continuous floats.

    A float measurement is ~100% unique by nature, so a bare uniqueness test
    labels every real numeric feature an ID and skips the columns that matter.
    """
    n = col.nunique(dropna=True) if n_unique is None else n_unique
    if n <= 50 or len(col) == 0 or n / len(col) <= ID_UNIQUE_RATIO:
        return False
    if pd.api.types.is_float_dtype(col) or _is_timelike(col):
        return False  # a timestamp is unique per row but it is not an ID
    return True


def _plain(v):
    """Python scalar, so user-facing text never shows "np.int64(3)"."""
    return getattr(v, "item", lambda: v)()


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
    yb = pd.Series(np.asarray(y))
    # Group the integer codes, not the values: identical grouping (factorize
    # and groupby agree on dropping nulls) without materialising a whole
    # column of Python objects, which costs more than the grouping itself.
    codes, _ = pd.factorize(col, use_na_sentinel=True)
    keep = codes >= 0
    g = yb[keep].groupby(codes[keep], sort=False).agg(["count", "nunique"])
    pure = g[(g["count"] >= MIN_CATEGORY_SUPPORT) & (g["nunique"] == 1)]
    return float(pure["count"].sum() / len(col)) if len(col) else 0.0


def analyse(df, target, split=None, group=None, ignore=()):
    """Return a list of Findings, worst first.

    `ignore` names columns you have already reviewed and accepted. Without it
    a run that finds anything can never go green, so a team wires this into CI,
    watches it fail on a column they have deliberately kept, and deletes the
    check. Accepted columns are still reported, at info level, so a suppression
    stays visible instead of quietly hiding a later regression.
    """
    # Decide from content, not storage - see _as_content. Done once here so
    # every check below sees the same column whichever way the file was read.
    df = _normalise(df)

    dupes = sorted({str(c) for c in df.columns[df.columns.duplicated()]})
    if dupes:
        raise ValueError(
            f"duplicate column name(s) in the data: {dupes}. Every check on "
            "them silently fails, so a leak would be reported as clean. "
            "Rename or drop the duplicates first - a bad join or "
            "pd.concat(axis=1) is the usual cause.")

    if target not in df.columns:
        raise ValueError(f"target column {target!r} not in data: {list(df.columns)[:12]}")
    y = df[target]
    if y.isna().all():
        raise ValueError(f"target column {target!r} is entirely null")
    ignore = {str(c) for c in (ignore or ())}
    kind = _target_kind(y)
    if kind == "degenerate":
        raise ValueError(
            f"target column {target!r} has fewer than 2 distinct values - "
            "there is nothing to predict.")
    if kind == "unsupported":
        n_cls = int(y.dropna().nunique())
        n_lab = int(y.notna().sum())
        raise ValueError(
            f"target column {target!r} has {n_cls:,} distinct non-numeric "
            f"values across {n_lab:,} labelled rows - "
            f"{n_cls / max(n_lab, 1):.0%} of the rows hold a distinct value. "
            f"That reads as free text or an identifier rather than a label "
            f"(the ceiling is {MAX_CLASSES} classes, and fewer than "
            f"{TARGET_CARDINALITY_MAX:.0%} of rows being distinct). Point "
            "--target at the right column, or encode it if it really is the "
            "label.")
    binary = kind == "binary"
    findings = []
    skip = {target} | ({split} if split else set())
    features = [c for c in df.columns if c not in skip]
    if not features:
        raise ValueError("no feature columns left after excluding target/split")
    unknown = ignore - set(df.columns)
    if unknown:
        findings.append(Finding(
            "warning", "stale-ignore", None,
            f"ignored column(s) not present in the data: "
            f"{sorted(unknown)}. An ignore list that outlives its column "
            "silently stops protecting you - remove them."))
    scored_features = [c for c in features if c not in ignore]

    findings.append(Finding("info", "target", target,
                            f"{kind}, {len(df):,} rows, "
                            f"{len(features)} features"
                            + (f", {len(ignore & set(features))} ignored"
                               if ignore & set(features) else "")))

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
    # Captured BEFORE the unlabelled rows are dropped: a column that varies in
    # the file but is constant on the labelled rows is not a dead column, it is
    # an upstream job that never reached the training set. On real data that
    # was 17 features, including an entire ingestion pipeline, and it was
    # filed as a bare "no information" note.
    labelled = y.notna()
    all_labelled = bool(labelled.all())
    # Only worth a second full pass when some rows are unlabelled. If every row
    # has a target then "varies in the file" and "varies on the labelled rows"
    # are the same question, and the per-column loop below already answers it.
    varies_in_file = set() if all_labelled else {
        c for c in features if _safe_nunique(df[c]) > 1}
    if int(labelled.sum()) < 20:
        raise ValueError(
            f"only {int(labelled.sum())} rows have a target value - too few to "
            "assess. Check you named the right column.")
    if not all_labelled:
        # Worth almost nothing on pandas 3, where copy-on-write makes an
        # all-True .loc a lazy view (measured: +0 MB). Kept because pyproject
        # still supports pandas 2.x, where it is a real copy of the whole
        # table. The saving in this branch is the skipped pass above, not this.
        df = df.loc[labelled]
        y = y.loc[labelled]

    # --- names that give a column away regardless of its score -----------
    for c in scored_features:
        hit = _suspicious_name(c, target)
        if hit:
            findings.append(Finding(hit[0], "suspicious-name", c, hit[1]))

    # --- per-column predictive power -------------------------------------
    for c in scored_features:
        # One hostile column must not take the whole run down. Real frames
        # carry list- and dict-valued columns (embeddings, JSON metadata)
        # that raise on nunique(); a tool that dies on them gets uninstalled.
        try:
            findings.extend(
                _column_findings(c, df[c], y, kind, len(scored_features),
                                 varied_in_file=c in varies_in_file))
        except Exception as e:
            findings.append(Finding(
                "info", "unscoreable", c,
                f"could not analyse this column ({type(e).__name__}) - skipped. "
                "Values that are lists, dicts or other unhashable objects "
                "cannot be scored; flatten the column if it matters."))

    # --- dataset-level checks --------------------------------------------
    try:
        dup = int(df.duplicated().sum())
    except TypeError:
        dup = 0   # list/dict cells: nothing to compare, not a reason to die
    if dup:
        findings.append(Finding(
            "warning", "duplicate-rows", None,
            f"{dup:,} fully duplicated rows ({dup / len(df):.1%}). Under a random "
            "split these land on both sides and inflate test scores."))

    if split is not None:
        findings.extend(
            _split_checks(df, target, split, scored_features, group))

    for c in scored_features:
        try:
            if _is_timelike(df[c]):
                findings.append(Finding(
                    "warning", "temporal-column", c,
                    "looks like a date/time. If you split randomly rather than by "
                    "time, the model trains on the future to predict the past."))
        except Exception:
            pass

    separable = {f.column for f in findings
                 if f.column and f.severity in ("critical", "warning")
                 and f.kind in ("target-proxy", "suspiciously-predictive",
                                "pure-categories")}
    if (len(separable) >= WIDESPREAD_MIN and scored_features
            and len(separable) / len(scored_features) >= WIDESPREAD_SHARE):
        share = len(separable) / len(scored_features)
        findings.append(Finding(
            "warning", "widespread-separability", None,
            f"{len(separable)} of {len(scored_features)} columns "
            f"({share:.0%}) are individually predictive of the target. That is "
            "the signature of an easy problem rather than a leaking one - a "
            "leak is normally one or two columns, not this many. Read the "
            "strongest few and treat the rest as the dataset working."))

    for c in sorted(ignore & set(features)):
        findings.append(Finding(
            "info", "ignored", c,
            "excluded by the ignore list, so nothing was measured on it. "
            "A leak introduced here in future will not be reported."))

    order = {"critical": 0, "warning": 1, "info": 2}
    findings.sort(key=lambda f: order[f.severity])
    return findings


def _safe_nunique(col):
    try:
        return int(col.nunique(dropna=True))
    except Exception:
        return 0


def _column_findings(c, col, y, kind, n_features=1, varied_in_file=False):
    """Every check for one column. Raises freely; the caller isolates it."""
    binary = kind == "binary"
    findings = []
    n_unique = col.nunique(dropna=True)
    if _looks_like_id(col, n_unique):
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
        # A column can be perfectly predictive on a SUBSET of rows and still
        # score near chance overall. The Titanic `body` column - a body
        # recovery number - is present for 121 people, every one of whom died,
        # yet its missingness AUC is only 0.575 because it says nothing about
        # the other 1,188. Ranking metrics are blind to this; asking whether
        # either group has a single target value is not.
        for present in (True, False):
            grp = y[col.isna() != present]
            if len(grp) < MIN_CATEGORY_SUPPORT or grp.nunique() != 1 \
                    or pd.Series(y).nunique() <= 1:
                continue
            # "All 44 of these rows are negative" is not evidence when 99% of
            # every row is negative - the probability of that under the null
            # is 0.99^44 = 0.64. Without this test the check fired on 9 of 15
            # pure-noise columns at a 1% base rate. p is the exact chance of a
            # group this size landing on one class, Bonferroni'd over the
            # columns tested.
            rate = float((pd.Series(y) == grp.iloc[0]).mean())
            p_null = rate ** len(grp)
            if p_null * max(n_features, 1) > 0.01:
                continue
            state = "present" if present else "missing"
            findings.append(Finding(
                "critical", "missingness-leak", c,
                f"wherever this column is {state} ({len(grp):,} rows) the "
                f"target is always {_plain(grp.iloc[0])!r}, and that class is "
                f"only {rate:.1%} of the data overall. It does not have to "
                "predict every row to give the answer away on the rows it "
                "does cover.",
                _evidence_missing(col, y, binary)))
            break

        na = _score_column(col.isna().astype(int), y, kind, n_features)
        if na and na["score"] >= AUC_WARN and na["z"] >= na["z_min"] \
                and not any(f.kind == "missingness-leak" for f in findings):
            sev = "critical" if na["score"] >= AUC_CRITICAL else "warning"
            findings.append(Finding(
                sev, "missingness-leak", c,
                f"whether this column is missing predicts the target at "
                f"{na['score']:.4f} ({na_share:.0%} missing). The NaN pattern "
                "carries the answer even if the values do not.",
                _evidence_missing(col, y, binary)))

    m = _score_column(col, y, kind, n_features, n_unique)
    if m is None:
        if varied_in_file:
            findings.append(Finding(
                "warning", "dead-on-labelled-rows", c,
                "varies elsewhere in the file but holds one single value on "
                "every labelled row, so the model has never seen it change. "
                "Most often the labels and this column cover different "
                "periods and the join leaves a constant; it can also just "
                "mark which rows were labelled. Either way it contributes "
                "nothing to training."))
        else:
            findings.append(Finding(
                "info", "constant", c,
                "single value throughout - no information to learn from."))
        return findings
    score, metric, z, z_min = m["score"], m["metric"], m["z"], m["z_min"]

    def scored(ev):
        """Carry the measurement and its reference band alongside the evidence,
        so a report can present it the way a lab value is read: against the
        range a healthy column falls in."""
        return {**(ev or {"kind": "score_only"}), "score": float(score),
                "auc": m["auc"], "metric": metric, "z": float(z),
                "z_min": float(z_min),
                "band": [0.5, AUC_WARN, AUC_CRITICAL, 1.0]}

    # Direction is reported, not folded away: an inverted relationship is
    # equally leaky but the raw number has to reconcile with the user's own.
    raw = "" if m["auc"] is None else (
        f" (true {metric} {m['auc']:.4f}"
        + (", inverted)" if m["auc"] < 0.5 else ")"))
    strength = (f"separates the target at {score:.4f}{raw}, "
                f"{z:.1f} SE above chance")

    if score < AUC_WARN:
        pass
    elif z < z_min:
        # The absolute threshold is met but the sample is too small to tell
        # this apart from luck. Reporting it as a leak is how a checker earns
        # a reputation for crying wolf.
        findings.append(Finding(
            "info", "underpowered", c,
            f"{strength}, which does not clear the {z_min:.1f} SE bar for "
            f"{n_features} columns tested on {len(col):,} rows. On a sample "
            "this small a column of pure noise reaches that score by luck, so "
            "it is reported without a severity rather than called a leak.",
            scored(_evidence(col, y, binary))))
    elif score >= AUC_CRITICAL:
        findings.append(Finding(
            "critical", "target-proxy", c,
            f"alone {strength}. One column solving the target means one of two "
            "things: it was computed from the answer or recorded after it was "
            "known, or the problem really is this easy. Measurement cannot "
            "tell those apart - check when this value is written, and whether "
            "it exists unchanged at the moment you predict.",
            scored(_evidence(col, y, binary))))
    else:
        findings.append(Finding(
            "warning", "suspiciously-predictive", c,
            f"alone {strength}. Plausible for a genuinely strong feature, but "
            "confirm it exists at prediction time.",
            scored(_evidence(col, y, binary))))

    if not pd.api.types.is_numeric_dtype(col) or _looks_categorical(col):
        purity = _category_purity(col, y)
        if purity >= 0.5 and score >= AUC_WARN and z >= z_min:
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
    # head(50) BEFORE the cast, not after: astype(str) on a full object column
    # is a per-row Python call and 49,950 of them were being thrown away.
    sample = col.dropna().head(50).astype(str)
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
        # Row hashes, not tuples of Python objects. Materialising one tuple
        # per row held the entire frame a second time over in the interpreter;
        # on a wide 10M-row table that is the difference between a check and
        # an out-of-memory kill.
        try:
            ha = pd.util.hash_pandas_object(a[feat], index=False)
            hb = pd.util.hash_pandas_object(b[feat], index=False)
            # isin() against the Series, never set(...): a Python set of n
            # hashes costs ~70 bytes a row and reintroduces the per-row-object
            # blow-up the hashing exists to avoid. pandas matches in C.
            shared = int(hb.isin(ha).sum())
        except TypeError:
            return out  # unhashable cell values; nothing to compare
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
    for label, v in (("train", train), ("val", val)):
        f = float(v)
        # NaN slips through every comparison, and min(1.0, nan) returns 1.0 -
        # a crashed fold or an empty metric log became "overfitting, 100% gap".
        if f != f or f in (float("inf"), float("-inf")):
            raise ValueError(f"{label} score is not a finite number: {v!r}")
        if not -1.0 <= f <= 1.0:
            raise ValueError(
                f"{label} score {f} is outside [-1, 1]; pass the metric value, "
                "not a percentage")

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
    ap.add_argument("--ignore", default="", metavar="COLS",
                    help="comma-separated columns you have reviewed and "
                         "accepted; still listed, but never fail the run")
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

    ig = [c.strip() for c in a.ignore.split(",") if c.strip()]

    if (a.train is None) != (a.val is None):
        ap.error("--train and --val must be given together")

    if a.train is not None:
        result = diagnose(a.train, a.val, a.baseline, a.metric)
        target = None
    elif a.demo:
        result = analyse(demo_frame(), "churned", a.split, a.group, ig)
        target = "churned"
    elif a.data and a.target:
        result = analyse(load(a.data), a.target, a.split, a.group, ig)
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
