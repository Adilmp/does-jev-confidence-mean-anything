"""
STEP 6 - Can the overconfidence be fixed?

Step 3 found two things that sound contradictory but are not:

    AUC 0.903  - the ranking is good. Jev reliably puts worse comments above
                 better ones.
    ECE 0.157  - the numbers attached to that ranking are wrong. It says 85%
                 and delivers 47%.

When those two appear together the model is not confused about the task, it is
just speaking in the wrong units. That is a fixable problem, and fixing it is
a standard piece of ML work called recalibration: learn a function that maps
Jev's stated probability to the one that actually holds, and apply it to
everything afterward.

Two ways to learn that mapping:

  Platt    Fit an S-curve (a one-feature logistic regression). Two parameters,
           so it needs very little data, but it can only apply a smooth
           squash - if the distortion has a weird shape it cannot follow it.

  Isotonic Fit any monotonic step function (pool-adjacent-violators). More
           flexible, follows whatever shape the data has, but with enough
           freedom to memorise noise if the sample is small.

The discipline that makes this honest: fit on one half of the data, report on
the half the fit never saw. A calibration fitted and scored on the same rows
always looks perfect and means nothing.

Watch AUC in the output. Both methods are monotonic - they reorder nothing - so
AUC should barely move while ECE drops. That is the proof you rescaled the
probabilities rather than changed the underlying decisions.

    python3 6_recalibrate.py hard
"""

import sys
from importlib import import_module

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

measure = import_module("3_measure")
chart = import_module("4_chart")

SEED = 11
EPS = 1e-6

BEFORE = "#eb6834"   # categorical slot 2
AFTER = "#2a78d6"    # categorical slot 1


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(z):
    # Clip before exp: |z| past ~700 overflows float64 and spams warnings.
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def fit_platt(p, y, iters=100, l2=1e-6):
    """Logistic regression of the outcome on logit(p). Newton's method.

    Returns a function mapping a raw probability to a corrected one, plus the
    two fitted parameters: a slope (how much to squash confidence inward) and
    an intercept (how far to slide every prediction).

    The targets are smoothed rather than plain 0/1, which is what Platt's
    original paper does and it is not optional. When the two classes are
    almost perfectly separable - a rare label the model nails, say - the
    maximum-likelihood fit does not exist, the weights run off toward infinity,
    and you get an "intercept" of -200 that means nothing. Nudging the targets
    off the boundary keeps the optimum finite. The small ridge is belt and
    braces for the same failure.
    """
    y = np.asarray(y, dtype=float)
    n_pos, n_neg = float((y == 1).sum()), float((y == 0).sum())
    hi = (n_pos + 1.0) / (n_pos + 2.0)   # just under 1
    lo = 1.0 / (n_neg + 2.0)             # just over 0
    target = np.where(y == 1, hi, lo)

    X = np.column_stack([logit(p), np.ones(len(p))])
    beta = np.zeros(2)

    for _ in range(iters):
        pred = sigmoid(X @ beta)
        grad = X.T @ (pred - target) + l2 * beta
        W = np.clip(pred * (1 - pred), 1e-12, None)
        H = X.T @ (X * W[:, None]) + (l2 + 1e-10) * np.eye(2)
        step = np.linalg.solve(H, grad)
        beta -= step
        if np.max(np.abs(step)) < 1e-10:
            break

    return lambda q: sigmoid(np.column_stack([logit(q), np.ones(len(q))]) @ beta), beta


def pava(y):
    """Pool adjacent violators: the closest non-decreasing fit to y.

    Walk left to right keeping a stack of blocks. Whenever the new block is
    lower than the one before it, the pair violates monotonicity, so merge them
    into their weighted average and check again. What is left is non-decreasing
    by construction.
    """
    stack = []  # each entry is [value, how many originals it covers]
    for value in y:
        stack.append([float(value), 1])
        while len(stack) > 1 and stack[-2][0] > stack[-1][0]:
            v2, c2 = stack.pop()
            v1, c1 = stack.pop()
            stack.append([(v1 * c1 + v2 * c2) / (c1 + c2), c1 + c2])

    out, i = np.empty(len(y)), 0
    for value, count in stack:
        out[i:i + count] = value
        i += count
    return out


def fit_isotonic(p, y):
    """Monotonic step fit, then linear interpolation for unseen points."""
    order = np.argsort(p)
    xs, ys = np.asarray(p)[order], pava(np.asarray(y, dtype=float)[order])
    # np.interp clamps outside the fitted range, which is what we want: never
    # extrapolate a calibration beyond the confidences actually observed.
    return lambda q: np.interp(q, xs, ys), (xs, ys)


def split(df):
    """Half to fit on, half to report on. Stratified so each question and each
    outcome is represented in both halves."""
    rng = np.random.default_rng(SEED)
    fit_mask = np.zeros(len(df), dtype=bool)
    for _, idx in df.groupby(["question", "outcome"]).groups.items():
        pos = df.index.get_indexer(idx)
        chosen = rng.permutation(pos)[: len(pos) // 2]
        fit_mask[chosen] = True
    return df[fit_mask], df[~fit_mask]


def scored(df, p):
    part = df.assign(p=p)
    return measure.summarise(part), measure.buckets(part)


def plot(before_tbl, after_tbl, stats_b, stats_a, name):
    fig, ax = plt.subplots(figsize=(8.0, 7.4), dpi=200)
    ax.plot([0, 1], [0, 1], color=chart.MUTED, linewidth=1.2, zorder=1)

    for tbl, colour, label in (
        (before_tbl, BEFORE, f"Jev as shipped · ECE {stats_b['ECE']:.3f}"),
        (after_tbl, AFTER, f"after recalibration · ECE {stats_a['ECE']:.3f}"),
    ):
        x, y = tbl["jev_said"].to_numpy(), tbl["actually_was"].to_numpy()
        ax.plot(x, y, color=colour, linewidth=2, zorder=3, label=label)

        # Hollow = too few judgments in that band to mean anything. The held-out
        # half is only ~800 rows, so the top bands get thin and a sharp kink up
        # there is sampling noise, not a real wobble in the calibration.
        solid = (tbl["n"] >= chart.MIN_N).to_numpy()
        ax.plot(x[solid], y[solid], "o", markersize=8, color=colour,
                markeredgecolor=chart.SURFACE, markeredgewidth=2,
                linestyle="none", zorder=4)
        ax.plot(x[~solid], y[~solid], "o", markersize=7, color=chart.SURFACE,
                markeredgecolor=colour, markeredgewidth=1.6,
                linestyle="none", zorder=4)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("Stated confidence", fontsize=10, labelpad=9)
    ax.set_ylabel("How often humans actually flagged it", fontsize=10, labelpad=9)
    leg = ax.legend(loc="upper left", frameon=False, fontsize=9)
    for text in leg.get_texts():
        text.set_color(chart.INK_2)
    chart.style(ax)

    fig.suptitle("The ranking was fine. The numbers needed rescaling.",
                 fontsize=15, color=chart.INK, x=0.055, ha="left", y=1.0)
    fig.text(0.055, 0.945,
             f"Held-out half, never seen during fitting · AUC {stats_b['AUC']:.3f} "
             f"-> {stats_a['AUC']:.3f} (unchanged: recalibration reorders nothing)",
             fontsize=9.5, color=chart.INK_2, ha="left")

    out = f"results/{name}_recalibrated.png"
    fig.savefig(out, bbox_inches="tight", facecolor=chart.SURFACE, pad_inches=0.42)
    plt.close(fig)
    return out


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: python3 6_recalibrate.py <natural|hard>")
    name = sys.argv[1]

    df = measure.load(name, simulate=False).reset_index(drop=True)
    fit_df, eval_df = split(df)
    print(f"{len(fit_df)} judgments to fit on, {len(eval_df)} held out\n")

    platt, beta = fit_platt(fit_df["p"].to_numpy(), fit_df["outcome"].to_numpy())
    iso, _ = fit_isotonic(fit_df["p"].to_numpy(), fit_df["outcome"].to_numpy())

    rows = {}
    tables = {}
    for label, p in (
        ("as shipped", eval_df["p"].to_numpy()),
        ("Platt", platt(eval_df["p"].to_numpy())),
        ("isotonic", iso(eval_df["p"].to_numpy())),
    ):
        stats, table = scored(eval_df, p)
        rows[label] = stats
        tables[label] = (stats, table)

    out = pd.DataFrame(rows).T[["ECE", "Brier", "AUC", "accuracy_at_0.5"]]
    pd.set_option("display.width", 200)
    print("=== Held-out half " + "=" * 45)
    print(out.round(4).to_string())

    thin = tables["isotonic"][1]
    thin = thin[thin["n"] < chart.MIN_N]
    if not thin.empty:
        print(f"\n  Thin bands (n < {chart.MIN_N}), drawn hollow and not worth "
              f"reading: {', '.join(thin['bucket'])}")

    slope, intercept = beta
    print(f"\nPlatt fit: corrected = sigmoid({slope:.3f} * logit(p) {intercept:+.3f})")
    print("  Two knobs, and which one does the work is the actual diagnosis:")
    print("    slope < 1      -> confidence is too EXTREME; squash it inward.")
    print("    intercept != 0 -> every prediction is SHIFTED one way; slide it back.")

    if abs(intercept) > 3 * abs(1 - slope):
        leaning = "yes" if intercept < 0 else "no"
        print(f"\n  Here the intercept ({intercept:+.3f}) dominates and the slope "
              f"({slope:.3f}) is\n  nearly 1. So this is not classic overconfidence - "
              f"the curve is not too\n  steep, it is bodily shifted. Jev leans "
              f"'{leaning}' at every confidence\n  level, low ones included. That is "
              f"a systematic bias, and it is the easier\n  of the two faults to fix: "
              f"essentially one number.")
    elif slope < 1:
        print(f"\n  Here the slope ({slope:.3f}) does the work: textbook "
              f"overconfidence.\n  The fix is to shrink confidence toward the middle.")

    best = out["ECE"].idxmin()
    shipped = out.loc["as shipped", "ECE"]
    if best != "as shipped":
        drop = (1 - out.loc[best, "ECE"] / shipped) * 100
        print(f"\n=== The sentence for your post " + "=" * 32)
        print(f"  {best} recalibration cuts calibration error {drop:.0f}% "
              f"({shipped:.3f} -> {out.loc[best,'ECE']:.3f})")
        print(f"  on data it never saw, while AUC moves "
              f"{out.loc['as shipped','AUC']:.3f} -> {out.loc[best,'AUC']:.3f}.")
        print(f"  The ranking was always there. Only the units were wrong.")

    out.to_csv(f"results/{name}_recalibrated.csv")
    print("\nwrote", plot(tables["as shipped"][1], tables[best][1],
                          tables["as shipped"][0], tables[best][0], name))
    print(f"wrote results/{name}_recalibrated.csv")


if __name__ == "__main__":
    main()
