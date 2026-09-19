"""
STEP 4 - Draw the chart. This is the image that goes in the post.

Two figures:

  <name>_calibration.png   The headline. One curve against the diagonal.
                           On the diagonal = the probabilities are honest.
                           Below it    = Jev claimed more certainty than it earned.
                           The bar strip underneath shows how many predictions
                           landed in each band, because a dot built from 4
                           comments is noise and should not be read as a trend.

  <name>_by_question.png   The same thing split by question. Worth looking at:
                           a model is often well behaved on the common case
                           (toxicity) and badly overconfident on the rare one
                           (threat), and the pooled curve hides that.

    python3 4_chart.py hard
    python3 4_chart.py hard --simulate
"""

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from importlib import import_module

measure = import_module("3_measure")

# Palette. One series, so one hue - the story is the gap, not the categories.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SERIES = "#2a78d6"
SERIES_SOFT = "#cde2fb"
OVER = "#d03b3b"   # claimed more than it delivered
UNDER = "#2a78d6"  # claimed less than it delivered

MIN_N = 8  # bands thinner than this are drawn hollow - too few to trust

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Segoe UI", "Helvetica", "Arial"],
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": "#c3c2b7",
        "axes.labelcolor": INK_2,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def style(ax):
    ax.grid(True, color=GRID, linewidth=0.8, alpha=1.0)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_linewidth(0.8)


def reliability(ax, table, label_diag=True):
    """The curve plus the reference diagonal, with the gap shaded."""
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1.2, zorder=1)

    x = table["jev_said"].to_numpy()
    y = table["actually_was"].to_numpy()

    if label_diag:
        # Park the label in whichever corner the curve is not using. An
        # overconfident curve hugs the area below the diagonal, leaving the
        # upper-left empty; an underconfident one leaves the lower-right.
        overconfident = float((y <= x).mean()) >= 0.5
        anchor = (0.36, 0.36) if overconfident else (0.64, 0.64)
        text_at = (0.06, 0.66) if overconfident else (0.54, 0.20)
        ax.annotate(
            "perfect calibration\n(claim = reality)",
            xy=anchor,
            xytext=text_at,
            fontsize=8.5,
            color=MUTED,
            ha="left",
            zorder=5,
            arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.8),
        )

    # Shade the gap. Red where it overclaimed, blue where it underclaimed.
    ax.fill_between(x, y, x, where=y <= x, color=OVER, alpha=0.11, zorder=2,
                    interpolate=True)
    ax.fill_between(x, y, x, where=y > x, color=UNDER, alpha=0.11, zorder=2,
                    interpolate=True)

    ax.plot(x, y, color=SERIES, linewidth=2, zorder=3)

    # Hollow markers where the band has too few comments to mean anything.
    solid = table["n"] >= MIN_N
    ax.plot(x[solid], y[solid], "o", markersize=8, color=SERIES,
            markeredgecolor=SURFACE, markeredgewidth=2, zorder=4, linestyle="none")
    ax.plot(x[~solid], y[~solid], "o", markersize=7, color=SURFACE,
            markeredgecolor=SERIES, markeredgewidth=1.6, zorder=4, linestyle="none")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    style(ax)


def headline(df, table, stats, name, simulate):
    fig = plt.figure(figsize=(8.2, 9.2), dpi=200)
    gs = fig.add_gridspec(2, 1, height_ratios=[3.1, 1], hspace=0.22)
    ax, axc = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    reliability(ax, table)
    ax.set_xlabel("What Jev said its confidence was", fontsize=10, labelpad=9)
    ax.set_ylabel("How often humans actually flagged it", fontsize=10, labelpad=9)

    # The one number that matters, as a stat tile rather than buried in a legend.
    ax.text(0.035, 0.955, f"{stats['ECE']:.3f}", transform=ax.transAxes,
            fontsize=30, color=INK, va="top", ha="left")
    ax.text(0.035, 0.876, "calibration error (ECE)\nlower is better · 0 is perfect",
            transform=ax.transAxes, fontsize=8.5, color=INK_2, va="top", ha="left")
    ax.text(0.035, 0.775,
            f"ranking quality (AUC) {stats['AUC']:.3f}\n{int(stats['n'])} judgments",
            transform=ax.transAxes, fontsize=8.5, color=MUTED, va="top", ha="left")

    # Counts strip - context for which dots above are trustworthy.
    # Bars sit at BUCKET CENTRES, not at mean confidence, so they line up with
    # the bands in the panel above instead of drifting toward whichever end of
    # the band the predictions happened to cluster in.
    centres = [(float(b.split("-")[0]) + float(b.split("-")[1])) / 2
               for b in table["bucket"]]
    axc.bar(centres, table["n"], width=0.09, color=SERIES_SOFT,
            edgecolor=SURFACE, linewidth=2)
    axc.set_xlim(0, 1)
    axc.set_xlabel("What Jev said its confidence was", fontsize=10, labelpad=9)
    axc.set_ylabel("judgments\nin this band", fontsize=9, labelpad=9)
    style(axc)

    # Only call out a band with enough comments behind it. The biggest gap is
    # usually in a near-empty band, and pointing at 4 comments is not a finding.
    solid = table[table["n"] >= MIN_N]
    worst = (solid if not solid.empty else table).loc[
        lambda t: t["gap"].abs().idxmax()
    ]
    direction = "overconfident" if worst["gap"] < 0 else "underconfident"
    sub = (
        f"Worst band (n={int(worst['n'])}): Jev said {worst['jev_said']:.0%}, "
        f"humans flagged {worst['actually_was']:.0%} — {direction} by "
        f"{abs(worst['gap']) * 100:.0f} percentage points."
    )

    title = "Does Jev's confidence mean anything?"
    if simulate:
        title = "[SIMULATED DATA] " + title
    fig.suptitle(title, fontsize=15, color=INK, x=0.055, ha="left", y=0.975)
    fig.text(0.055, 0.939, sub, fontsize=9.5, color=INK_2, ha="left")
    # `--strict` only renames the results, not the data. Saying
    # "natural_strict.csv" on a published chart would name a file that does not
    # exist, so credit the real dataset and note the wording separately.
    dataset = name[: -len("_strict")] if name.endswith("_strict") else name
    wording = " · tightened criteria" if name.endswith("_strict") else ""
    fig.text(0.055, 0.022,
             f"{dataset}.csv · civil_comments{wording} · ground truth = share "
             f"of human annotators who flagged the comment",
             fontsize=8, color=MUTED, ha="left")

    # The reliability panel is locked to a square aspect so the diagonal really
    # is 45 degrees, which shrinks it inside its grid slot. Draw once, then pull
    # the counts strip to the same x-extent - otherwise the two panels share an
    # x-axis visually but not actually, and the bars point at the wrong bands.
    fig.canvas.draw()
    top, bottom = ax.get_position(original=False), axc.get_position(original=False)
    axc.set_position([top.x0, bottom.y0, top.width, bottom.height])

    out = f"results/{name}_calibration{'.SIMULATED' if simulate else ''}.png"
    fig.savefig(out, bbox_inches="tight", facecolor=SURFACE, pad_inches=0.42)
    plt.close(fig)
    return out


def by_question(df, name, simulate):
    questions = sorted(df["question"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 10.2), dpi=200)

    for ax, q in zip(axes.flat, questions):
        part = df[df["question"] == q]
        reliability(ax, measure.buckets(part), label_diag=False)
        s = measure.summarise(part)
        # pad leaves room for the base-rate line to sit between title and axes.
        ax.set_title(f"{q}   ECE {s['ECE']:.3f}   AUC {s['AUC']:.3f}",
                     fontsize=10.5, color=INK, pad=26, loc="left")
        # Say the base rate out loud. A question humans flagged 1% of the time
        # has almost no positives to check against, so its curve swings wildly
        # on a handful of comments. That is a property of the data, not a
        # finding about the model, and the reader needs to know which is which.
        rare = s["base_rate"] < 0.05
        note = f"{s['base_rate']:.1%} of comments flagged by humans"
        ax.text(0.0, 1.015, note + (" - too rare to trust" if rare else ""),
                transform=ax.transAxes, fontsize=8.5,
                color=OVER if rare else MUTED, va="bottom")
        ax.tick_params(labelsize=8)

    for ax in axes.flat[len(questions):]:
        ax.set_visible(False)

    fig.supxlabel("What Jev said", fontsize=10, color=INK_2)
    fig.supylabel("What humans actually did", fontsize=10, color=INK_2)

    title = "Calibration is not uniform across questions"
    if simulate:
        title = "[SIMULATED DATA] " + title
    fig.suptitle(title, fontsize=14, color=INK, x=0.055, ha="left", y=0.99)
    fig.text(0.055, 0.962,
             "Red base rates are too rare to draw conclusions from - too few "
             "positives to check against.",
             fontsize=9, color=INK_2, ha="left")
    # h_pad keeps the lower row's titles off the upper row's tick labels.
    fig.tight_layout(rect=[0.03, 0.02, 1, 0.945], h_pad=4.0)

    out = f"results/{name}_by_question{'.SIMULATED' if simulate else ''}.png"
    fig.savefig(out, bbox_inches="tight", facecolor=SURFACE, pad_inches=0.42)
    plt.close(fig)
    return out


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: python3 4_chart.py <natural|hard> [--simulate]")
    name, simulate = sys.argv[1], "--simulate" in sys.argv

    df = measure.load(name, simulate)
    table = measure.buckets(df)
    stats = measure.summarise(df)

    print("wrote", headline(df, table, stats, name, simulate))
    print("wrote", by_question(df, name, simulate))


if __name__ == "__main__":
    main()
