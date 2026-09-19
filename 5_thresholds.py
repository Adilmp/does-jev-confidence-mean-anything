"""
STEP 5 - The part a hiring manager actually cares about.

Step 3 tells you whether the probabilities are honest. This one turns that into
a decision you could defend in a design review:

    "We auto-handle anything Jev is more than 88% sure about in either
     direction. That covers 61% of the queue at 95% accuracy. The other
     39% goes to a human."

The rule is a band around 0.5. Anything confidently toxic gets auto-flagged,
anything confidently clean gets auto-cleared, and the uncertain middle is
escalated. Widening the band buys accuracy on what you automate and costs you
coverage. That trade is the whole design decision, and it is a curve, not a
vibe.

Note the honest caveat, and say it out loud in the post: this assumes the
comments you see in production look like the ones you measured on. Run it on
natural.csv and hard.csv and you will see how much that assumption is worth.

    python3 5_thresholds.py hard
    python3 5_thresholds.py hard --simulate
"""

import json
import os
import sys
from importlib import import_module

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

measure = import_module("3_measure")
chart = import_module("4_chart")

TARGET_ACCURACY = 0.95   # what you promise about anything you automate
HUMAN_COST = 0.10        # $ per escalated comment. Change to your real number.
PRICE_PER_MTOK = 0.042   # TypeSafe's published input price


def measured_jev_cost(name, simulate):
    """Dollars per judgment, from the tokens the API actually billed.

    Worth doing rather than assuming: a first guess here was 63x too high,
    which would have made Jev look like a material share of the bill when it
    is a rounding error. The whole cost story is human escalation - but you
    only get to say that if you measured instead of estimated.
    """
    path = f"results/{name}_predictions{'.SIMULATED' if simulate else ''}.jsonl"
    tokens = judgments = 0
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("input_tokens"):
                    tokens += r["input_tokens"]
                    judgments += len(r.get("pred") or {})
    if not judgments:
        return None
    return tokens / judgments * PRICE_PER_MTOK / 1e6


def sweep(df, jev_cost):
    rows = []
    for w in np.arange(0.0, 0.50, 0.01):
        hi, lo = 0.5 + w, 0.5 - w
        auto = (df["p"] >= hi) | (df["p"] <= lo)
        n_auto = int(auto.sum())
        if n_auto == 0:
            continue

        part = df[auto]
        pred, truth = (part["p"] >= 0.5).astype(int), part["outcome"]
        correct = int((pred == truth).sum())
        n_esc = len(df) - n_auto

        # Accuracy alone is a trap when positives are rare: at a 3% base rate
        # "flag nothing" scores 97%. Precision and recall are what a moderation
        # queue is actually judged on, so carry them alongside.
        tp = int(((pred == 1) & (truth == 1)).sum())
        fp = int(((pred == 1) & (truth == 0)).sum())
        fn = int(((pred == 0) & (truth == 1)).sum())

        rows.append(
            {
                "band": f"{lo:.2f}-{hi:.2f}",
                "half_width": round(w, 2),
                "coverage": n_auto / len(df),
                "accuracy_on_auto": correct / n_auto,
                "precision": tp / (tp + fp) if tp + fp else float("nan"),
                "recall": tp / (tp + fn) if tp + fn else float("nan"),
                "mistakes_automated": n_auto - correct,
                "escalated": n_esc,
                # Every judgment costs a Jev call; only escalated ones also
                # cost a human.
                "cost_per_1k": (len(df) * jev_cost + n_esc * HUMAN_COST)
                / len(df)
                * 1000,
            }
        )
    return pd.DataFrame(rows)


def plot(s, name, simulate, pick):
    fig, ax = plt.subplots(figsize=(8.2, 6.2), dpi=200)

    ax.axhline(TARGET_ACCURACY, color=chart.MUTED, linewidth=1.2, zorder=1)
    # Anchor the label in axes coords, not data coords - the x-range here starts
    # wherever the widest band lands, so a data-space x of 0.02 falls off-plot.
    ax.annotate(f"{TARGET_ACCURACY:.0%} target",
                xy=(0.012, TARGET_ACCURACY), xycoords=("axes fraction", "data"),
                xytext=(0, 4), textcoords="offset points",
                fontsize=8.5, color=chart.MUTED, ha="left", va="bottom")

    ax.plot(s["coverage"], s["accuracy_on_auto"], color=chart.SERIES,
            linewidth=2, zorder=3)

    # Direct-label a few points with the confidence band that produces them,
    # rather than a number on every dot. Labels go down-left, into the empty
    # region under a curve that falls left to right, so they never cross it.
    for w in (0.10, 0.20, 0.30, 0.40):
        row = s.iloc[(s["half_width"] - w).abs().idxmin()]
        ax.plot(row["coverage"], row["accuracy_on_auto"], "o", markersize=7,
                color=chart.SERIES, markeredgecolor=chart.SURFACE,
                markeredgewidth=2, zorder=4)
        ax.annotate(f"p>{0.5 + w:.2f}", xy=(row["coverage"], row["accuracy_on_auto"]),
                    xytext=(-8, -15), textcoords="offset points", ha="right",
                    fontsize=8, color=chart.INK_2, zorder=5)

    if pick is not None:
        ax.plot(pick["coverage"], pick["accuracy_on_auto"], "o", markersize=13,
                color=chart.SURFACE, markeredgecolor=chart.OVER,
                markeredgewidth=2.4, zorder=5)

    ax.set_xlabel("Share of the queue handled automatically", fontsize=10, labelpad=9)
    ax.set_ylabel("Accuracy on what you automated", fontsize=10, labelpad=9)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    chart.style(ax)

    title = "How much can you safely automate?"
    if simulate:
        title = "[SIMULATED DATA] " + title
    if pick is not None and pick["half_width"] == 0:
        sub = (f"This sample clears {TARGET_ACCURACY:.0%} with nothing escalated. "
               f"Raise the target, or test on harder data.")
    elif pick is not None:
        sub = (f"Widest band hitting {TARGET_ACCURACY:.0%}: automate "
               f"{pick['coverage']:.0%} of the queue, escalate the rest.")
    else:
        sub = (f"No confidence band reaches {TARGET_ACCURACY:.0%} accuracy on "
               f"this data. That is the finding.")

    fig.suptitle(title, fontsize=15, color=chart.INK, x=0.055, ha="left", y=1.0)
    fig.text(0.055, 0.945, sub, fontsize=9.5, color=chart.INK_2, ha="left")

    out = f"results/{name}_thresholds{'.SIMULATED' if simulate else ''}.png"
    fig.savefig(out, bbox_inches="tight", facecolor=chart.SURFACE, pad_inches=0.42)
    plt.close(fig)
    return out


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: python3 5_thresholds.py <natural|hard> [--simulate]")
    name, simulate = sys.argv[1], "--simulate" in sys.argv

    df = measure.load(name, simulate)
    if simulate:
        print("!! SIMULATED DATA - not real Jev output.\n")

    jev_cost = measured_jev_cost(name, simulate)
    if jev_cost is None:
        sys.exit("No billed token counts in the predictions file - re-run step 2.")
    print(f"Jev cost per judgment, from billed tokens: ${jev_cost:.8f}\n")

    # The baseline that makes accuracy meaningless, stated before anything else
    # so it cannot be skipped over.
    base = df["outcome"].mean()
    lazy = max(base, 1 - base)
    print(f"Base rate: {base:.1%} of judgments are a real 'yes'.")
    print(f"A constant '{'yes' if base > 0.5 else 'no'}' predictor scores "
          f"{lazy:.1%} accuracy for free, catching nothing.")
    if lazy > 0.9:
        print("  ^ At this base rate ACCURACY IS A TRAP. Read precision and")
        print("    recall below instead, and treat any accuracy number that")
        print("    does not beat the line above as worthless.\n")
    else:
        print()

    s = sweep(df, jev_cost)
    pd.set_option("display.width", 200)
    print(s[s["half_width"] % 0.05 < 0.005].round(3).to_string(index=False))

    ok = s[s["accuracy_on_auto"] >= TARGET_ACCURACY]
    pick = ok.loc[ok["coverage"].idxmax()] if not ok.empty else None

    print("\n=== The sentence for your post " + "=" * 32)
    if pick is not None and pick["half_width"] == 0:
        # Degenerate case: accuracy clears the target with nothing escalated, so
        # the "widest band" is no band at all. Say that, rather than printing
        # "auto-handle anything outside 0.50-0.50", which means nothing.
        print(
            f"  Accuracy is already {pick['accuracy_on_auto']:.1%} with nothing "
            f"escalated, so no confidence\n  band is needed to hit "
            f"{TARGET_ACCURACY:.0%} on this sample. Raise TARGET_ACCURACY to find "
            f"where\n  escalation starts to matter - and be suspicious: an easy "
            f"sample flatters the model."
        )
    elif pick is not None:
        print(
            f"  Auto-handle anything outside {pick['band']}. That covers "
            f"{pick['coverage']:.0%} of the queue at {pick['accuracy_on_auto']:.1%} "
            f"accuracy,\n  escalating {int(pick['escalated'])} of {len(df)} "
            f"judgments to a human."
        )
    if pick is not None:
        print(
            f"  Cost per 1,000 comments: ${pick['cost_per_1k']:.2f} "
            f"(vs ${HUMAN_COST * 1000:.2f} all-human) "
            f"— assuming ${HUMAN_COST:.2f}/review."
        )
        print(f"  {int(pick['mistakes_automated'])} wrong calls still got automated. "
              f"Decide if you can live with that.")
    else:
        print(f"  Nothing reaches {TARGET_ACCURACY:.0%}. Either lower the bar, "
              f"escalate everything,\n  or this model is not ready for this task. "
              f"A null result is still a result.")

    suffix = ".SIMULATED" if simulate else ""
    s.to_csv(f"results/{name}_thresholds{suffix}.csv", index=False)
    print("\nwrote", plot(s, name, simulate, pick))
    print(f"wrote results/{name}_thresholds{suffix}.csv")


if __name__ == "__main__":
    main()
