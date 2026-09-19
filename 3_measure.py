"""
STEP 3 - The actual measurement.

The idea in one sentence: take every comment where Jev said "80% sure", and
check how many of them the humans actually flagged. If it is about 80%, the
number means something. If it is 55%, the number is decoration.

To do that we sort predictions into ten buckets by confidence (0.0-0.1,
0.1-0.2, ... 0.9-1.0) and compare, inside each bucket, what Jev claimed against
what actually happened.

Three numbers come out of it:

  ECE   Expected Calibration Error. The average gap between what Jev claimed
        and what happened, weighted by how many comments fell in each bucket.
        0 is perfect. Under ~0.05 is good. Over ~0.15 means the probabilities
        should not be trusted as probabilities.

  Brier The average squared error of the probability. Unlike ECE it punishes
        being wrong AND being vague, so a model that says 0.5 to everything
        scores badly here even though it looks honest to ECE.

  AUC   Whether the ranking is any good, ignoring the actual numbers. This is
        the important one to report alongside ECE, because a model can sort
        comments perfectly (AUC 0.95) while its probabilities are complete
        nonsense (ECE 0.30). Those two failures need different fixes: bad
        ranking means the model can't do the task, bad calibration just means
        the numbers need rescaling.

    python3 3_measure.py hard
    python3 3_measure.py hard --simulate
"""

import json
import os
import sys

import numpy as np
import pandas as pd

N_BUCKETS = 10
HUMAN_THRESHOLD = 0.5  # "most annotators flagged it" counts as a yes


def load(name, simulate):
    suffix = ".SIMULATED" if simulate else ""
    path = f"results/{name}_predictions{suffix}.jsonl"
    if not os.path.exists(path):
        sys.exit(f"No predictions at {path}. Run 2_ask_jev.py first.")

    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    rows = []
    for r in records:
        for question, p in r["pred"].items():
            rows.append(
                {
                    "question": question,
                    "p": float(p),
                    "human_fraction": float(r["human"][question]),
                    "outcome": int(float(r["human"][question]) >= HUMAN_THRESHOLD),
                    "latency_s": r.get("latency_s", float("nan")),
                }
            )
    return pd.DataFrame(rows)


def auc(p, y):
    """Area under the ROC curve, via average ranks. No sklearn needed.

    Reads as: pick a random flagged comment and a random clean one. How often
    does the model give the flagged one a higher score? 0.5 is a coin flip.
    """
    pos, neg = int(y.sum()), int((1 - y).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    ranks = pd.Series(p).rank().to_numpy()
    return (ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg)


def buckets(df):
    """Group predictions into confidence bands and compare claim vs reality."""
    edges = np.linspace(0, 1, N_BUCKETS + 1)
    # right=False keeps 0.0 in bucket 0; clip keeps a p of exactly 1.0 in the last.
    idx = np.clip(np.digitize(df["p"], edges, right=False) - 1, 0, N_BUCKETS - 1)

    out = []
    for b in range(N_BUCKETS):
        part = df[idx == b]
        if part.empty:
            continue
        out.append(
            {
                "bucket": f"{edges[b]:.1f}-{edges[b+1]:.1f}",
                "n": len(part),
                "jev_said": part["p"].mean(),
                "actually_was": part["outcome"].mean(),
                "human_agreement": part["human_fraction"].mean(),
            }
        )
    out = pd.DataFrame(out)
    out["gap"] = out["actually_was"] - out["jev_said"]
    return out


def summarise(df):
    ece = 0.0
    b = buckets(df)
    for _, row in b.iterrows():
        ece += (row["n"] / len(df)) * abs(row["gap"])
    return {
        "n": len(df),
        "base_rate": df["outcome"].mean(),
        "accuracy_at_0.5": ((df["p"] >= 0.5).astype(int) == df["outcome"]).mean(),
        "ECE": ece,
        "Brier": ((df["p"] - df["outcome"]) ** 2).mean(),
        "AUC": auc(df["p"].to_numpy(), df["outcome"].to_numpy()),
        "corr_with_human_spread": df["p"].corr(df["human_fraction"]),
    }


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: python3 3_measure.py <natural|hard> [--simulate]")
    name, simulate = sys.argv[1], "--simulate" in sys.argv

    df = load(name, simulate)
    if simulate:
        print("!! SIMULATED DATA - not real Jev output.\n")

    rows = []
    for question in ["ALL"] + sorted(df["question"].unique()):
        part = df if question == "ALL" else df[df["question"] == question]
        rows.append({"question": question, **summarise(part)})
    summary = pd.DataFrame(rows).set_index("question")

    pd.set_option("display.width", 200)
    print(f"=== {name}.csv " + "=" * 50)
    print(summary.round(3).to_string())

    print("\n=== Calibration table, all questions pooled " + "=" * 18)
    table = buckets(df)
    print(table.round(3).to_string(index=False))
    print(
        "\n  jev_said       = average confidence Jev reported in that band"
        "\n  actually_was   = fraction the humans actually flagged"
        "\n  gap            = reality minus claim. Negative = overconfident."
    )

    overall = summary.loc["ALL"]
    print("\n=== Read this " + "=" * 48)
    print(f"  Ranking (AUC):     {overall['AUC']:.3f}", end="  ")
    print("- can it tell flagged from clean at all?")
    print(f"  Honesty (ECE):     {overall['ECE']:.3f}", end="  ")
    print("- do the numbers mean what they say?")

    if overall["AUC"] > 0.8 and overall["ECE"] > 0.12:
        print("\n  Good ranking, dishonest numbers. The model CAN do the task,")
        print("  but its probabilities are not usable as probabilities. Anyone")
        print("  writing `if p > 0.9` against this is not getting what they think.")
    elif overall["ECE"] <= 0.06:
        print("\n  Well calibrated. The probabilities can be used directly as")
        print("  thresholds, which is the whole promise. Step 5 finds where.")

    suffix = ".SIMULATED" if simulate else ""
    summary.to_csv(f"results/{name}_summary{suffix}.csv")
    table.to_csv(f"results/{name}_buckets{suffix}.csv", index=False)
    df.to_csv(f"results/{name}_flat{suffix}.csv", index=False)
    print(f"\nSaved -> results/{name}_summary{suffix}.csv (+ buckets, flat)")


if __name__ == "__main__":
    main()
