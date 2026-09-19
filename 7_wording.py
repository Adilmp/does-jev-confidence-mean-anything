"""
STEP 7 - Is the bias mine or the model's?

The obvious objection to this whole project: "your criteria were badly worded,
so of course it over-flags." That is a fair challenge and it is testable.

Four wordings of the SAME question go into ONE call per comment, as four
independent questions over the same state. Jev answers them in parallel and
they cannot see one another, so the only thing that varies is the phrasing -
same model, same comment, same moment, same everything else.

  v1_original    what the main experiment used
  v2_annotator   mirrors the wording the human raters were actually given
  v3_strict      a deliberately high bar, with explicit exclusions
  v4_bare        instructions only, no criteria at all

Then compare the Platt intercept across the four. That is the right statistic
because step 6 showed the miscalibration is a SHIFT, and the intercept is
exactly how big the shift is.

  If wording is the cause  -> the intercept swings a lot between variants, and
                              v3_strict lands near zero.
  If the model is the cause -> all four stay stubbornly negative and the range
                              between them is small next to the shift itself.

    python3 7_wording.py hard
    python3 7_wording.py natural
"""

import csv
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from importlib import import_module

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import jev

measure = import_module("3_measure")
chart = import_module("4_chart")
recal = import_module("6_recalibrate")

WORKERS = int(os.environ.get("JEV_WORKERS", "8"))
PRICE_PER_MTOK = 0.042
TRUTH_COLUMN = "insult"   # the human label all four variants are scored against

VARIANTS = {
    "v1_original": {
        "instructions": "Does this comment insult a person or group?",
        "criteria": {
            "true": "Demeans, mocks or attacks a person or group directly.",
            "false": "Attacks or criticises ideas, actions, policies or "
                     "institutions without demeaning the people involved.",
        },
    },
    # Jigsaw's raters were asked about comments that are insulting,
    # inflammatory or negative toward a person or group. Matching their
    # language removes "you asked a different question than they answered"
    # as an explanation.
    "v2_annotator": {
        "instructions": "Is this an insulting, inflammatory, or negative "
                        "comment toward a person or group of people?",
        "criteria": {
            "true": "Insulting, inflammatory or negative toward a person or "
                    "group of people.",
            "false": "Not directed at a person or group in an insulting, "
                     "inflammatory or negative way.",
        },
    },
    # If over-flagging is a wording artefact, this one should cure it.
    "v3_strict": {
        "instructions": "Does this comment contain a direct personal insult?",
        "criteria": {
            "true": "Contains an explicit insult aimed at a specific person or "
                    "group: name-calling, slurs, or demeaning characterisations "
                    "of them as people.",
            "false": "Everything else. Harsh criticism, sarcasm, contempt for "
                     "ideas or institutions, generalised anger, and negative "
                     "statements with no named target are all false.",
        },
    },
    # No criteria at all - isolates how much the criteria are contributing.
    "v4_bare": {
        "instructions": "Does this comment insult a person or group?",
    },
}


def collect(name):
    """One call per comment, four wordings inside it. Resumable like step 2."""
    out_path = f"results/{name}_wording.jsonl"
    with open(f"data/{name}.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    done = set()
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["id"])
                except (json.JSONDecodeError, KeyError):
                    continue

    todo = [r for r in rows if r["id"] not in done]
    print(f"{len(rows)} comments, {len(done)} done, {len(todo)} to go")
    if not todo:
        return out_path

    lock = threading.Lock()
    fh = open(out_path, "a", encoding="utf-8")
    counts = {"ok": 0, "fail": 0, "tokens": 0, "abort": False}

    def handle(row):
        if counts["abort"]:
            return
        try:
            probs, usage = jev.ask(row["text"], VARIANTS)
        except Exception as e:
            with lock:
                counts["fail"] += 1
                if counts["fail"] <= 3:
                    print(f"  FAILED id={row['id']}: {str(e)[:300]}")
                if counts["ok"] == 0 and counts["fail"] >= 5:
                    counts["abort"] = True
            return

        record = {
            "id": row["id"],
            "input_tokens": usage.get("input_tokens"),
            "model": usage.get("model"),
            "pred": probs,
            "human": float(row[TRUTH_COLUMN]),
        }
        with lock:
            fh.write(json.dumps(record) + "\n")
            fh.flush()
            counts["ok"] += 1
            counts["tokens"] += usage.get("input_tokens") or 0
            if counts["ok"] % 50 == 0:
                print(f"  {counts['ok']}/{len(todo)}")

    started = time.perf_counter()
    try:
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            list(pool.map(handle, todo))
    finally:
        fh.close()

    print(f"\n{counts['ok']} ok, {counts['fail']} failed, "
          f"{time.perf_counter() - started:.1f}s")
    print(f"{counts['tokens']:,} tokens = "
          f"${counts['tokens'] / 1e6 * PRICE_PER_MTOK:.5f}")
    if counts["abort"]:
        sys.exit("Aborted - run probe.py to check the request shape.")
    return out_path


def load(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            for variant, p in r["pred"].items():
                rows.append({
                    "question": variant,
                    "p": float(p),
                    "human_fraction": r["human"],
                    "outcome": int(r["human"] >= 0.5),
                })
    return pd.DataFrame(rows)


def plot(df, name):
    variants = list(VARIANTS)
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 10.0), dpi=200)

    for ax, variant in zip(axes.flat, variants):
        part = df[df["question"] == variant]
        chart.reliability(ax, measure.buckets(part), label_diag=False)
        s = measure.summarise(part)
        ax.set_title(f"{variant}   ECE {s['ECE']:.3f}   AUC {s['AUC']:.3f}",
                     fontsize=10.5, color=chart.INK, pad=9, loc="left")
        ax.tick_params(labelsize=8)

    fig.supxlabel("Stated confidence", fontsize=10, color=chart.INK_2)
    fig.supylabel("How often humans actually flagged it", fontsize=10,
                  color=chart.INK_2)
    fig.suptitle("Four wordings of the same question, one model",
                 fontsize=14, color=chart.INK, x=0.055, ha="left", y=0.985)
    fig.tight_layout(rect=[0.03, 0.02, 1, 0.955])

    out = f"results/{name}_wording.png"
    fig.savefig(out, bbox_inches="tight", facecolor=chart.SURFACE, pad_inches=0.42)
    plt.close(fig)
    return out


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: python3 7_wording.py <natural|hard>")
    name = sys.argv[1]

    df = load(collect(name))
    base = df.groupby("question")["outcome"].mean().iloc[0]

    rows = []
    for variant in VARIANTS:
        part = df[df["question"] == variant]
        stats = measure.summarise(part)
        _, beta = recal.fit_platt(part["p"].to_numpy(), part["outcome"].to_numpy())
        rows.append({
            "variant": variant,
            "mean_p": part["p"].mean(),
            "ECE": stats["ECE"],
            "AUC": stats["AUC"],
            "acc@0.5": stats["accuracy_at_0.5"],
            "platt_slope": beta[0],
            "platt_intercept": beta[1],
        })

    out = pd.DataFrame(rows).set_index("variant")
    pd.set_option("display.width", 200)
    print(f"\n=== '{TRUTH_COLUMN}' on {name}.csv " + "=" * 34)
    print(f"Human base rate: {base:.1%}. "
          f"A perfectly calibrated model would average {base:.3f}.\n")
    print(out.round(4).to_string())

    spread = out["platt_intercept"].max() - out["platt_intercept"].min()
    worst = out["platt_intercept"].min()
    print("\n=== The verdict " + "=" * 47)
    print(f"  Intercept range across wordings: {spread:.2f}")
    print(f"  Most negative intercept:         {worst:.2f}")

    if out["platt_intercept"].max() < -0.5:
        print(f"\n  Every wording still needs a large negative correction, "
              f"including\n  v3_strict, which was written specifically to stop "
              f"over-flagging. Wording\n  moves the bias by {spread:.2f} but does "
              f"not remove it. The lean toward\n  'yes' survives rephrasing, so "
              f"it is a property of the model on this\n  task, not an artefact "
              f"of how the question was asked.")
    else:
        print(f"\n  At least one wording lands near zero. The bias is "
              f"substantially a\n  prompting artefact - report it that way, and "
              f"use the best wording as\n  the headline result rather than v1.")

    out.to_csv(f"results/{name}_wording.csv")
    print("\nwrote", plot(df, name))
    print(f"wrote results/{name}_wording.csv")


if __name__ == "__main__":
    main()
