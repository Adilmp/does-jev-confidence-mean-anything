"""
STEP 2 - Ask Jev to judge every comment, and write down what it said.

One call per comment, four questions per call. Jev evaluates all four against
the same shared state, which is the thing it is actually built to do.

The wording of the questions below matters more than it looks. The humans who
labelled this dataset were given a specific definition of "toxic" - rude,
disrespectful, or unreasonable enough that you would leave the conversation.
If we asked Jev a vaguer question, any disagreement we measure would just be
the two sides using different definitions, not the model being wrong. Matching
the annotator instructions is what makes the comparison fair.

    python3 2_ask_jev.py hard              # real API call, needs JEV_API_KEY
    python3 2_ask_jev.py hard --simulate   # fake data, no key needed

Results append to results/<name>_predictions.jsonl as they arrive. Stop the
script whenever you like and re-run it - it skips comments it already has.
"""

import csv
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import jev

QUESTIONS = {
    "toxicity": {
        "instructions": "Is this comment rude, disrespectful, or unreasonable?",
        "criteria": {
            "true": "Rude, disrespectful or unreasonable enough that a reader "
                    "would be likely to leave the discussion.",
            "false": "Civil enough that a reader would stay in the discussion. "
                     "Blunt, critical, sarcastic or strongly worded opinions "
                     "count as false on their own.",
        },
    },
    "insult": {
        "instructions": "Does this comment insult a person or group?",
        "criteria": {
            "true": "Demeans, mocks or attacks a person or group directly.",
            "false": "Attacks or criticises ideas, actions, policies or "
                     "institutions without demeaning the people involved.",
        },
    },
    "threat": {
        "instructions": "Does this comment threaten violence or harm?",
        "criteria": {
            "true": "Expresses intent or desire to inflict physical harm, or "
                    "wishes harm on someone.",
            "false": "Angry, hostile or aggressive in tone but threatens no "
                     "harm. Describing violence that already happened is false.",
        },
    },
    "obscene": {
        "instructions": "Does this comment contain obscene or vulgar language?",
        "criteria": {
            "true": "Contains profanity, vulgar terms or sexually explicit "
                    "language.",
            "false": "No profanity or vulgar terms, however hostile the "
                     "content is.",
        },
    },
}

WORKERS = int(os.environ.get("JEV_WORKERS", "8"))
ABORT_AFTER = 5                 # give up if this many fail with zero successes

# Step 7 tested four wordings of the `insult` question and found the strictest
# one roughly halved its calibration error. `--strict` applies that lesson to
# all four questions: a high bar in `true`, and exclusions spelled out in
# `false`. Results land in a separate file so the original run is still there
# to compare against - you cannot claim wording does not rescue the model
# unless you kept the evidence of both attempts.
STRICT_CRITERIA = {
    "toxicity": {
        "true": "So rude or hostile that a reasonable reader would abandon the "
                "thread rather than reply.",
        "false": "Everything else. Blunt disagreement, sarcasm, cynicism, "
                 "contempt for institutions, political anger and strong "
                 "language are all false unless aimed at degrading people.",
    },
    "insult": {
        "true": "Contains an explicit insult aimed at a specific person or "
                "group: name-calling, slurs, or demeaning characterisations "
                "of them as people.",
        "false": "Everything else. Harsh criticism, sarcasm, contempt for "
                 "ideas or institutions, generalised anger, and negative "
                 "statements with no named target are all false.",
    },
    "threat": {
        "true": "States an intent or wish to physically harm a specific "
                "person or group.",
        "false": "Everything else. Hostility, insults, predictions of "
                 "consequences, legal or electoral threats, and descriptions "
                 "of violence that already happened are all false.",
    },
    "obscene": {
        "true": "Contains profanity, vulgar terms, or sexually explicit "
                "language in the text itself.",
        "false": "Everything else, no matter how hostile. Discussing "
                 "obscenity without using it is false.",
    },
}
PRICE_PER_MTOK = 0.042          # TypeSafe's published input price
CHARS_PER_TOKEN = 4             # rough industry rule of thumb, fine for an estimate


def load_rows(name):
    with open(f"data/{name}.csv", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def already_done(path):
    """Read back whatever a previous run finished, so we never pay twice."""
    if not os.path.exists(path):
        return set()
    done = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                done.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                continue  # half-written final line from a hard stop
    return done


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: python3 2_ask_jev.py <natural|hard> [--simulate]")

    name = sys.argv[1]
    simulate = "--simulate" in sys.argv

    # Check this before opening anything. Otherwise a missing key produces 400
    # identical failures and leaves an empty predictions file that looks real.
    if not simulate and not os.environ.get("JEV_API_KEY"):
        sys.exit(
            "JEV_API_KEY is not set.\n"
            "  export JEV_API_KEY=sk-...        then re-run, or\n"
            "  python3 2_ask_jev.py %s --simulate   to test the pipeline "
            "without a key." % name
        )

    rows = load_rows(name)

    if "--strict" in sys.argv:
        # Same instructions, tightened criteria. Only the wording changes, so
        # any difference in the outcome is attributable to the wording. The
        # data file is unchanged; only the results filename gains a suffix.
        for question, criteria in STRICT_CRITERIA.items():
            QUESTIONS[question]["criteria"] = criteria
        name = f"{name}_strict"
        print("Tightened criteria (step 7's v3_strict style).\n")

    out_path = f"results/{name}_predictions.jsonl"
    if simulate:
        out_path = f"results/{name}_predictions.SIMULATED.jsonl"

    done = already_done(out_path)
    todo = [r for r in rows if r["id"] not in done]

    if simulate:
        print("!! SIMULATE MODE - inventing a plausible overconfident model.")
        print("!! These numbers are NOT from Jev. Do not put them in a post.\n")

    print(f"{len(rows)} comments, {len(done)} already done, {len(todo)} to go")
    if not todo:
        print("Nothing to do.")
        return

    est_tokens = sum(len(r["text"]) for r in todo) / CHARS_PER_TOKEN
    print(f"Rough cost estimate: ${est_tokens / 1e6 * PRICE_PER_MTOK:.4f}\n")

    lock = threading.Lock()
    out = open(out_path, "a", encoding="utf-8")
    counts = {"ok": 0, "fail": 0, "tokens": 0, "models": set(), "abort": False}

    def handle(row):
        # If nothing has succeeded and failures are piling up, the request shape
        # is wrong and every remaining comment will fail the same way. Stop
        # rather than printing the same error 400 times.
        if counts["abort"]:
            return

        started = time.perf_counter()
        try:
            if simulate:
                truth = {q: float(row[q]) for q in QUESTIONS}
                probs, usage = jev.ask_simulated(row["text"], QUESTIONS, truth)
            else:
                probs, usage = jev.ask(row["text"], QUESTIONS)
        except Exception as e:
            with lock:
                counts["fail"] += 1
                if counts["fail"] <= 3:  # first few in full - that is the diagnosis
                    print(f"  FAILED id={row['id']}: {str(e)[:400]}")
                if counts["ok"] == 0 and counts["fail"] >= ABORT_AFTER:
                    counts["abort"] = True
            return

        record = {
            "id": row["id"],
            "latency_s": round(time.perf_counter() - started, 3),
            "chars": len(row["text"]),
            "input_tokens": usage.get("input_tokens"),
            "model": usage.get("model"),
            # what Jev said
            "pred": probs,
            # what the humans said - the fraction who flagged it
            "human": {q: float(row[q]) for q in QUESTIONS},
        }
        with lock:
            out.write(json.dumps(record) + "\n")
            out.flush()  # so a crash never loses work you already paid for
            counts["ok"] += 1
            counts["tokens"] += usage.get("input_tokens") or 0
            counts["models"].add(usage.get("model"))
            n = counts["ok"] + counts["fail"]
            if n % 25 == 0:
                print(f"  {n}/{len(todo)}")

    started = time.perf_counter()
    try:
        with ThreadPoolExecutor(max_workers=1 if simulate else WORKERS) as pool:
            list(pool.map(handle, todo))
    finally:
        out.close()

    elapsed = time.perf_counter() - started
    print(f"\n{counts['ok']} ok, {counts['fail']} failed, {elapsed:.1f}s wall clock")
    if counts["tokens"]:
        # Billed tokens, not a chars/4 estimate - the response reports them.
        spent = counts["tokens"] / 1e6 * PRICE_PER_MTOK
        per_call = counts["tokens"] / max(counts["ok"], 1)
        print(f"{counts['tokens']:,} input tokens billed = ${spent:.5f} "
              f"({per_call:.0f} per comment)")

    served = sorted(m for m in counts["models"] if m)
    if served:
        print(f"served by: {', '.join(served)}")
        if len(served) > 1:
            print("  NOTE: more than one version served this run. Pin JEV_MODEL "
                  "to one of them and re-run before reporting anything.")
    print(f"-> {out_path}")

    if counts["abort"]:
        print(f"\nStopped after {counts['fail']} failures with nothing succeeding.")
        print("The request shape is wrong. Find the right one with:")
        print("    python3 probe.py")
        print("It sends a single request, tries several shapes, and prints the")
        print("raw response so you can fix build_request/parse_response in jev.py.")
        sys.exit(1)
    if counts["fail"]:
        print("\nRe-run when fixed - finished comments are skipped automatically.")


if __name__ == "__main__":
    main()
