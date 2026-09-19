"""
STEP 1 - Get comments that real humans have already judged.

We use `civil_comments`, a public dataset of news-site comments. Every comment
was shown to a panel of human annotators, and the dataset records what fraction
of them flagged it. So `toxicity = 0.7` means 7 in 10 humans called it toxic.

That fraction is what makes this project work. We are not comparing Jev against
one person's opinion, we are comparing it against a crowd.

We build two samples, and the difference between them is the interesting part:

  natural.csv - a plain random sample. Looks like real traffic: mostly clean,
                a small amount of clearly awful stuff, very little in between.

  hard.csv    - deliberately spread across the whole toxicity range, so roughly
                a third of it is genuinely ambiguous (humans split 30-70).

Models usually look well calibrated on natural.csv and fall apart on hard.csv,
because natural.csv is mostly easy calls. Running both is how you find that out.

    python3 1_get_data.py
"""

import csv
import json
import random
import urllib.parse
import urllib.request

DATASET = "google/civil_comments"
SPLIT = "test"
ROWS_URL = "https://datasets-server.huggingface.co/rows"

POOL_PAGES = 45        # 45 pages x 100 rows = 4,500 comments to draw from
PER_SAMPLE = 400       # comments in each of the two output files
MAX_CHARS = 1200       # trim very long comments to keep calls small and cheap
SEED = 7

# The four judgments we ask about. The dataset has a human-agreement fraction
# for each one, so each comment gives us four things to check, not one.
LABEL_COLUMNS = ["toxicity", "insult", "threat", "obscene"]


def fetch_page(offset, length=100):
    query = urllib.parse.urlencode(
        {
            "dataset": DATASET,
            "config": "default",
            "split": SPLIT,
            "offset": offset,
            "length": length,
        }
    )
    with urllib.request.urlopen(f"{ROWS_URL}?{query}", timeout=30) as resp:
        return [r["row"] for r in json.load(resp)["rows"]]


def build_pool():
    """Pull comments from random offsets so we are not just reading the front."""
    rng = random.Random(SEED)
    offsets = rng.sample(range(0, 97_000, 100), POOL_PAGES)
    pool, seen = [], set()

    for i, off in enumerate(offsets, 1):
        print(f"  fetching page {i}/{POOL_PAGES} (offset {off})", end="\r", flush=True)
        for row in fetch_page(off):
            text = (row.get("text") or "").strip()
            # Skip comments too short to judge or duplicates.
            if len(text) < 25 or text in seen:
                continue
            seen.add(text)
            pool.append(
                {
                    "text": text[:MAX_CHARS],
                    **{c: float(row.get(c) or 0.0) for c in LABEL_COLUMNS},
                }
            )
    print(f"\n  pool: {len(pool)} unique comments")
    return pool


def stratified(pool, n, rng):
    """Even coverage across the toxicity range, so the ambiguous middle is
    actually represented instead of being drowned out by clean comments."""
    bands = [(0.0, 0.05), (0.05, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.01)]
    per_band = n // len(bands)
    out = []

    for lo, hi in bands:
        members = [r for r in pool if lo <= r["toxicity"] < hi]
        rng.shuffle(members)
        take = members[:per_band]
        out.extend(take)
        print(f"  toxicity {lo:.2f}-{hi:.2f}: wanted {per_band}, got {len(take)}")

    rng.shuffle(out)
    return out


def write(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "text"] + LABEL_COLUMNS)
        w.writeheader()
        for i, r in enumerate(rows):
            w.writerow({"id": i, **r})
    print(f"  wrote {len(rows)} rows -> {path}")


def main():
    rng = random.Random(SEED)

    print("Downloading comments...")
    pool = build_pool()

    print("\nnatural.csv (random sample, real-world mix):")
    natural = rng.sample(pool, min(PER_SAMPLE, len(pool)))
    ambiguous = sum(1 for r in natural if 0.3 <= r["toxicity"] <= 0.7)
    flagged = sum(1 for r in natural if r["toxicity"] >= 0.5)
    print(f"  {flagged}/{len(natural)} comments most humans called toxic")
    print(f"  {ambiguous}/{len(natural)} where humans genuinely disagreed")
    write("data/natural.csv", natural)

    print("\nhard.csv (spread across the toxicity range):")
    hard = stratified(pool, PER_SAMPLE, rng)
    write("data/hard.csv", hard)

    print("\nDone. Open data/hard.csv and read a few rows before moving on -")
    print("you should be able to see why some of them are genuinely hard calls.")


if __name__ == "__main__":
    main()
