"""
The only file that knows what a Jev API call looks like.

Everything else in this project just calls `ask(text, questions)` and gets back
a dict of {question_name: probability}. If TypeSafe's request or response format
is different from what I guessed below, you change it HERE and nowhere else.

Read this file top to bottom before running anything. It is ~120 lines and it is
the only part of the project with an external dependency.
"""

import json
import os
import random
import time
import urllib.error
import urllib.request

# ---------------------------------------------------------------------------
# Config. Set JEV_API_KEY in your environment, or drop it in a .env file next
# to this script (see .env.example). .env is gitignored.
# ---------------------------------------------------------------------------

def _load_dotenv():
    """Read KEY=value lines from .env into the environment.

    Ten lines instead of a dependency. Anything already exported wins, so you
    can still override a .env value for a single run. Resolved relative to this
    file, so it works no matter which directory you run from.
    """
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip("\"'"))
    except FileNotFoundError:
        pass


_load_dotenv()

BASE_URL = os.environ.get("JEV_BASE_URL", "https://api.typesafe.ai")
ENDPOINT = "/v1/systemone"
MODEL = os.environ.get("JEV_MODEL", "jev-latest")


class JevError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# The wire format, verified against https://docs.typesafe.ai/api.md
#
#   REQUEST                             RESPONSE
#   {                                   {
#     "state": "...",                     "model": "jev-latest",
#     "model": "jev-latest",              "answers": {
#     "questions": {                        "is_urgent": {
#       "is_urgent": {                        "type": "noul",
#         "type": "noul",                     "noul": 0.92
#         "instructions": "...",            }
#         "criteria": {                   },
#           "true":  "...",               "usage": {"input_tokens": 312, ...}
#           "false": "..."              }
#         }
#       }
#     }
#   }
#
# `instructions` carries the judgment; `criteria` defines what each answer
# means. `questions` is a dict keyed by name - a list is rejected with
# 422 loc=["body","questions"] "Input should be a valid dictionary".
# ---------------------------------------------------------------------------

def build_request(text, questions):
    """Turn a comment + a dict of question specs into the POST body.

    `state` is the shared context - here, the comment being judged. Every
    question is evaluated against that same state, in one call. That is the
    whole point of Jev: N judgments, one request.

    Each spec is {"instructions": str, "criteria": {"true": str, "false": str}}.
    """
    return {
        "model": MODEL,
        "state": text,
        "questions": {
            name: {"type": "noul", **spec} for name, spec in questions.items()
        },
    }


def parse_response(payload, questions):
    """Pull {question_name: probability} out of the response.

    For a Noul the probability lives in the `noul` field, 0 to 1. Note there is
    no separate confidence number: for a yes/no question the probability IS the
    confidence, which is exactly what makes it worth checking.
    """
    answers = payload.get("answers")
    if not isinstance(answers, dict):
        raise JevError(
            "Response has no `answers` object:\n"
            + json.dumps(payload, indent=2)[:1500]
        )

    out = {}
    for name in questions:
        item = answers.get(name)
        if isinstance(item, dict) and isinstance(item.get("noul"), (int, float)):
            out[name] = float(item["noul"])

    if not out:
        raise JevError(
            "No noul probabilities found for any question. Raw response:\n"
            + json.dumps(payload, indent=2)[:1500]
        )
    return out


# ---------------------------------------------------------------------------
# Transport. Plain urllib so there is nothing to install and nothing hidden.
# ---------------------------------------------------------------------------

def ask(text, questions, retries=4, timeout=30):
    """Ask Jev every question in `questions` about `text`. One HTTP call.

    Returns ({name: probability}, usage) - usage carries the real token counts,
    so cost comes from what was actually billed rather than a chars/4 guess.
    """
    key = os.environ.get("JEV_API_KEY")
    if not key:
        raise JevError("JEV_API_KEY is not set. See .env.example.")

    body = json.dumps(build_request(text, questions)).encode()
    req = urllib.request.Request(
        BASE_URL.rstrip("/") + ENDPOINT,
        data=body,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    last = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = json.load(resp)
                # Record the version the alias resolved to. Asking for
                # "jev-latest" and getting "jev-1.13.0" back is the difference
                # between a reproducible result and one that drifts silently.
                meta = dict(payload.get("usage") or {})
                meta["model"] = payload.get("model")
                return parse_response(payload, questions), meta
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:400]
            last = JevError(f"HTTP {e.code}: {detail}")
            # 4xx that is not rate limiting will never succeed on retry.
            if e.code not in (408, 409, 429) and e.code < 500:
                raise last
        except Exception as e:  # network blips, timeouts
            last = JevError(f"{type(e).__name__}: {e}")
        time.sleep(2 ** attempt)  # 1s, 2s, 4s, 8s

    raise last


# ---------------------------------------------------------------------------
# Simulation mode, so the pipeline runs end to end before your key works.
#
# This does NOT call Jev. It invents a model that is decent but overconfident,
# which is the most common real failure. Use it to confirm steps 3-5 work, then
# throw the output away and run for real. Every file it touches is tagged
# SIMULATED so you cannot mix them up.
# ---------------------------------------------------------------------------

def ask_simulated(text, questions, truth, seed=0):
    """Fake a response. `truth` is the real human agreement per question.

    Same (probabilities, usage) shape as ask(), so callers do not branch.
    """
    rng = random.Random(f"{seed}:{text}")
    out = {}
    for name in questions:
        signal = truth.get(name, 0.0)
        p = min(0.98, max(0.02, signal + rng.gauss(0, 0.18)))
        # Push probabilities toward 0 and 1 - this is what overconfidence is.
        g = 1.9
        out[name] = round(p**g / (p**g + (1 - p) ** g), 4)
    return out, {"input_tokens": len(text) // 4}
