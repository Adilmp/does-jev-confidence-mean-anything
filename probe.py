"""
Send ONE request and show exactly what comes back.

Use this whenever something looks wrong - never debug by re-running 400
comments. It prints the request it sent and the raw response, then checks that
jev.parse_response() can read it.

    python3 probe.py
"""

import json
import os
import urllib.error
import urllib.request

import jev

TEXT = "You are completely clueless and everyone here knows it."

QUESTION = {
    "insult": {
        "instructions": "Does this comment insult a person or group?",
        "criteria": {
            "true": "Demeans, mocks or attacks a person or group directly.",
            "false": "Attacks or criticises ideas, actions, policies or "
                     "institutions without demeaning the people involved.",
        },
    }
}


def main():
    if not os.environ.get("JEV_API_KEY"):
        raise SystemExit("JEV_API_KEY is not set. Put it in .env or export it.")

    body = jev.build_request(TEXT, QUESTION)
    print("=== Request ===")
    print(json.dumps(body, indent=2))

    req = urllib.request.Request(
        jev.BASE_URL.rstrip("/") + jev.ENDPOINT,
        data=json.dumps(body).encode(),
        headers={
            "Authorization": f"Bearer {os.environ['JEV_API_KEY']}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            status, payload = r.status, json.load(r)
    except urllib.error.HTTPError as e:
        raw = e.read().decode(errors="replace")
        print(f"\n=== HTTP {e.code} ===")
        print(raw[:2000])
        print("\nThe error names the field it disliked - fix build_request() "
              "in jev.py to match.")
        return

    print(f"\n=== Raw response (HTTP {status}) ===")
    print(json.dumps(payload, indent=2)[:3000])

    print("\n=== What parse_response() extracts ===")
    try:
        print(jev.parse_response(payload, QUESTION))
        print(f"\nThe comment was an insult, so expect a probability near 1.")
        print("If that looks right: ./run_all.sh hard")
    except jev.JevError as e:
        print(f"FAILED: {e}")


if __name__ == "__main__":
    main()
