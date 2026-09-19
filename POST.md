# LinkedIn post

Images, in order: `results/natural_strict_calibration.png` (the problem, using
the *best* wording I found), then `results/natural_strict_recalibrated.png`
(the fix). LinkedIn shows the first in the feed, so the reliability curve
carries the hook.

Swap `[repo link]` for your GitHub URL before posting.

---

TypeSafe's Jev told me it was 75% confident. On those exact cases, human annotators agreed 10% of the time.

Jev is a "decision model" — it doesn't generate text, it returns a probability. So your code ends up doing `if p > 0.9: auto_approve()`. I wanted to know whether that number is honest.

I ran 8,000 judgments against civil_comments, a public dataset where every comment carries the *fraction of human annotators* who flagged it. Real ground truth, not another model's opinion.

**The ranking is genuinely good.** AUC 0.91. Jev reliably sorts worse content above better. It understands the task.

**The probabilities attached to that ranking are not.** Every confidence band sat below the diagonal — a systematic lean toward "yes," not noise.

**My first thought was that I'd written bad prompts.** So I tested four wordings of the same question, all in one call, so the model and the comment were identical and only the phrasing moved. Results:

• Wording matters a lot — calibration error ranged from 0.21 to 0.52 across the four.
• The wording that mirrors the human raters' own published definition was the **worst** of the four.
• AUC barely moved (0.88–0.91). Wording shifts the scale, not the signal.
• No wording fixed it. The strictest version — explicit exclusions, high bar — still predicted 26% where reality was 5%.

So I re-ran everything with the best wording I had. It helped: error fell from 0.209 to 0.156, precision from 0.12 to 0.17. The bias survived.

**The error scales with how rare your positives are.** At a 19% base rate, error was 0.157. On realistic traffic at 2.9%, it grew. Jev's probabilities don't adapt to your base rate — and rare-event detection is the entire use case.

**It's fixable, cheaply.** A logistic fit — two parameters, ~20 lines — removed 96% of the calibration error on data it never saw. AUC went 0.918 → 0.918. The ranking was always there. Only the units were wrong.

Caveats, because they're load-bearing:
• One dataset, one domain, one pinned version (jev-1.13.0).
• I nearly reported "95.4% accuracy" as a win — until I noticed that flagging nothing scores 97.1% at this base rate. Accuracy is a trap on imbalanced data; the honest read is precision 0.17 at recall 0.74.
• Four wordings is four, not all of them. Someone may well find a better one.

None of this is an argument against the model. TypeSafe's own docs say to validate calibration on your own data, and this is what that looks like when you actually do it. Used as a high-recall pre-filter with a calibration layer on top, it's genuinely useful. Used as `if p > 0.9`, it isn't what you think.

Total cost of finding out: five cents.

Code and charts: [repo link]

---

## Notes

**What changed after the wording test.** The first draft carried "my prompt
might be the problem" as a caveat. Now it's a finding with numbers behind it,
and the headline uses the best wording rather than the first — which removes
the single most obvious attack on the post.

**The strongest single detail** is that the annotators' own wording scored
worst. It kills "you asked a different question than the humans answered"
outright, and it's genuinely counterintuitive.

**Why the caveats stay in.** They're the credibility. Listing the accuracy trap
you nearly fell into signals more competence than never mentioning it. Cut
anything else first.

**Why it's fair rather than a takedown.** AUC 0.91 appears before any
criticism, the fix is the payoff, and the conclusion names a real use case
where the model works. If TypeSafe's team reads it, the worst case is a
disagreement about prompt wording — which you pre-empted by testing four.

**If you want it shorter:** cut "the error scales with rarity" and the fourth
wording bullet. Keep ranking-good / numbers-wrong / wording-tested / fixable,
and keep all three caveats.

**Expected pushback, and your answers:**
- *"You binarised human agreement at 0.5."* The gap holds against the raw
  annotator fraction too — negative in all 10 bands, widening at high
  confidence. Binarising isn't doing the work.
- *"Your prompts were bad."* Tested. Four wordings, best one used for the
  headline, bias survived all four.
- *"n is small."* 8,000 judgments, two base rates, three wording conditions.
- *"Base rate shift isn't miscalibration."* Correct, and that's the point — a
  model sold on calibrated decisions should track the base rate of the data
  you give it, or say that it doesn't.

## Numbers, for reference

| | natural (v1 wording) | natural (best wording) |
|---|---|---|
| ECE | 0.209 | 0.156 |
| AUC | 0.906 | 0.912 |
| precision @ 0.5 | 0.124 | 0.165 |
| recall @ 0.5 | 0.848 | 0.739 |
| Platt intercept | −3.34 | −2.91 |
| after recalibration (ECE) | 0.007 | 0.006 |

Wording test on `insult`, natural traffic (true base rate 4.8%):

| wording | mean p | ECE | AUC |
|---|---|---|---|
| v1_original | 0.445 | 0.397 | 0.896 |
| v2_annotator | 0.566 | 0.519 | 0.881 |
| v3_strict | 0.259 | 0.211 | 0.913 |
| v4_bare | 0.488 | 0.441 | 0.888 |

8,000 judgments, 1,215,623 input tokens, $0.0511 total.
