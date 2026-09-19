"""
STEP 8 - Build the LinkedIn carousel.

LinkedIn suppresses posts carrying external links, and its native document
format is the opposite: people swipe through it without leaving the feed. So
the deck is the post, and the repo link goes in the first comment.

Portrait 4:5 (1080x1350) because that is the most screen a post can occupy on a
phone, which is where nearly all of this gets read.

Everything here uses the palette from 4_chart.py, so the slides and the figures
they contain look like one artefact rather than a deck about a project.

    python3 8_slides.py        -> results/carousel.pdf + results/slides/*.png
"""

import os
import textwrap
from importlib import import_module

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

chart = import_module("4_chart")

W, H, DPI = 8.0, 10.0, 135          # 1080 x 1350
SURFACE, INK, INK_2 = chart.SURFACE, chart.INK, chart.INK_2
MUTED, SERIES, OVER = chart.MUTED, chart.SERIES, chart.OVER

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Segoe UI", "Helvetica", "Arial"],
})


def blank():
    fig = plt.figure(figsize=(W, H), dpi=DPI, facecolor=SURFACE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    return fig, ax


def footer(ax, n):
    ax.text(0.08, 0.045, "Adil Muhammad Pervez", fontsize=10, color=MUTED)
    ax.text(0.92, 0.045, str(n), fontsize=10, color=MUTED, ha="right")


def wrap(ax, text, x, y, width, size, colour, weight="normal", spacing=1.38,
         family=None):
    """matplotlib does not wrap text, so wrap it before drawing.

    Returns the height consumed in axes units so callers can stack blocks
    without hand-placing every y, which is how the rule ended up struck
    through the headline the first time.
    """
    lines = []
    for para in text.split("\n"):
        lines.extend(textwrap.wrap(para, width) or [""])
    kw = {"fontfamily": family} if family else {}
    ax.text(x, y, "\n".join(lines), fontsize=size, color=colour, va="top",
            linespacing=spacing, fontweight=weight, **kw)
    return len(lines) * size * spacing / (H * 72)


def title_slide(n):
    fig, ax = blank()
    y = 0.86
    y -= wrap(ax, "TypeSafe's Jev told me\nit was 75% confident.",
              0.08, y, 30, 31, INK, spacing=1.3) + 0.055
    y -= wrap(ax, "Human annotators agreed\n10% of the time.",
              0.08, y, 30, 31, OVER, spacing=1.3) + 0.05

    ax.plot([0.08, 0.26], [y, y], color=MUTED, linewidth=1.5)
    y -= 0.035
    y -= wrap(ax, "I audited whether a decision model's confidence scores mean "
                  "what they claim. 8,000 judgments, measured against human "
                  "labels.", 0.08, y, 40, 17, INK_2) + 0.075

    # Three numbers instead of dead space. They also pre-empt the "is this just
    # a rant" read before anyone swipes.
    for i, (val, lab) in enumerate([("0.91", "AUC\nranking"),
                                    ("0.156", "ECE\ncalibration"),
                                    ("$0.05", "total\ncost")]):
        x = 0.08 + i * 0.29
        ax.text(x, y, val, fontsize=26, color=INK, va="top")
        ax.text(x, y - 0.045, lab, fontsize=12, color=MUTED, va="top",
                linespacing=1.3)

    ax.text(0.08, 0.135, "Swipe →", fontsize=15, color=SERIES)
    footer(ax, n)
    return fig


def text_slide(n, kicker, heading, body=None, bullets=None, accent=None,
               mono=False):
    fig, ax = blank()
    ax.text(0.08, 0.93, kicker.upper(), fontsize=12, color=SERIES, va="top")
    y = 0.875
    y -= wrap(ax, heading, 0.08, y, 28, 27, INK, spacing=1.3) + 0.05

    if body:
        y -= wrap(ax, body, 0.08, y, 40, 17, INK_2) + 0.045

    for b in bullets or []:
        ax.plot([0.088], [y - 0.014], marker="o", markersize=5, color=SERIES)
        y -= wrap(ax, b, 0.13, y, 34 if mono else 36, 16 if mono else 17,
                  INK_2, family="monospace" if mono else None) + 0.03

    if accent:
        # Flow the box under the content rather than pinning it to a fixed y:
        # short slides were leaving a hole between the two.
        lines = sum(len(textwrap.wrap(p, 44)) or 1 for p in accent.split("\n"))
        box_h = 0.045 + lines * 0.031
        top = max(0.135 + box_h, min(y - 0.03, 0.40 + box_h))

        # If the content ran so long that the box has to climb back over it,
        # say so loudly. A silent overlap is how slide 9 shipped with the
        # accent box printed across the last bullet.
        if top > y - 0.015:
            print(f"  !! slide {n}: content overruns the accent box by "
                  f"{(top - y + 0.015) * H * 72:.0f}pt - shorten it")
        ax.add_patch(plt.Rectangle((0.08, top - box_h), 0.84, box_h,
                                   facecolor="#f2f1ed", edgecolor="none",
                                   transform=ax.transAxes))
        wrap(ax, accent, 0.108, top - 0.028, 44, 15.5, INK)
    footer(ax, n)
    return fig


def stat_slide(n, kicker, figure, label, note):
    fig, ax = blank()
    ax.text(0.08, 0.93, kicker.upper(), fontsize=12, color=SERIES, va="top")
    ax.text(0.08, 0.68, figure, fontsize=104, color=INK, va="center")
    y = 0.52
    y -= wrap(ax, label, 0.08, y, 34, 21, INK) + 0.055
    wrap(ax, note, 0.08, y, 40, 17, INK_2)
    footer(ax, n)
    return fig


def image_slide(n, kicker, path, caption):
    fig, ax = blank()
    ax.text(0.08, 0.955, kicker.upper(), fontsize=12, color=SERIES, va="top")
    img = mpimg.imread(path)
    # Fit inside the frame, preserving aspect; the figures are portrait-ish so
    # they sit naturally in a 4:5 slide.
    box = fig.add_axes([0.05, 0.20, 0.90, 0.715])
    box.imshow(img)
    box.set_axis_off()

    used = wrap(ax, caption, 0.08, 0.175, 58, 13.5, INK_2)
    if 0.175 - used < 0.075:  # footer sits at 0.045
        print(f"  !! slide {n}: caption runs into the footer - shorten it")
    footer(ax, n)
    return fig


def build():
    R = "results"
    slides = [
        title_slide(1),

        text_slide(
            2, "why it matters", "Jev doesn't write text.\nIt returns a number.",
            body="It's a \"decision model\": you hand it state, it hands back a "
                 "probability. Which means your code ends up doing this:",
            bullets=["if p > 0.9:  auto_approve()",
                     "elif p < 0.1: auto_reject()",
                     "else:         send_to_human()"],
            mono=True,
            accent="Every threshold you write is a bet that the number is "
                   "honest. So I checked."),

        text_slide(
            3, "the method", "Ground truth from\nhumans, not models.",
            body="civil_comments shows each comment to a panel of annotators "
                 "and records what fraction flagged it. A toxicity of 0.7 means "
                 "7 in 10 real people agreed.",
            bullets=["8,000 judgments across two base rates",
                     "Three wording conditions",
                     "One pinned model version (jev-1.13.0)"],
            accent="Total cost of the entire study: five cents."),

        image_slide(
            4, "the result", f"{R}/natural_strict_calibration.png",
            "Every band sits below the diagonal: a systematic lean toward "
            "\"yes\", not an artefact of where I drew the threshold."),

        text_slide(
            5, "the split", "The ranking is good.\nThe numbers are not.",
            body="A model that emits probabilities has two ways to be good, and "
                 "they fail separately.",
            bullets=["AUC 0.91 — it reliably sorts worse content above better. "
                     "It understands the task.",
                     "ECE 0.156 — the probabilities attached to that ranking "
                     "don't mean what they say."],
            accent="On one question, Jev at the default threshold scored 61% "
                   "where answering \"no\" every time scores 68% — while "
                   "ranking at AUC 0.83."),

        text_slide(
            6, "the obvious objection", "\"Your prompts were\njust badly written.\"",
            body="Fair. So I tested four, in a single call over identical "
                 "state, so only the phrasing varied.",
            bullets=["Error ranged 0.21 to 0.52 — and the annotators' own "
                     "definition scored worst",
                     "AUC barely moved: wording shifts the scale, not the signal",
                     "The strictest still predicted 26% where reality was 5%"],
            accent="Re-ran everything with the best wording. Error fell 0.209 → "
                   "0.156. The bias survived."),

        image_slide(
            7, "the fix", f"{R}/natural_strict_recalibrated.png",
            "Two parameters, fitted on one half of the data and scored on the "
            "half it never saw."),

        stat_slide(
            8, "the fix", "96%", "of the calibration error removed",
            "A logistic fit — about 20 lines — on held-out data. AUC went "
            "0.918 → 0.918, because a monotonic rescale reorders nothing.\n\n"
            "The ranking was always there. Only the units were wrong."),

        text_slide(
            9, "honestly", "What I'd want asked\nback at me.",
            bullets=["One dataset, one domain, one model version.",
                     "One question has a single positive in 400. Meaningless, "
                     "and labelled as such rather than dropped.",
                     "Four wordings is four, not all of them.",
                     "I nearly called 95% accuracy a win — flagging nothing "
                     "scores 97% here."],
            accent="This isn't an argument against the model. As a high-recall "
                   "pre-filter with a calibration layer, it works."),

        text_slide(
            10, "the takeaway", "Typed output guarantees\nthe interface.\nNot the truth.",
            body="If a model hands you a probability and you write a threshold "
                 "against it, measure it first. Mine took an evening and cost "
                 "five cents, and every threshold I'd have written was wrong.",
            accent="Full code, charts and raw data:\n"
                   "github.com/Adilmp/does-jev-confidence-mean-anything"),
    ]

    os.makedirs(f"{R}/slides", exist_ok=True)
    pdf_path = f"{R}/carousel.pdf"
    with PdfPages(pdf_path) as pdf:
        for i, fig in enumerate(slides, 1):
            pdf.savefig(fig, facecolor=SURFACE)
            fig.savefig(f"{R}/slides/{i:02d}.png", facecolor=SURFACE, dpi=DPI)
            plt.close(fig)

    print(f"wrote {pdf_path} ({len(slides)} slides)")
    print(f"wrote {R}/slides/*.png")
    return pdf_path


if __name__ == "__main__":
    build()
