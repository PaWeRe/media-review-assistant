#!/usr/bin/env python3
"""
TMI peer-review assistant using Ollama vision models.

2-pass pipeline:
  Pass 1 — Detailed analysis (vision: PDF page images + reviewer notes)
  Pass 2 — Formal TMI review (evaluation matrix, comments, recommendation)

Renders PDF pages as images so the model sees figures, tables, and equations
exactly as a human reviewer would.
All processing stays on your machine — no data is sent externally.
"""

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path

import requests

_print = print


def print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    _print(*args, **kwargs)


OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "qwen3-vl:30b"
DEFAULT_DPI = 200
DEFAULT_CONTEXT_LENGTH = 32768

# ---------------------------------------------------------------------------
# TMI evaluation criteria
# ---------------------------------------------------------------------------

TMI_EVAL_CATEGORIES = [
    "Scientific Merit",
    "Originality",
    "Impact",
    "Practical Value",
    "Reader Interest",
    "Overall Evaluation",
]

TMI_RATING_SCALE = "Excellent / Very Good / Good / Fair / Poor"

TMI_RECOMMENDATIONS = [
    "Accept as submitted",
    "Accept after minor revision (no re-review needed)",
    "Major revision required (re-review needed)",
    "Reject and encourage resubmission",
    "Reject — no further consideration",
]

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

NOTES_BLOCK = """\

The reviewer has provided preliminary notes and observations below. \
Use these to guide and prioritize your analysis. Validate the reviewer's \
concerns and hypotheses, investigate any open questions they raised, and \
surface additional issues the notes may have missed.

REVIEWER NOTES:
---
{notes}
---
"""

PASS1_PROMPT_TEMPLATE = """\
You are helping a peer reviewer assess a manuscript submitted to \
IEEE Transactions on Medical Imaging (TMI). The complete paper is provided \
as page images.
{notes_block}\
Produce a structured analysis covering the sections below. Be concrete: \
cite page numbers, figure/table numbers, equation numbers, and quote or \
closely paraphrase specific author claims. Avoid generic praise or filler. \
Do NOT produce a recommendation or rating — the formal review will be \
generated in a separate step.

## 1. Paper Digest

5-6 sentences: problem addressed, proposed method (core idea, not a laundry \
list of components), datasets used, headline quantitative results. Then for \
each key figure and table, one sentence on what it shows.

## 2. Scientific Merit

Evaluate the soundness and rigor of the methodology. Identify any \
methodological flaws, unjustified assumptions, or missing validations. \
Comment on experimental design, hyperparameter choices, and reproducibility.

## 3. Novelty Audit

For EACH novelty claim the authors make:
- Quote or paraphrase the claim with section/page reference.
- Identify the closest prior work and how the authors differentiate.
- Assess whether the differentiation is substantiated or superficial.

Flag anything incremental, overstated, or already addressed by existing work.

## 4. Literature Gaps

List specific missing references or research directions that would \
materially change how the contribution is positioned. For each entry:
- Give enough detail to verify (first author, short title, venue, approx year).
- Explain WHY it matters for this paper.
- Mark confidence: [HIGH] if certain, [MEDIUM] if likely, [LOW] if uncertain.

Do NOT fabricate full bibliographic details.

## 5. Experimental Scrutiny

Address each explicitly:
- Are baselines current and fairly compared (same backbone, data, budget)?
- Are datasets standard and sufficient for the claimed scope?
- Do effect sizes support the claims, or are improvements within noise?
- Are ablations sufficient to isolate each claimed contribution?
- What analysis is missing (statistical tests, failure cases, compute cost)?

## 6. Figures & Tables

For each key figure and table: what does it convey, is it effective, and \
is anything misleading or redundant? Identify which are critical evidence \
and which add little. Flag any visual evidence that contradicts the text.

## 7. Writing & Presentation

Comment on structure, clarity, grammar, and whether technical details are \
accessible. Note any repetitive or unclear passages.

## 8. Specific Issues

Numbered list of concrete, actionable problems the authors should address. \
Reference specific sections, figures, equations, or tables.
"""

PASS2_PROMPT_TEMPLATE = """\
You are writing the FORMAL peer review for a manuscript submitted to \
IEEE Transactions on Medical Imaging (TMI). You have already performed \
a detailed analysis of the paper (provided below). Now synthesize that \
analysis into the official review format.
{notes_block}\
DETAILED ANALYSIS:
---
{pass1_output}
---

Produce your review in EXACTLY the following format:

## 1. Evaluation Matrix

Rate each category using the scale: {rating_scale}

| Category | Rating |
|----------|--------|
{eval_rows}

Immediately below the table, provide a 1-2 sentence justification for EACH \
category rating.

## 2. Comments to the Author

Write your comments as focused paragraphs covering ALL of the following:

**Significance and innovation:** Assess the paper's main contribution and \
novelty claims. Be specific about what is genuinely new vs. incremental.

**Study design and technical approach:** Comment on methodology, baselines, \
datasets, metrics, ablations. Note any gaps.

**Content and results:** Evaluate whether results support the claims. \
Comment on effect sizes, statistical rigor, missing analyses.

**Writing and presentation:** Comment on clarity, structure, figures, \
tables, and presentation issues.

**Specific concerns:** Numbered list of the most important issues the \
authors must address, with section/figure/table references.

**Missing references:** List critical references the authors should \
consider, with brief justification.

## 3. Recommendation

Choose EXACTLY ONE of:
{recommendations}

Format: **Recommendation: <chosen option>**

Justify your recommendation in 2-3 sentences referencing key strengths \
and weaknesses.
"""


# ---------------------------------------------------------------------------
# PDF rendering
# ---------------------------------------------------------------------------

def render_pdf_pages(filepath: str, dpi: int = DEFAULT_DPI) -> list[str]:
    """Render each PDF page to a base64-encoded PNG with progress tracking."""
    import fitz

    doc = fitz.open(filepath)
    n_pages = len(doc)
    pages = []

    try:
        from tqdm import tqdm
        iterator = tqdm(doc, total=n_pages, desc="  Rendering pages", unit="pg")
    except ImportError:
        iterator = doc

    first = True
    for page in iterator:
        pix = page.get_pixmap(dpi=dpi)
        if first:
            print(f"  Page dimensions: {pix.width}x{pix.height} px")
            first = False
        png_bytes = pix.tobytes("png")
        if len(png_bytes) < 100:
            print(f"  Warning: page {page.number + 1} rendered to only "
                  f"{len(png_bytes)} bytes — may be blank")
        pages.append(base64.b64encode(png_bytes).decode("ascii"))

    doc.close()
    return pages


# ---------------------------------------------------------------------------
# Ollama interface
# ---------------------------------------------------------------------------

def query_ollama(
    prompt: str,
    model: str,
    images: list[str] | None = None,
    context_length: int = DEFAULT_CONTEXT_LENGTH,
    stream: bool = True,
) -> str:
    """Send prompt (+ optional page images) to a local Ollama model."""
    message: dict = {"role": "user", "content": prompt}
    if images:
        message["images"] = images

    payload = {
        "model": model,
        "messages": [message],
        "stream": stream,
        "options": {
            "temperature": 0.3,
            "num_ctx": context_length,
        },
    }

    try:
        resp = requests.post(
            OLLAMA_CHAT_URL, json=payload, timeout=7200, stream=stream,
        )
        resp.raise_for_status()
    except requests.ConnectionError:
        sys.exit(
            "Cannot connect to Ollama. Make sure it's running:\n"
            "  ollama serve"
        )
    except requests.Timeout:
        sys.exit("Ollama request timed out (2-hour limit).")
    except requests.HTTPError as e:
        body = ""
        try:
            body = resp.text[:500]
        except Exception:
            pass
        sys.exit(f"Ollama HTTP error: {e}\n{body}")
    except Exception as e:
        sys.exit(f"Ollama error: {e}")

    if not stream:
        data = resp.json()
        return data.get("message", {}).get("content", "")

    chunks: list[str] = []
    token_count = 0
    t0 = time.time()
    eval_count = 0

    for line in resp.iter_lines():
        if not line:
            continue
        data = json.loads(line)
        token = data.get("message", {}).get("content", "")
        if token:
            _print(token, end="", flush=True)
            chunks.append(token)
            token_count += 1
        if data.get("done"):
            eval_count = data.get("eval_count", token_count)
            break

    elapsed = time.time() - t0
    _print()
    tok_s = eval_count / elapsed if elapsed > 0 else 0
    print(f"  [{eval_count} tokens in {elapsed:.0f}s — {tok_s:.1f} tok/s]")
    return "".join(chunks)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def build_notes_block(notes_path: str | None) -> str:
    """Read reviewer notes and format for inclusion in prompts."""
    if not notes_path:
        return ""
    path = Path(notes_path)
    if not path.is_file():
        sys.exit(f"Notes file not found: {notes_path}")
    notes = path.read_text(encoding="utf-8").strip()
    if not notes:
        return ""
    return NOTES_BLOCK.format(notes=notes)


def run_pipeline(
    pages: list[str],
    notes_path: str | None,
    model: str,
    context_length: int,
    stream: bool,
) -> tuple[str, str]:
    """Run the 2-pass review pipeline. Returns (pass1, pass2) outputs."""
    notes_block = build_notes_block(notes_path)

    # --- Pass 1: Detailed analysis (vision) ---
    print("\n" + "=" * 60)
    print("[Pass 1/2] Detailed Analysis (vision model)")
    print("=" * 60 + "\n")

    prompt1 = PASS1_PROMPT_TEMPLATE.format(notes_block=notes_block)
    t0 = time.time()
    pass1 = query_ollama(
        prompt1, model,
        images=pages,
        context_length=context_length,
        stream=stream,
    )
    print(f"  Pass 1 total: {time.time() - t0:.0f}s")

    # --- Pass 2: Formal TMI review (text only) ---
    print("\n" + "=" * 60)
    print("[Pass 2/2] Formal TMI Review")
    print("=" * 60 + "\n")

    eval_rows = "\n".join(f"| {cat} | |" for cat in TMI_EVAL_CATEGORIES)
    recommendations = "\n".join(f"- {r}" for r in TMI_RECOMMENDATIONS)

    prompt2 = PASS2_PROMPT_TEMPLATE.format(
        notes_block=notes_block,
        pass1_output=pass1,
        rating_scale=TMI_RATING_SCALE,
        eval_rows=eval_rows,
        recommendations=recommendations,
    )
    t0 = time.time()
    pass2 = query_ollama(
        prompt2, model,
        context_length=context_length,
        stream=stream,
    )
    print(f"  Pass 2 total: {time.time() - t0:.0f}s")

    return pass1, pass2


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_review(pass2_output: str) -> str:
    border = "=" * 70
    return "\n".join([
        border,
        "IEEE TRANSACTIONS ON MEDICAL IMAGING — PEER REVIEW",
        border,
        "",
        pass2_output.strip(),
        "",
        border,
        "Generated by local Ollama review assistant. No data left this machine.",
        border,
    ])


def format_full_report(pass1: str, pass2: str) -> str:
    border = "=" * 70
    return "\n".join([
        border,
        "IEEE TRANSACTIONS ON MEDICAL IMAGING — PEER REVIEW  (full report)",
        border,
        "",
        "## Pass 1: Detailed Analysis",
        "-" * 40,
        pass1.strip(),
        "",
        "## Pass 2: Formal Review",
        "-" * 40,
        pass2.strip(),
        "",
        border,
        "Generated by local Ollama review assistant. No data left this machine.",
        border,
    ])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "TMI review assistant — 2-pass local Ollama VL pipeline. "
            "Sends PDF pages as images so figures/tables are visible. "
            "Privacy-preserving: all data stays on this machine."
        ),
    )
    parser.add_argument(
        "--manuscript", "-m", required=True,
        help="Path to the manuscript PDF",
    )
    parser.add_argument(
        "--notes", "-n",
        help="Path to reviewer notes file (TXT/MD) to guide the analysis",
    )
    parser.add_argument(
        "--output", "-o",
        help="Save review to this markdown file",
    )
    parser.add_argument(
        "--full-report", action="store_true",
        help="Also save full report with intermediate Pass 1 analysis",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Ollama VL model (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--dpi", type=int, default=DEFAULT_DPI,
        help=f"PDF render resolution in DPI (default: {DEFAULT_DPI})",
    )
    parser.add_argument(
        "--context-length", type=int, default=DEFAULT_CONTEXT_LENGTH,
        help=f"Model context window (default: {DEFAULT_CONTEXT_LENGTH})",
    )
    parser.add_argument(
        "--no-stream", action="store_true",
        help="Wait for full response instead of streaming tokens",
    )
    parser.add_argument(
        "--list-models", action="store_true",
        help="List available Ollama models and exit",
    )

    args = parser.parse_args()

    if args.list_models:
        try:
            resp = requests.get("http://localhost:11434/api/tags", timeout=5)
            models = resp.json().get("models", [])
            _print("Available Ollama models:")
            for m in models:
                size_gb = m.get("size", 0) / 1e9
                _print(f"  {m['name']:<40s} {size_gb:.1f} GB")
        except Exception:
            _print("Cannot connect to Ollama. Is it running?")
        return

    if not os.path.isfile(args.manuscript):
        sys.exit(f"File not found: {args.manuscript}")

    ext = Path(args.manuscript).suffix.lower()
    if ext != ".pdf":
        sys.exit(f"Only PDF files are supported (got {ext}).")

    # Render PDF pages as images
    print(f"Rendering {args.manuscript} at {args.dpi} DPI ...")
    pages = render_pdf_pages(args.manuscript, dpi=args.dpi)
    raw_mb = sum(len(p) for p in pages) * 3 / 4 / 1e6
    print(f"  {len(pages)} pages (~{raw_mb:.1f} MB image data)")

    if args.notes:
        notes_text = Path(args.notes).read_text(encoding="utf-8").strip()
        print(f"  Reviewer notes: {args.notes} ({len(notes_text):,} chars)")

    print(f"\nModel:   {args.model}")
    print(f"Context: {args.context_length:,} tokens")
    print("All processing is LOCAL. No data leaves this machine.")

    t_start = time.time()
    pass1, pass2 = run_pipeline(
        pages,
        notes_path=args.notes,
        model=args.model,
        context_length=args.context_length,
        stream=not args.no_stream,
    )
    total_elapsed = time.time() - t_start

    print(f"\n{'=' * 60}")
    print(f"Pipeline complete in {total_elapsed:.0f}s "
          f"({total_elapsed / 60:.1f} min)")
    print(f"{'=' * 60}")

    if args.output:
        out = Path(args.output)
        out.write_text(format_review(pass2), encoding="utf-8")
        print(f"Review saved to: {out}")

        if args.full_report:
            full_path = out.with_stem(out.stem + "_full")
            full_path.write_text(
                format_full_report(pass1, pass2), encoding="utf-8",
            )
            print(f"Full report saved to: {full_path}")


if __name__ == "__main__":
    main()
