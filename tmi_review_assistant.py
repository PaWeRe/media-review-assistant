#!/usr/bin/env python3
"""
TMI journal 1st-round review assistant using Ollama.
Reads a single manuscript and produces a structured TMI review
via a 2-pass local LLM pipeline.
All processing stays on your machine -- no data is sent externally.
"""

import argparse
import os
import sys
import textwrap
import time
from pathlib import Path

import requests

_print = print
def print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    _print(*args, **kwargs)

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "gemma3:12b"
DEFAULT_CONTEXT_LENGTH = 32768
MAX_CHARS = 100000  # ~25K tokens, safe for 65K context per pass


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

def extract_text_from_pdf(filepath: str) -> str:
    import fitz
    doc = fitz.open(filepath)
    text_parts = []
    for page in doc:
        text_parts.append(page.get_text())
    doc.close()
    return "\n".join(text_parts)


def extract_text_from_docx(filepath: str) -> str:
    from docx import Document
    doc = Document(filepath)
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def extract_text(filepath: str) -> str:
    path = Path(filepath)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_text_from_pdf(filepath)
    elif suffix in (".docx", ".doc"):
        return extract_text_from_docx(filepath)
    elif suffix in (".txt", ".md", ".tex"):
        return path.read_text(encoding="utf-8")
    else:
        sys.exit(f"Unsupported file type: {suffix}. Use PDF, DOCX, TXT, MD, or TEX.")


# ---------------------------------------------------------------------------
# Ollama interface
# ---------------------------------------------------------------------------

def query_ollama(prompt: str, model: str, context_length: int = DEFAULT_CONTEXT_LENGTH) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_ctx": context_length,
            "temperature": 0.3,
        },
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=900)
        resp.raise_for_status()
        return resp.json().get("response", "")
    except requests.ConnectionError:
        sys.exit(
            "Cannot connect to Ollama. Make sure it's running:\n"
            "  ollama serve"
        )
    except requests.Timeout:
        sys.exit("Ollama request timed out. Try reducing --context-length.")
    except Exception as e:
        sys.exit(f"Ollama error: {e}")


def truncate_text(text: str, max_chars: int = MAX_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return (
        text[:half]
        + "\n\n[... middle section truncated to fit context window ...]\n\n"
        + text[-half:]
    )


# ---------------------------------------------------------------------------
# 2-pass prompts
# ---------------------------------------------------------------------------

PASS1_PROMPT = textwrap.dedent("""\
    You are an expert peer reviewer for IEEE Transactions on Medical Imaging
    (TMI), a top-tier journal in medical imaging, image processing, and
    computational methods for biomedical applications. You have deep expertise
    in research methodology, biostatistics, machine learning for medical imaging,
    and scientific writing.

    Carefully analyse the following manuscript and produce a DETAILED assessment.

    MANUSCRIPT:
    ---
    {manuscript_text}
    ---

    Produce a thorough analysis covering each of the following dimensions.
    For each dimension, provide specific evidence from the manuscript.

    ## 1. Scientific Merit
    Evaluate the soundness of the research question, methodology, experimental
    design, and validity of conclusions. Are the methods rigorous? Are the
    results convincing? Are there methodological flaws?

    ## 2. Originality
    How novel is the contribution? Does it advance the state of the art? Is there
    sufficient differentiation from prior work? Are the claims of novelty
    justified?

    ## 3. Impact
    What is the potential impact on the field of medical imaging? Will this work
    influence future research or clinical practice?

    ## 4. Practical Value
    Is the work practically useful? Can the methods be reproduced and applied
    by others? Are datasets/code available or described sufficiently?

    ## 5. Reader Interest
    Would TMI readers find this paper interesting and relevant? Is it well-suited
    for the journal's scope?

    ## 6. Significance and Innovation
    What is the key innovation? How significant is the contribution relative to
    the existing literature?

    ## 7. Study Design and Technical Approach
    Evaluate the experimental design, baselines, datasets, evaluation metrics,
    and statistical analyses. Are comparisons fair? Are ablation studies adequate?

    ## 8. Writing and Presentation
    Assess clarity, structure, figures, tables, and overall readability. Are
    there grammar or formatting issues?

    ## 9. Missing References
    Identify any important related work that the authors failed to cite. For each
    missing reference, provide as much bibliographic detail as you can (authors,
    title, journal/conference, year).

    Be specific and constructive throughout. Reference sections, figures, tables,
    and equations by number where possible.
""")

PASS2_PROMPT = textwrap.dedent("""\
    You are an expert peer reviewer writing the FORMAL first-round review for a
    manuscript submitted to IEEE Transactions on Medical Imaging (TMI). You have
    already performed a detailed analysis of the manuscript (below). Now produce
    the official review.

    YOUR DETAILED ANALYSIS:
    ---
    {pass1_output}
    ---

    MANUSCRIPT:
    ---
    {manuscript_text}
    ---

    Produce your review in EXACTLY the following format:

    ## 1. Evaluation Matrix

    Rate the manuscript on each criterion using EXACTLY one of: Excellent, Very
    Good, Good, Fair, Poor. Provide the ratings in a table:

    | Category           | Rating |
    |--------------------|--------|
    | Scientific Merit   | ...    |
    | Originality        | ...    |
    | Impact             | ...    |
    | Practical Value    | ...    |
    | Reader Interest    | ...    |
    | Overall Evaluation | ...    |

    After the table, provide a brief (1-2 sentence) justification for each
    rating.

    ## 2. Comments to the Author

    Write your comments as a series of short, focused paragraphs. Be SPECIFIC
    and CONSTRUCTIVE. You MUST cover all of the following:

    **Significance and innovation**: What is the paper's main contribution and
    how significant is it? Is the claimed novelty justified relative to prior
    work?

    **Study design and technical approach**: Comment on the experimental design,
    baselines, evaluation, and statistical rigour. Identify specific strengths
    and weaknesses.

    **Content and results**: Discuss the quality and presentation of results.
    Are the conclusions supported by the evidence? Are there over-claims?

    **Writing and presentation**: Comment on clarity, structure, figures, tables,
    and overall readability. Note specific issues.

    **Specific concerns**: List any concrete issues (numbered) that the authors
    should address. Be precise -- reference sections, figures, equations, or
    tables by number.

    **Additional references**: List references the authors should consider. For
    EACH reference, provide the FULL bibliographic citation including authors,
    title, journal or conference name, volume, pages, year, and DOI (if known).
    Format example:
    - A. Smith, B. Jones, "Title of Paper," IEEE Trans. Med. Imaging, vol. 40,
      no. 3, pp. 800-810, Mar. 2021, doi: 10.1109/TMI.2020.XXXXXXX.

    If no additional references are needed, state "No additional references
    suggested."

    ## 3. Recommendation

    Choose EXACTLY ONE of these options (copy the text verbatim):
    - Accept
    - Accept pending minor revision: no external review required
    - Reject/Resubmit: major revisions needed and new external review required
    - Reject with recommendation to submit to another journal
    - Reject

    Format: **Recommendation: <chosen option>**

    Provide a brief (2-3 sentence) justification for your recommendation.
""")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    manuscript_text: str,
    model: str,
    context_length: int,
) -> tuple[str, str]:
    """Run the 2-pass review pipeline. Returns (pass1, pass2) outputs."""

    text = truncate_text(manuscript_text)

    # Pass 1
    print("\n[Pass 1/2] Deep analysis of manuscript...")
    t0 = time.time()
    prompt1 = PASS1_PROMPT.format(manuscript_text=text)
    pass1 = query_ollama(prompt1, model, context_length)
    print(f"  Done ({time.time() - t0:.0f}s)")

    # Pass 2
    print("[Pass 2/2] Generating formal TMI review...")
    t0 = time.time()
    prompt2 = PASS2_PROMPT.format(
        pass1_output=pass1,
        manuscript_text=text,
    )
    pass2 = query_ollama(prompt2, model, context_length)
    print(f"  Done ({time.time() - t0:.0f}s)")

    return pass1, pass2


def format_final_review(pass2_output: str) -> str:
    border = "=" * 70
    lines = [
        border,
        "IEEE TRANSACTIONS ON MEDICAL IMAGING — PEER REVIEW",
        border,
        "",
        pass2_output.strip(),
        "",
        border,
        "Generated by local Ollama review assistant. No data left this machine.",
        border,
    ]
    return "\n".join(lines)


def format_full_report(pass1: str, pass2: str) -> str:
    border = "=" * 70
    lines = [
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
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "TMI journal 1st-round review assistant (Ollama). "
            "All data stays on your machine."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Example:
              python tmi_review_assistant.py \\
                --manuscript paper.pdf \\
                --output review.md --full-report
        """),
    )
    parser.add_argument(
        "--manuscript", "-m",
        help="Path to the manuscript (PDF, DOCX, TXT, MD, TEX)",
    )
    parser.add_argument(
        "--output", "-o",
        help="Save review to this file (default: print to stdout)",
    )
    parser.add_argument(
        "--full-report", action="store_true",
        help="Also save the detailed analysis pass (appends _full to filename)",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Ollama model to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--context-length", type=int, default=DEFAULT_CONTEXT_LENGTH,
        help=f"Model context window size (default: {DEFAULT_CONTEXT_LENGTH})",
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
            print("Available Ollama models:")
            for m in models:
                size_gb = m.get("size", 0) / 1e9
                print(f"  {m['name']:<20s} {size_gb:.1f} GB")
        except Exception:
            print("Cannot connect to Ollama. Is it running?")
        return

    if not args.manuscript:
        parser.error("the following argument is required: --manuscript/-m")

    if not os.path.isfile(args.manuscript):
        sys.exit(f"File not found: {args.manuscript}")

    print(f"Reading: {args.manuscript}")
    manuscript_text = extract_text(args.manuscript)
    print(f"  Extracted {len(manuscript_text):,} characters")

    if len(manuscript_text) < 100:
        sys.exit(
            f"Extracted text is very short ({len(manuscript_text)} chars). "
            "The file may be scanned images without OCR."
        )

    print(f"\nModel: {args.model}  |  Context: {args.context_length} tokens")
    print("All processing is LOCAL. No data leaves this machine.")

    pass1, pass2 = run_pipeline(
        manuscript_text, args.model, args.context_length,
    )

    review = format_final_review(pass2)
    print("\n" + review)

    if args.output:
        out = Path(args.output)
        out.write_text(review, encoding="utf-8")
        print(f"\nReview saved to: {out}")

        if args.full_report:
            full_path = out.with_stem(out.stem + "_full")
            full_path.write_text(
                format_full_report(pass1, pass2), encoding="utf-8",
            )
            print(f"Full report saved to: {full_path}")


if __name__ == "__main__":
    main()
