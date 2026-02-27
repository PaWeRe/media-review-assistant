#!/usr/bin/env python3
"""
MEDIA journal 2nd-round review assistant using Ollama.
Reads the 1st draft, reviewer comments, and revised (2nd) draft,
then produces a structured review via a 3-pass local LLM pipeline.
All processing stays on your machine -- no data is sent externally.
"""

import argparse
import os
import sys
import textwrap
import time
from pathlib import Path

import requests

# Ensure progress output is visible immediately (unbuffered print).
_print = print
def print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    _print(*args, **kwargs)

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "gemma3:12b"
DEFAULT_CONTEXT_LENGTH = 32768
MAX_CHARS = 50000  # ~12.5K tokens, safe for 32K context per pass


# ---------------------------------------------------------------------------
# Text extraction (reused from review_assistant.py)
# ---------------------------------------------------------------------------

def extract_text_from_pdf(filepath: str) -> str:
    import fitz  # pymupdf
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
# 3-pass prompts
# ---------------------------------------------------------------------------

PASS1_PROMPT = textwrap.dedent("""\
    You are an expert peer reviewer for a scientific journal. Your task is to
    carefully analyse the ORIGINAL MANUSCRIPT and the REVIEWER COMMENTS from the
    first round of peer review.

    ORIGINAL MANUSCRIPT (1st draft):
    ---
    {draft1_text}
    ---

    REVIEWER COMMENTS (from 1st round):
    ---
    {comments_text}
    ---

    Produce a structured analysis with the following format:

    ## Reviewer Concerns
    For EACH concern or suggestion raised by the reviewer(s), create a numbered
    item with:
    - **Concern**: A concise restatement of the reviewer's point.
    - **Section affected**: Which part of the manuscript it relates to
      (e.g. Methods, Results, Discussion, Figures, References).
    - **Expected change**: What the authors should have done to address this.
    - **Severity**: Critical / Major / Minor

    Be exhaustive -- do not skip any reviewer comment, even minor ones.
""")

PASS2_PROMPT = textwrap.dedent("""\
    You are an expert peer reviewer conducting a 2nd-round review. You have
    already analysed the reviewer concerns from the first round (listed below).
    Now you must check the REVISED MANUSCRIPT to determine how the authors
    addressed each concern.

    REVISION EXPECTATIONS (from your previous analysis):
    ---
    {pass1_output}
    ---

    REVISED MANUSCRIPT (2nd draft):
    ---
    {draft2_text}
    ---

    For EACH numbered concern from the expectations list, produce:
    - **Concern #N**: Restate briefly.
    - **Status**: Fully Addressed / Partially Addressed / Not Addressed
    - **Evidence**: Quote or reference the specific section in the revised
      manuscript that shows how (or whether) the authors responded.
    - **Remaining issue** (if any): What is still missing or insufficient.

    After the per-concern assessment, add a section:

    ## New Issues
    List any NEW problems you notice in the revised manuscript that were not
    present in the original (e.g. introduced errors, inconsistencies, unclear
    new text). If there are none, write "No new issues identified."
""")

PASS3_PROMPT = textwrap.dedent("""\
    You are an expert peer reviewer writing the FINAL 2nd-round review for a
    manuscript submitted to the MEDIA journal. You have already assessed how the
    authors addressed all first-round reviewer concerns (assessment below).
    Now produce the formal review.

    REVISION FULFILLMENT ASSESSMENT:
    ---
    {pass2_output}
    ---

    REVISED MANUSCRIPT (2nd draft):
    ---
    {draft2_text}
    ---

    This is a 2nd-round review. The authors have already revised the manuscript
    once in response to reviewer feedback. Your review should acknowledge the
    revision effort and focus on whether the concerns were satisfactorily
    addressed. Be fair and constructive.

    Produce your review in EXACTLY the following format:

    ## 1. Overall Manuscript Rating

    Provide a single integer score from 1 to 100, where:
    - 90-100 = Excellent, ready for publication
    - 75-89  = Good, minor issues remain
    - 60-74  = Acceptable but needs further work
    - 40-59  = Significant problems remain
    - 1-39   = Fundamentally flawed

    Format: **Rating: XX/100**

    ## 2. Recommendation

    Choose EXACTLY ONE of these options (copy the text verbatim):
    - Accept, no additional revision required
    - Accept, subject to minor revision - no second review
    - Accept, subject to minor revision - second review
    - Reject - Major revision & resubmit in 3 months
    - Reject - No further consideration

    Format: **Recommendation: <chosen option>**

    ## 3. Reviewer Comments to Author

    Write your comments as a series of short paragraphs. Cover ALL of the
    following aspects:

    **Summary of the revised manuscript**: Briefly describe the paper's topic,
    aim, and main contribution (2-3 sentences).

    **Assessment of revisions**: Discuss how the authors addressed the previous
    reviewer feedback. Highlight concerns that were well addressed and note any
    that remain partially or fully unaddressed.

    **Content and methodology**: Comment on the scientific content, study design,
    data analysis, and validity of conclusions in the revised version.

    **Presentation and form**: Comment on writing quality, clarity, figures,
    tables, and overall presentation.

    **Remaining concerns**: If any issues persist, list them clearly and
    specifically so the authors know what to fix. If none, state that the
    revisions are satisfactory.

    **Final remarks**: A brief closing statement summarising your overall
    impression of the revised manuscript.
""")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    draft1_text: str,
    comments_text: str,
    draft2_text: str,
    model: str,
    context_length: int,
) -> tuple[str, str, str]:
    """Run the 3-pass review pipeline. Returns (pass1, pass2, pass3) outputs."""

    # Pass 1
    print("\n[Pass 1/3] Analysing reviewer concerns and revision expectations...")
    t0 = time.time()
    prompt1 = PASS1_PROMPT.format(
        draft1_text=truncate_text(draft1_text),
        comments_text=truncate_text(comments_text),
    )
    pass1 = query_ollama(prompt1, model, context_length)
    print(f"  Done ({time.time() - t0:.0f}s)")

    # Pass 2
    print("[Pass 2/3] Checking how revisions were addressed in 2nd draft...")
    t0 = time.time()
    prompt2 = PASS2_PROMPT.format(
        pass1_output=pass1,
        draft2_text=truncate_text(draft2_text),
    )
    pass2 = query_ollama(prompt2, model, context_length)
    print(f"  Done ({time.time() - t0:.0f}s)")

    # Pass 3
    print("[Pass 3/3] Generating final MEDIA review...")
    t0 = time.time()
    prompt3 = PASS3_PROMPT.format(
        pass2_output=pass2,
        draft2_text=truncate_text(draft2_text),
    )
    pass3 = query_ollama(prompt3, model, context_length)
    print(f"  Done ({time.time() - t0:.0f}s)")

    return pass1, pass2, pass3


def format_final_review(pass3_output: str) -> str:
    border = "=" * 70
    lines = [
        border,
        "MEDIA JOURNAL — 2ND ROUND PEER REVIEW",
        border,
        "",
        pass3_output.strip(),
        "",
        border,
        "Generated by local Ollama review assistant. No data left this machine.",
        border,
    ]
    return "\n".join(lines)


def format_full_report(pass1: str, pass2: str, pass3: str) -> str:
    """Verbose report including intermediate passes (saved alongside the review)."""
    border = "=" * 70
    sections = [
        (border,),
        ("MEDIA JOURNAL — 2ND ROUND PEER REVIEW  (full report)",),
        (border,),
        ("",),
        ("## Pass 1: Revision Expectations",),
        ("-" * 40,),
        (pass1.strip(),),
        ("",),
        ("## Pass 2: Revision Fulfillment Check",),
        ("-" * 40,),
        (pass2.strip(),),
        ("",),
        ("## Pass 3: Final Review",),
        ("-" * 40,),
        (pass3.strip(),),
        ("",),
        (border,),
        ("Generated by local Ollama review assistant. No data left this machine.",),
        (border,),
    ]
    return "\n".join(s[0] for s in sections)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "MEDIA journal 2nd-round review assistant (Ollama). "
            "All data stays on your machine."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Example:
              python media_review_assistant.py \\
                --draft1 first_draft.pdf \\
                --comments reviewer_comments.txt \\
                --draft2 revised_draft.pdf \\
                --output review.md
        """),
    )
    parser.add_argument(
        "--draft1", "-d1",
        help="Path to the 1st (original) draft (PDF, DOCX, TXT, MD, TEX)",
    )
    parser.add_argument(
        "--comments", "-c",
        help="Path to the 1st-round reviewer comments (PDF, DOCX, TXT, MD, TEX)",
    )
    parser.add_argument(
        "--draft2", "-d2",
        help="Path to the 2nd (revised/final) draft (PDF, DOCX, TXT, MD, TEX)",
    )
    parser.add_argument(
        "--output", "-o",
        help="Save review to this file (default: print to stdout)",
    )
    parser.add_argument(
        "--full-report", action="store_true",
        help="Also save intermediate pass outputs (appends _full to filename)",
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

    missing = []
    if not args.draft1:
        missing.append("--draft1/-d1")
    if not args.comments:
        missing.append("--comments/-c")
    if not args.draft2:
        missing.append("--draft2/-d2")
    if missing:
        parser.error(f"the following arguments are required: {', '.join(missing)}")

    for label, path in [("1st draft", args.draft1), ("Comments", args.comments), ("2nd draft", args.draft2)]:
        if not os.path.isfile(path):
            sys.exit(f"File not found ({label}): {path}")

    print("Reading input documents...")
    draft1_text = extract_text(args.draft1)
    comments_text = extract_text(args.comments)
    draft2_text = extract_text(args.draft2)

    print(f"  1st draft:  {len(draft1_text):>7,} chars  ({args.draft1})")
    print(f"  Comments:   {len(comments_text):>7,} chars  ({args.comments})")
    print(f"  2nd draft:  {len(draft2_text):>7,} chars  ({args.draft2})")

    for label, text in [("1st draft", draft1_text), ("2nd draft", draft2_text)]:
        if len(text) < 100:
            sys.exit(
                f"{label} extracted text is very short ({len(text)} chars). "
                "The file may be scanned images without OCR."
            )

    print(f"\nModel: {args.model}  |  Context: {args.context_length} tokens")
    print("All processing is LOCAL. No data leaves this machine.")

    pass1, pass2, pass3 = run_pipeline(
        draft1_text, comments_text, draft2_text,
        args.model, args.context_length,
    )

    review = format_final_review(pass3)
    print("\n" + review)

    if args.output:
        out = Path(args.output)
        out.write_text(review, encoding="utf-8")
        print(f"\nReview saved to: {out}")

        if args.full_report:
            full_path = out.with_stem(out.stem + "_full")
            full_path.write_text(
                format_full_report(pass1, pass2, pass3), encoding="utf-8",
            )
            print(f"Full report saved to: {full_path}")


if __name__ == "__main__":
    main()
