#!/usr/bin/env python3
"""
Fully local manuscript review assistant using Ollama.
All processing stays on your machine -- no data is sent externally.
"""

import argparse
import json
import os
import sys
import textwrap
from pathlib import Path

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "gemma3:12b"

REVIEW_SECTIONS = {
    "summary": {
        "title": "Manuscript Summary",
        "prompt": textwrap.dedent("""\
            Provide a concise summary of this manuscript in 3-5 sentences.
            Identify: the research question, study design, main findings, and primary conclusion.
        """),
    },
    "strengths": {
        "title": "Strengths",
        "prompt": textwrap.dedent("""\
            Identify the major strengths of this manuscript. Consider:
            - Novelty and significance of the research question
            - Appropriateness of the study design and methodology
            - Quality and clarity of data presentation
            - Strength of the conclusions relative to the evidence
            - Clinical or scientific relevance
            List each strength as a numbered point with a brief explanation.
        """),
    },
    "weaknesses": {
        "title": "Weaknesses / Major Concerns",
        "prompt": textwrap.dedent("""\
            Identify the major weaknesses and concerns with this manuscript. Consider:
            - Methodological limitations or flaws
            - Statistical analysis issues
            - Missing controls or comparisons
            - Overstated or unsupported conclusions
            - Gaps in the literature review
            - Ethical concerns
            - Sample size and power considerations
            List each weakness as a numbered point with a specific, constructive explanation.
        """),
    },
    "minor": {
        "title": "Minor Comments",
        "prompt": textwrap.dedent("""\
            List minor comments for improving this manuscript. Consider:
            - Writing clarity and grammar
            - Figure and table quality and labeling
            - Reference completeness and accuracy
            - Formatting issues
            - Suggestions for additional analyses that would strengthen (but are not essential to) the paper
            Be specific -- reference sections, paragraphs, or figures where possible.
        """),
    },
    "statistics": {
        "title": "Statistical Review",
        "prompt": textwrap.dedent("""\
            Evaluate the statistical methods and reporting in this manuscript:
            - Are the statistical tests appropriate for the data type and study design?
            - Is the sample size adequate or is a power analysis reported?
            - Are effect sizes and confidence intervals reported (not just p-values)?
            - Are multiple comparisons accounted for?
            - Is there potential for bias (selection, information, confounding)?
            - Are the data presented in a way that supports reproducibility?
            Provide specific, actionable feedback.
        """),
    },
    "recommendation": {
        "title": "Overall Recommendation",
        "prompt": textwrap.dedent("""\
            Based on your analysis, provide an overall recommendation. Choose one of:
            - Accept as-is
            - Minor revision
            - Major revision
            - Reject

            Justify your recommendation in 2-3 sentences, referencing the most critical
            strengths and weaknesses you identified above.
        """),
    },
}

FULL_REVIEW_PROMPT = textwrap.dedent("""\
    You are an expert peer reviewer for a medical/scientific journal. You have deep
    expertise in research methodology, biostatistics, and scientific writing.

    Review the following manuscript thoroughly and provide a structured peer review.
    Be constructive, specific, and fair. Reference specific sections or data where possible.

    MANUSCRIPT TEXT:
    ---
    {manuscript_text}
    ---

    Provide your review for the following section:
    {section_prompt}
""")

CUSTOM_QUESTION_PROMPT = textwrap.dedent("""\
    You are an expert peer reviewer for a medical/scientific journal.
    You have been provided with the following manuscript text.

    MANUSCRIPT TEXT:
    ---
    {manuscript_text}
    ---

    Answer the following question about this manuscript:
    {question}
""")


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


def query_ollama(prompt: str, model: str, context_length: int = 8192) -> str:
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
        resp = requests.post(OLLAMA_URL, json=payload, timeout=600)
        resp.raise_for_status()
        return resp.json().get("response", "")
    except requests.ConnectionError:
        sys.exit(
            "Cannot connect to Ollama. Make sure it's running:\n"
            "  ollama serve"
        )
    except requests.Timeout:
        sys.exit("Ollama request timed out. The manuscript may be too long for your model's context.")
    except Exception as e:
        sys.exit(f"Ollama error: {e}")


def truncate_text(text: str, max_chars: int = 24000) -> str:
    """Truncate to fit within model context. 24k chars ~ 6k tokens (safe for 8k context)."""
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return (
        text[:half]
        + "\n\n[... middle section truncated to fit context window ...]\n\n"
        + text[-half:]
    )


def run_full_review(manuscript_text: str, model: str, sections: list[str] | None = None) -> dict:
    if sections is None:
        sections = list(REVIEW_SECTIONS.keys())

    results = {}
    text = truncate_text(manuscript_text)

    for key in sections:
        if key not in REVIEW_SECTIONS:
            print(f"  [!] Unknown section '{key}', skipping.")
            continue

        section = REVIEW_SECTIONS[key]
        print(f"  Reviewing: {section['title']}...")

        prompt = FULL_REVIEW_PROMPT.format(
            manuscript_text=text,
            section_prompt=section["prompt"],
        )
        response = query_ollama(prompt, model)
        results[key] = {"title": section["title"], "content": response}
        print(f"  Done: {section['title']}")

    return results


def format_review(results: dict) -> str:
    lines = ["=" * 70, "PEER REVIEW REPORT", "=" * 70, ""]
    for key, data in results.items():
        lines.append(f"## {data['title']}")
        lines.append("-" * 40)
        lines.append(data["content"].strip())
        lines.append("")
    lines.append("=" * 70)
    lines.append("Generated by local Ollama review assistant. No data left this machine.")
    lines.append("=" * 70)
    return "\n".join(lines)


def interactive_mode(manuscript_text: str, model: str):
    text = truncate_text(manuscript_text)
    print("\nInteractive mode -- ask questions about the manuscript.")
    print("Type 'quit' or 'exit' to stop. Type 'review' for a full structured review.\n")

    while True:
        try:
            question = input("Your question> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            break
        if question.lower() == "review":
            results = run_full_review(manuscript_text, model)
            print("\n" + format_review(results))
            continue

        prompt = CUSTOM_QUESTION_PROMPT.format(
            manuscript_text=text,
            question=question,
        )
        print("  Thinking...")
        response = query_ollama(prompt, model)
        print(f"\n{response}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Local manuscript review assistant (Ollama). All data stays on your machine.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              # Full structured review
              python review_assistant.py manuscript.pdf

              # Review specific sections only
              python review_assistant.py manuscript.pdf --sections strengths weaknesses recommendation

              # Interactive Q&A mode
              python review_assistant.py manuscript.pdf --interactive

              # Use a different model
              python review_assistant.py manuscript.pdf --model llama3.1:8b

              # Save review to file
              python review_assistant.py manuscript.pdf --output review_report.txt
        """),
    )
    parser.add_argument("manuscript", nargs="?", help="Path to manuscript file (PDF, DOCX, TXT, MD, TEX)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama model (default: {DEFAULT_MODEL})")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive Q&A mode")
    parser.add_argument("--output", "-o", help="Save review to this file")
    parser.add_argument(
        "--sections", nargs="+",
        choices=list(REVIEW_SECTIONS.keys()),
        help="Review only specific sections (default: all)",
    )
    parser.add_argument("--context-length", type=int, default=8192, help="Model context window size (default: 8192)")
    parser.add_argument("--max-chars", type=int, default=24000, help="Max manuscript chars to send (default: 24000)")
    parser.add_argument("--list-models", action="store_true", help="List available Ollama models and exit")

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
        parser.error("the following arguments are required: manuscript")

    if not os.path.isfile(args.manuscript):
        sys.exit(f"File not found: {args.manuscript}")

    print(f"Reading: {args.manuscript}")
    manuscript_text = extract_text(args.manuscript)
    print(f"Extracted {len(manuscript_text):,} characters from manuscript.")

    if len(manuscript_text) < 100:
        sys.exit("Extracted text is very short. The file may be scanned images (not OCR'd text).")

    print(f"Model: {args.model}")
    print(f"All processing is LOCAL. No data leaves this machine.\n")

    if args.interactive:
        interactive_mode(manuscript_text, args.model)
    else:
        results = run_full_review(manuscript_text, args.model, args.sections)
        report = format_review(results)
        print("\n" + report)

        if args.output:
            Path(args.output).write_text(report, encoding="utf-8")
            print(f"\nReview saved to: {args.output}")


if __name__ == "__main__":
    main()
