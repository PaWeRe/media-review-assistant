#!/usr/bin/env python3
"""
TMI peer-review assistant using Ollama vision models.

Renders PDF pages as images so the model sees figures, tables, and equations
exactly as a human reviewer would.
All processing stays on your machine -- no data is sent externally.
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

REVIEW_PROMPT = """\
You are helping a peer reviewer assess a manuscript submitted to \
IEEE Transactions on Medical Imaging (TMI). The complete paper is provided \
as page images.

Produce a structured analysis covering the sections below. Be concrete: \
cite page numbers, figure/table numbers, equation numbers, and quote or \
closely paraphrase specific author claims. Avoid generic praise or filler. \
Do NOT produce a recommendation or rating -- the reviewer will form their \
own judgment.

## 1. Paper Digest

5-6 sentences: problem addressed, proposed method (core idea, not a laundry \
list of components), datasets used, headline quantitative results. Then for \
each key figure and table, one sentence on what it shows.

## 2. Novelty Audit

For EACH novelty claim the authors make:
- Quote or paraphrase the claim with section/page reference.
- Identify the closest prior work and how the authors differentiate.
- Assess whether the differentiation is substantiated or superficial.

Flag anything that appears incremental, overstated, or already addressed \
by existing work the authors may not have cited.

## 3. Literature Gaps

List specific missing references or research directions that would \
materially change how the contribution is positioned. For each entry:
- Give enough detail to verify (first author, short title, venue, \
  approximate year).
- Explain WHY it matters for this paper specifically.
- Mark confidence: [HIGH] if you are certain the work exists, \
  [MEDIUM] if likely, [LOW] if uncertain.

Do NOT fabricate full bibliographic details. The reviewer will verify \
each suggestion independently.

## 4. Experimental Scrutiny

Address each of these explicitly:
- Are baselines current and fairly compared (same backbone, training data, \
  hyperparameter budget)?
- Are the datasets standard and representative for the claimed scope? \
  Would the community consider them sufficient?
- Do effect sizes support the claims, or are improvements within noise / \
  margin of error? Look at absolute numbers, not just "outperforms."
- Are ablations sufficient to isolate each claimed contribution?
- What analysis is missing (statistical significance tests, failure cases, \
  computational cost, cross-dataset generalization)?

## 5. Figures & Tables

For each key figure and table: what does it convey, is it effective, and \
is anything misleading or redundant? Identify which are critical evidence \
and which add little. Flag any figure where the visual evidence contradicts \
or does not clearly support the text's claims.

## 6. Specific Issues

Numbered list of concrete, actionable problems the authors should address. \
Reference specific sections, figures, equations, or tables. No generic \
advice (e.g., not "improve writing quality" but "Section III-B conflates \
X with Y, making the loss formulation unclear").
"""


# ---------------------------------------------------------------------------
# PDF rendering
# ---------------------------------------------------------------------------

def render_pdf_pages(filepath: str, dpi: int = DEFAULT_DPI) -> list[str]:
    """Render each PDF page to a base64-encoded PNG."""
    import fitz

    doc = fitz.open(filepath)
    pages = []
    for page in doc:
        pix = page.get_pixmap(dpi=dpi)
        pages.append(base64.b64encode(pix.tobytes("png")).decode("ascii"))
    doc.close()
    return pages


# ---------------------------------------------------------------------------
# Ollama vision interface
# ---------------------------------------------------------------------------

def query_ollama_vision(
    prompt: str,
    images: list[str],
    model: str,
    stream: bool = True,
) -> str:
    """Send prompt + page images to a local Ollama VL model."""
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt, "images": images},
        ],
        "stream": stream,
        "options": {
            "temperature": 0.3,
        },
    }
    try:
        resp = requests.post(
            OLLAMA_CHAT_URL, json=payload, timeout=3600, stream=stream,
        )
        resp.raise_for_status()
    except requests.ConnectionError:
        sys.exit(
            "Cannot connect to Ollama. Make sure it's running:\n"
            "  ollama serve"
        )
    except requests.Timeout:
        sys.exit("Ollama request timed out (60 min limit).")
    except Exception as e:
        sys.exit(f"Ollama error: {e}")

    if not stream:
        return resp.json().get("message", {}).get("content", "")

    chunks = []
    for line in resp.iter_lines():
        if not line:
            continue
        data = json.loads(line)
        token = data.get("message", {}).get("content", "")
        if token:
            _print(token, end="", flush=True)
            chunks.append(token)
        if data.get("done"):
            break
    _print()
    return "".join(chunks)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "TMI review assistant — local Ollama VL model, privacy-preserving. "
            "Sends full PDF pages as images so figures/tables are visible."
        ),
    )
    parser.add_argument(
        "--manuscript", "-m", required=True,
        help="Path to the manuscript PDF",
    )
    parser.add_argument(
        "--output", "-o",
        help="Save analysis to this markdown file",
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
                _print(f"  {m['name']:<30s} {size_gb:.1f} GB")
        except Exception:
            _print("Cannot connect to Ollama. Is it running?")
        return

    if not os.path.isfile(args.manuscript):
        sys.exit(f"File not found: {args.manuscript}")

    ext = Path(args.manuscript).suffix.lower()
    if ext != ".pdf":
        sys.exit(f"Only PDF files are supported (got {ext}).")

    print(f"Rendering {args.manuscript} at {args.dpi} DPI ...")
    pages = render_pdf_pages(args.manuscript, dpi=args.dpi)
    raw_mb = sum(len(p) for p in pages) * 3 / 4 / 1e6
    print(f"  {len(pages)} pages (~{raw_mb:.1f} MB image data)")

    print(f"Model: {args.model}")
    print("All processing is LOCAL. No data leaves this machine.\n")

    t0 = time.time()
    result = query_ollama_vision(
        REVIEW_PROMPT, pages, args.model, stream=not args.no_stream,
    )
    elapsed = time.time() - t0
    print(f"\n[Done in {elapsed:.0f}s]")

    if args.output:
        out = Path(args.output)
        out.write_text(result, encoding="utf-8")
        print(f"Saved to: {out}")


if __name__ == "__main__":
    main()
