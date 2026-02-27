# Peer Review Assistant

Fully local peer-review assistant for scientific journals using
[Ollama](https://ollama.com). Journal-specific scripts handle different review
workflows while keeping **all data on your machine**.

Currently supports:
- **TMI** (IEEE Transactions on Medical Imaging) -- 1st-round reviews
- **MEDIA** -- 2nd-round revision reviews

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/PaWeRe/peer-review-assistant.git
cd peer-review-assistant

# 2. Install Python dependencies (requires Python 3.10+)
pip install -r requirements.txt      # or: uv pip install -r requirements.txt

# 3. Make sure Ollama is running
ollama serve

# 4. Pull a model (see model table below)
ollama pull gemma3:12b                # 16 GB machine
ollama pull qwen3:30b-a3b             # 32 GB machine (recommended)
```

## TMI Review (1st Round)

Single-manuscript first-round review with evaluation matrix, detailed comments,
and recommendation.

```bash
python tmi_review_assistant.py \
  --manuscript paper.pdf \
  --output review.md --full-report
```

### Output Format

1. **Evaluation Matrix** (Excellent / Very Good / Good / Fair / Poor):

   | Category | Rating |
   |----------|--------|
   | Scientific Merit | ... |
   | Originality | ... |
   | Impact | ... |
   | Practical Value | ... |
   | Reader Interest | ... |
   | Overall Evaluation | ... |

2. **Comments to the Author** -- paragraphs covering significance & innovation,
   study design, content & results, writing & presentation, specific concerns,
   and additional references (full bibliographic citations with DOIs).

3. **Recommendation**: Accept / Accept pending minor revision / Reject-Resubmit /
   Reject (submit elsewhere) / Reject

### TMI Options

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--manuscript` | `-m` | (required) | Manuscript file |
| `--output` | `-o` | stdout | Save review to file |
| `--full-report` | | off | Also save the detailed analysis pass |
| `--model` | | `gemma3:12b` | Ollama model name |
| `--context-length` | | `32768` | Context window (tokens) |

## MEDIA Review (2nd Round)

Three-document revision review: reads the 1st draft, first-round reviewer
comments, and the revised 2nd draft.

```bash
python media_review_assistant.py \
  --draft1 first_draft.pdf \
  --comments reviewer_comments.txt \
  --draft2 revised_draft.pdf \
  --output review.md --full-report
```

### Output Format

1. **Overall Manuscript Rating** (1--100)
2. **Recommendation**: Accept (no revision) / Accept (minor, no 2nd review) /
   Accept (minor, 2nd review) / Reject-Major revision / Reject
3. **Reviewer Comments to Author** -- summary, revision assessment, content,
   methodology, presentation, remaining concerns, final remarks

### MEDIA Options

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--draft1` | `-d1` | (required) | 1st (original) draft |
| `--comments` | `-c` | (required) | 1st-round reviewer comments |
| `--draft2` | `-d2` | (required) | 2nd (revised) draft |
| `--output` | `-o` | stdout | Save review to file |
| `--full-report` | | off | Also save intermediate pass outputs |
| `--model` | | `gemma3:12b` | Ollama model name |
| `--context-length` | | `32768` | Context window (tokens) |

## Recommended Models

| Machine | Model | Pull command | Context | Notes |
|---------|-------|-------------|---------|-------|
| M1 Pro 16 GB | Gemma 3 12B | `ollama pull gemma3:12b` | 32K | Best reasoning at this RAM level |
| M1 Pro 16 GB | Gemma 3 4B | `ollama pull gemma3:4b` | 32K | Faster, lighter |
| 32 GB machine | Qwen3 30B-A3B | `ollama pull qwen3:30b-a3b` | 65K | MoE: 30B params, 3B active -- fast + smart |
| 32 GB machine | Gemma 3 27B | `ollama pull gemma3:27b` | 65K | Dense 27B, slower but strong |

### 32 GB Example

```bash
python tmi_review_assistant.py \
  --manuscript paper.pdf \
  --model qwen3:30b-a3b --context-length 65536 \
  --output review.md --full-report
```

### Performance Tips

- **Close other heavy apps** (Cursor, VS Code, browsers) before running.
  Ollama needs free RAM for the model + KV cache.
- On 16 GB, expect ~2-5 min per pass.
- On 32 GB with `qwen3:30b-a3b`, expect ~1-2 min per pass.

## How It Works

### TMI (2-pass pipeline)

1. **Pass 1 -- Deep Analysis**: Evaluates scientific merit, originality, impact,
   design, writing, and missing references.
2. **Pass 2 -- Formal Review**: Produces the evaluation matrix, structured
   comments with references, and recommendation.

### MEDIA (3-pass pipeline)

1. **Pass 1 -- Revision Expectations**: Extracts a structured checklist of every
   reviewer concern from the 1st round.
2. **Pass 2 -- Fulfillment Check**: Checks the revised manuscript against each
   concern (Fully / Partially / Not Addressed).
3. **Pass 3 -- Final Review**: Generates the formal MEDIA review with rating,
   recommendation, and comments.

## Supported File Formats

- PDF (text-based; scanned PDFs without OCR are not supported)
- DOCX
- TXT, Markdown, LaTeX

## Also Included

`review_assistant.py` -- a general-purpose single-manuscript reviewer
(not journal-specific). See `python review_assistant.py --help`.

## Privacy

- All LLM inference runs locally via Ollama (`localhost:11434`)
- No internet connection required after model download
- Safe for confidential manuscript review
- `.gitignore` excludes PDFs, DOCX, and review outputs
