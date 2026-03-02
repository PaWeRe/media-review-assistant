# Peer Review Assistant

Local, privacy-preserving peer-review assistant powered by [Ollama](https://ollama.com). No data leaves your machine.

## Setup

```bash
uv venv && uv pip install -r requirements.txt
ollama serve
ollama pull qwen3-vl:30b
```

## TMI Review (1st Round, Vision)

2-pass pipeline: PDF pages are rendered as images so the model sees figures, tables, and equations directly.

```bash
uv run python tmi_review_assistant.py \
  -m tmi/paper.pdf \
  -o tmi/review.md --full-report
```

With optional reviewer notes to guide the analysis:

```bash
uv run python tmi_review_assistant.py \
  -m tmi/paper.pdf \
  -n tmi/notes.txt \
  -o tmi/review.md --full-report
```

## MEDIA Review (2nd Round)

3-pass text pipeline comparing original draft, reviewer comments, and revised draft.

```bash
uv run python media_review_assistant.py \
  --draft1 media/first_draft.pdf \
  --comments media/reviewer_comments.txt \
  --draft2 media/revised_draft.pdf \
  -o media/review.md --full-report
```

## General-Purpose Reviewer

```bash
uv run python review_assistant.py manuscript.pdf
```

## Options

All scripts support `--help`. Common flags:

| Flag | Description |
|------|-------------|
| `--model` | Ollama model name (default varies per script) |
| `--context-length` | Context window size in tokens |
| `--dpi` | PDF render resolution (TMI script only, default 200) |
| `--notes`, `-n` | Reviewer notes file to guide analysis (TMI script only) |
| `--full-report` | Save intermediate analysis alongside final review |
| `--no-stream` | Wait for full response instead of streaming |
| `--list-models` | List available Ollama models and exit |
