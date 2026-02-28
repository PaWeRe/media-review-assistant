# Peer Review Assistant

Local peer-review assistant using [Ollama](https://ollama.com). All data stays on your machine.

| Journal | Review type | Script |
|---------|-------------|--------|
| **TMI** (IEEE Trans. Medical Imaging) | 1st-round | `tmi_review_assistant.py` |
| **MEDIA** | 2nd-round revision | `media_review_assistant.py` |

## Setup

```bash
git clone https://github.com/PaWeRe/peer-review-assistant.git
cd peer-review-assistant
uv venv && uv pip install -r requirements.txt
ollama serve
ollama pull qwen3:30b-a3b
```

## Folder Structure

Put manuscripts in the corresponding folder:

```
tmi/          # TMI manuscripts
media/        # MEDIA drafts, comments, and revisions
```

## TMI Review (1st Round)

```bash
uv run python tmi_review_assistant.py \
  --manuscript tmi/paper.pdf \
  --model qwen3:30b-a3b --context-length 65536 \
  --output tmi/review.md --full-report
```

## MEDIA Review (2nd Round)

```bash
uv run python media_review_assistant.py \
  --draft1 media/first_draft.pdf \
  --comments media/reviewer_comments.txt \
  --draft2 media/revised_draft.pdf \
  --model qwen3:30b-a3b --context-length 65536 \
  --output media/review.md --full-report
```

## Models

| RAM | Model | Command |
|-----|-------|---------|
| 16 GB | Gemma 3 12B | `ollama pull gemma3:12b` |
| 32 GB | Qwen3 30B-A3B (recommended) | `ollama pull qwen3:30b-a3b` |

On 16 GB, use `--model gemma3:12b --context-length 32768` instead.

## General-Purpose Reviewer

```bash
uv run python review_assistant.py manuscript.pdf --model qwen3:30b-a3b
```

See `uv run python review_assistant.py --help` for options including interactive mode.
