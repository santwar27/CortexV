# Architecture

Cortex is organised around a single idea: **decide the cheapest correct way to
answer, then do it locally.** A second idea follows from running a small model
on a laptop: **anything code can know for certain, code decides** — the model is
only asked for what genuinely needs language. This document walks the request
lifecycle and the main components.

## Request lifecycle

1. **Input** arrives as text (HTTP/UI/terminal) or speech (voice pipeline →
   transcript).
2. **The orchestrator** ([`src/core/orchestrator.py`](../src/core/orchestrator.py))
   classifies the request and picks a path, cheapest first:
   - **Reflex** — deterministic answers that never need a model (time, date,
     weekday, simple system status). No LLM.
   - **Dispatch route** — a phrase the router recognises ("junta a.pdf e b.pdf",
     "volume 30", "me lembra daqui a 2 horas…") goes straight to its handler.
   - **Memory lookup** — pull relevant facts/context from the Palace.
   - **Language model** — fall back to the local LLM for open-ended reasoning,
     with memory, tool results and code-built fact notes folded into the prompt.
3. **The gate** ([`src/core/gate_policy.py`](../src/core/gate_policy.py)) sits in
   front of any irreversible ("S3") action. Unless a narrow, explicit rule
   pre-approves it, the action is staged and waits for the user's confirmation.
4. **Reply checks** run before anything is shown or spoken: a reply that claims an
   action no tool performed, or misstates a fact Cortex owns about itself, is
   replaced by an honest one.
5. **Output** is rendered back to the UI or spoken via TTS.

## Components

### Engine — `src/core/engine.py`, `src/core/prompts.py`
Thin wrapper over [Ollama](https://ollama.com), today running **Gemma 4 12B**
(`gemma4:12b-it-qat`, see [`Modelfile`](../Modelfile)). Owns the versioned system
prompts, model selection (an analytical model and a creative one), the 8,192-token
window shared by every turn, and the chat/stream/tool-call call sites. The
personality block is used only on calls without tools: measured with tools in
the prompt, it made the model promise actions instead of calling them. Personal
memory is loaded here per turn, which is exactly why this path is local-only.

### Orchestrator and dispatch — `src/core/orchestrator.py`, `src/core/dispatch/`
The routing brain. A registry of ordered routes (regex families per domain:
calendar, macOS, Spotify, notes, reminders, studies, jobs…) maps recognised
phrases to handlers mixed into the orchestrator by family. It also runs the reflex
layer, multi-step flows and language mirroring (PT-BR ↔ UK English).

### Deterministic helpers — `src/core/clock.py`, `src/core/self_facts.py`, …
Small, pure modules that keep guesswork away from the model: the local clock
(dates, weekdays, "daqui a N horas" computed by code), code-owned facts about
Cortex itself (model, window, version), exact token usage of the previous answer,
and a currency policy for price answers from web search.

### Guards — `src/core/hallucination.py`, `src/core/self_answer_check.py`
Narrow, deterministic detectors for the failure modes measured on real models:
announcing an action no tool ran, writing a tool call as plain text, inventing
a "command failed" message, or contradicting a code-owned fact. Each guard has a fixed honest fallback.

### Palace (memory) — `src/memory/`
The long-term store: SQLite with full-text search (FTS5) plus semantic search
over local `nomic-embed-text` embeddings. Memory has **time decay** — episodic
chatter fades fast, durable facts persist — and a consolidation step that keeps
the store tidy. Nothing here ever leaves the machine.

### Integrations — `src/integrations/`
Adapters for Gmail, Google Calendar, Apple Notes and Reminders, Spotify, macOS
automation, contacts, and the job-search engine (public ATS / apprenticeship APIs
+ local CV ranking by cosine similarity). Each adapter has an injectable
transport so tests run fully offline.

### Documents and code under containment — `src/core/pdf_*.py`, `src/core/code_*.py`
PDF work uses pypdf first and Poppler when the text is lost or glued; Poppler
runs under a macOS Seatbelt (`sandbox-exec`) profile with time and output
limits, and results are published atomically without ever replacing an existing
file. Code changes follow a fixed cycle — explain, propose, apply on an exact
"sim", undo within a time window — and test runs happen on a disposable copy
of the project inside the same kind of sandbox.

### Staging and effects — `src/core/staging_mixin.py`, `src/core/effect_runner.py`
Every confirmable action is staged with a human-readable preview. When approved,
its intent is recorded before it runs; an outcome that cannot be confirmed is
reported as uncertain and never retried automatically.

### Gate & policy — `src/core/gate_policy.py`, `src/core/gate_queue.py`
The safety layer. Risk levels classify each action; irreversible ones require a
human unless a deterministic, auditable rule says otherwise. A background queue
lets actions wait safely for approval.

### Voice — `src/voice/`
On-device pipeline: wake-word detection (a custom "Hey Cortex" model),
speech-to-text (faster-whisper), and text-to-speech (Piper or the system voice).
Audio is captured, processed, and discarded locally.

### API & UI — `src/api.py`, `src/ui/`
FastAPI backend (auth-gated) and a web UI in plain JavaScript and CSS:
conversations grouped into projects, attachments, pending approvals, a
microphone button, and an animated avatar. It works on desktop and phone.

## Testing & evals — `tests/`, `evals/`
Two layers, both offline. **Unit/integration tests** (pytest: 2,553 tests,
27,737 cases with their parametrized variations in this snapshot) cover the plumbing. **Behavioural evals** (`evals/`, 397 cases) assert
*how Cortex should respond* across real scenarios — routing, refusals,
anti-hallucination, i18n, flow control — without flaky live calls. `make check`
runs both and is the same gate the git hooks enforce. A separate real-model set
(`make eval-llm`) measures the model itself and is how model and prompt changes
are decided.

## Design constraints

- **Local-first** — the assistant works without the cloud; see
  [PRINCIPLES.md](../PRINCIPLES.md).
- **Fast paths stay fast** — reflexes and routes never pay for a model call.
- **Code decides what code can know** — time, facts about itself and action
  outcomes come from code, not from the model.
- **Testable by construction** — transports and clocks are injectable, so the
  suite is deterministic and offline.
