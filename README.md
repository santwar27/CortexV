# Cortex — a local-first personal AI assistant

Cortex is a voice- and text-driven AI assistant that runs **entirely on local
hardware** — no cloud model, no personal data leaving the machine. It takes a
natural-language request (typed or spoken), routes it to the right tool or
integration, remembers context across conversations, and asks for confirmation
before doing anything irreversible.

I build it as my main engineering project: a place to practise system design,
clean code, and disciplined testing on something real that I actually use every
day.

> **Status:** personal project, actively developed. Built and maintained solo.
> This repository is a curated public snapshot of the private working repo.

---

## What it does

- **Understands and routes requests.** A lightweight orchestrator decides whether
  a request is a quick reflex (e.g. "what time is it?"), a memory lookup, a tool
  call, or a full language-model answer — so the fast paths stay fast and never
  pay for a model call.
- **Integrates with real tools.** Gmail, Google Calendar, Apple Notes and
  Reminders, Spotify, macOS controls (apps, volume, brightness, dark mode, sleep),
  web search, and a job-search engine — all behind a FastAPI backend with SQLite
  data stores.
- **Works with documents.** Reads PDFs (falling back to Poppler when the Python
  parser loses or glues words), merges and splits them, renders pages as images,
  and extracts embedded images. Poppler runs inside a macOS sandbox profile with
  time and size limits.
- **Helps with code, under supervision.** It can explain a file, propose a change,
  and apply it only after the user types an exact "sim" — keeping an exact copy of
  the original and a time-limited "undo". Syntax checks never execute the code,
  and test runs happen on a disposable copy inside a sandbox.
- **Remembers.** A retrieval-augmented memory ("Palace") stores facts and past
  context using local embeddings, with time decay so recent things weigh more.
  Conversations can be grouped into projects.
- **Talks.** A fully on-device voice pipeline: wake-word detection ("Hey
  Cortex"), speech-to-text, and text-to-speech. Audio never leaves the machine.
- **Stays honest.** Dates, weekdays and relative times ("in 2 hours") are computed
  by code, not guessed by the model; questions about Cortex itself (which model,
  which context window) are answered from code-owned facts; and guards catch
  replies that claim an action happened when no tool ran.
- **Stays safe.** A permission layer gates irreversible actions (sending an
  email, creating an event, quitting an app) behind explicit, human-readable
  rules — a model is never allowed to auto-approve them on its own. An approved
  action's intent is recorded before it runs, and an uncertain outcome is never
  retried blindly.

## Architecture at a glance

```
            speak / type
                 │
                 ▼
        ┌──────────────────┐      reflex ──────────────► instant answer
        │   Orchestrator   │      memory lookup ───────► Palace (local)
        │  (routing layer) │      tool call ───────────► integrations
        └──────────────────┘      language model ──────► local LLM (Ollama)
                 │
                 ▼
        ┌──────────────────┐   irreversible?  ┌──────────────────┐
        │   Gate / policy  │ ───────────────► │ human-in-the-loop │
        └──────────────────┘   (S3 actions)   └──────────────────┘
                 │
                 ▼
     reply checks → voice / UI reply
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full picture, and
[PRINCIPLES.md](PRINCIPLES.md) for the privacy and safety lines the design holds.

## Tech stack

- **Language:** Python
- **Local LLM:** [Ollama](https://ollama.com) running **Gemma 4 12B** (QAT build,
  8,192-token window) on an Apple Silicon laptop. It replaced Llama 3.1 8B in
  November 2026 after scoring 34/37 against the 8B's 30/37 on the project's
  real-model evals.
- **Backend:** FastAPI + Uvicorn
- **Storage / memory:** SQLite (with FTS5) + local `nomic-embed-text` embeddings
- **Documents:** pypdf + Poppler (sandboxed)
- **Voice:** openWakeWord (custom "Hey Cortex" model) · faster-whisper (STT) ·
  Piper / system TTS
- **UI:** a web UI in plain JavaScript and CSS (no framework), for desktop and
  phone, with an animated avatar
- **Tooling:** Make, pytest

## Testing

Cortex is test-driven. Two layers, both offline and deterministic:

- **2,553 tests, 27,737 cases with their variations** (pytest `parametrize`)
  covering the plumbing, routing, guards and integrations. Many of the cases come
  from checking hundreds of natural ways of saying the same command (phrasing ×
  punctuation × streaming) and hundreds of model replies the guards must catch.
  27,708 cases pass offline; 29 are opt-in checks that need the real macOS
  sandbox and Poppler installed. Every adapter has an injectable transport, so
  the suite never reaches the network or a real model.
- **397 behavioural evaluation cases** (`evals/`) that assert how the assistant
  *should* respond — routing, refusals, anti-hallucination, internationalisation
  (PT-BR / UK English) and flow control.

All numbers come from this snapshot, all green. A separate real-model eval set
(`make eval-llm`, needs Ollama) is what decided the model swap above.

```bash
make test    # full pytest suite
make eval    # deterministic behavioural evals (offline)
make check   # both — the gate the git hooks enforce
```

## Running it

Cortex targets Apple Silicon (macOS) with [Ollama](https://ollama.com) installed.

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then fill in your own keys
ollama pull gemma4:12b-it-qat
ollama create cortex-lite -f Modelfile
ollama create cortex-criativo -f Modelfile-creative
make check                    # verify the suite is green
```

`.env.example` documents every setting; nothing personal is committed — secrets
and local data stay out of the repo by design.

## License

The code is released under the [MIT licence](LICENSE), with one exception: the
custom wake-word model `src/voice/models/hey_cortex.onnx` is licensed
**CC BY-NC-SA 4.0** (non-commercial), because part of its training data carries
that licence. See
[src/voice/models/LICENSE-hey_cortex.md](src/voice/models/LICENSE-hey_cortex.md).

---

*Cortex is the flagship project in my software engineering portfolio. Feedback
and questions are welcome.*
