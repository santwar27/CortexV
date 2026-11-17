# Design principles

Cortex is local-first on purpose. These are the lines the design holds even when
a cloud shortcut would be easier — the criteria I judge every new feature
against. They double as a statement of how I think about privacy and safety in
software.

## 1. Personal memory never leaves the machine

Anything that loads the user's profile, past context, or stored knowledge is
local-only. The chat path loads personal memory on every turn, so that path
never calls out to a third-party service. Outbound calls are allowed only for
narrow, public, enumerable payloads (e.g. a web search query) — never for data
derived from personal memory.

## 2. Audio never leaves the machine

Raw audio is the most personal signal in the system — it captures the room and
everyone in it. Speech-to-text and text-to-speech are local, permanently. Better
voice quality is a *hardware* problem (a faster machine, better models on it),
never a reason to send audio to the cloud.

## 3. Irreversible actions need a human — decided by rules, not by a model

An irreversible action (sending an email, creating an event, a reminder) only
runs without a human when a **narrow, deterministic, auditable rule** has
pre-approved it. A language model never decides on its own that an irreversible
action is safe to take. No matter how capable the model gets, a model verdict is
not a security policy — it isn't enumerable or stable across versions. The
auto-approval list ships **empty** by design.

## 4. The test gate is deterministic

The commit/push gate (`make check`) uses deterministic matching — substring and
cosine checks against recorded baselines. It is cheap, stable, and never flaky.
If a model-as-judge is ever added, it stays opt-in and out of the CI path; the
gate that guards every commit stays deterministic.

---

*These principles are enforced in code and in review, not just documented. A
change that violates one is wrong by definition, even if the tests pass.*
