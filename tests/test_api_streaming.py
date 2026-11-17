"""
_sse_serialise — turns process_stream tuples into SSE lines.

Regression target: a FINAL event that carries reply text (reflex,
Spotify, tool calls — anything non-streamable) must reach the voice
listener as a `chunk`, not be dropped on the floor. The listener only
speaks `chunk` payloads, so dropping it = the assistant goes silent.

s62 (C8) contract additions:
  - a `{"language": ...}` prelude precedes the first text frame, so the
    listener picks the TTS voice the server decided (M5-A4/M10-A3);
  - an INTERNAL_ERROR final becomes a structured error frame with a
    speakable localised `reply` — never a chunk mixed into real content
    (M5-A3/M10-A5);
  - the done frame repeats the language.
"""
import json

from src.api import _sse_serialise


def _parse(lines):
    """Strip the 'data: ' prefix and JSON-decode each emitted line."""
    out = []
    for line in lines:
        assert line.startswith("data: ")
        out.append(json.loads(line[len("data: "):].strip()))
    return out


class TestReflexAndNonStreamable:
    def test_final_with_text_emits_prelude_chunk_then_done(self):
        # reflex: (reply, True, "REFLEX", lang)
        events = [("São 20:01.", True, "REFLEX", "pt")]
        msgs = _parse(_sse_serialise(events))
        # language first (so the voice is known before any words), then the
        # text (so the listener speaks it), then the done marker
        assert msgs[0] == {"language": "pt"}
        assert msgs[1] == {"chunk": "São 20:01."}
        assert msgs[2] == {"done": True, "status_mode": "REFLEX",
                           "language": "pt"}

    def test_spotify_reply_is_spoken(self):
        events = [("Tocando Matuê.", True, "SPOTIFY_PLAY", "pt")]
        msgs = _parse(_sse_serialise(events))
        assert msgs[0] == {"language": "pt"}
        assert msgs[1] == {"chunk": "Tocando Matuê."}
        assert msgs[2]["status_mode"] == "SPOTIFY_PLAY"

    def test_mode_defaults_to_offline(self):
        events = [("hi", True, None, "en")]
        msgs = _parse(_sse_serialise(events))
        assert msgs[-1] == {"done": True, "status_mode": "OFFLINE",
                            "language": "en"}

    def test_none_mode_is_logged_not_silent(self, caplog):
        # M10-A11b: the OFFLINE fallback stays (clients expect a string),
        # but a None mode is a routing path that forgot to set one — the
        # masquerade must leave a trace.
        import logging
        events = [("hi", True, None, "en")]
        with caplog.at_level(logging.WARNING, logger="cortex.api"):
            _parse(_sse_serialise(events))
        assert any("mode=None" in r.message for r in caplog.records)

    def test_real_mode_does_not_warn(self, caplog):
        import logging
        events = [("hi", True, "REFLEX", "en")]
        with caplog.at_level(logging.WARNING, logger="cortex.api"):
            _parse(_sse_serialise(events))
        assert not [r for r in caplog.records if "mode=None" in r.message]


class TestLLMStreamingPath:
    def test_replacement_without_prior_tokens_is_still_spoken(self):
        # The engine's existing JSON sanitiser can hold the whole reply;
        # a capability correction must not turn that path into silence.
        from src.core.stream_reply import ReplyReplacement

        msgs = _parse(_sse_serialise([
            (ReplyReplacement("I can save files."), True, "OFFLINE", "en"),
        ]))
        assert msgs == [
            {"language": "en"},
            {"chunk": "I can save files."},
            {"done": True, "status_mode": "OFFLINE", "language": "en",
             "reply": "I can save files."},
        ]

    def test_prelude_chunks_then_empty_final_no_duplicate(self):
        # LLM path: real chunks, then an empty final marker.
        events = [
            ("Os ", False, None, "pt"),
            ("agentes ", False, None, "pt"),
            ("são…", False, None, "pt"),
            ("", True, "OFFLINE", "pt"),
        ]
        msgs = _parse(_sse_serialise(events))
        # prelude + three chunks + one done — the empty final must NOT add
        # a chunk, and the language must NOT be re-announced per chunk
        assert msgs == [
            {"language": "pt"},
            {"chunk": "Os "},
            {"chunk": "agentes "},
            {"chunk": "são…"},
            {"done": True, "status_mode": "OFFLINE", "language": "pt"},
        ]

    def test_empty_non_final_chunks_skipped(self):
        events = [("", False, None, "pt"), ("real", False, None, "pt"),
                  ("", True, "OFFLINE", "pt")]
        msgs = _parse(_sse_serialise(events))
        assert msgs == [
            {"language": "pt"},
            {"chunk": "real"},
            {"done": True, "status_mode": "OFFLINE", "language": "pt"},
        ]

    def test_missing_lang_emits_no_prelude(self):
        # Defensive: a tuple without a language (None) must not emit a
        # bogus {"language": null} frame — the listener falls back to its
        # own heuristics, exactly the pre-C8 behaviour.
        events = [("hi", False, None, None), ("", True, "OFFLINE", None)]
        msgs = _parse(_sse_serialise(events))
        assert all("language" not in m or m.get("done") for m in msgs[:-1])
        assert msgs[0] == {"chunk": "hi"}


class TestAwaitingApproval:
    def test_staged_prompt_text_is_spoken(self):
        # Confirmation prompts are non-streamable too: (prompt, True, mode).
        events = [("Confirma o envio? (sim/não)", True, "AWAITING_APPROVAL", "pt")]
        msgs = _parse(_sse_serialise(events))
        assert msgs[0] == {"language": "pt"}
        assert msgs[1] == {"chunk": "Confirma o envio? (sim/não)"}
        assert msgs[2]["status_mode"] == "AWAITING_APPROVAL"


class TestEngineErrorFrame:
    """C8 (M10-A5): an INTERNAL_ERROR final must NOT go out as a speakable
    chunk — it becomes a structured error frame whose `reply` carries the
    short localised sentence the orchestrator built for the TTS."""

    def test_internal_error_becomes_error_frame_not_chunk(self):
        events = [
            ("Os agentes ", False, None, "pt"),
            ("Tive um problema pra gerar a resposta. Tenta de novo.",
             True, "INTERNAL_ERROR", "pt"),
        ]
        msgs = _parse(_sse_serialise(events))
        assert msgs[0] == {"language": "pt"}
        assert msgs[1] == {"chunk": "Os agentes "}
        err = msgs[2]
        assert err["error"] == "engine_stream_failure"
        assert err["reply"].startswith("Tive um problema")
        assert err["language"] == "pt"
        # The error text must never have been emitted as a chunk.
        chunk_texts = [m["chunk"] for m in msgs if "chunk" in m]
        assert all("problema" not in c for c in chunk_texts)
        # And the stream still terminates with a proper done frame.
        assert msgs[3] == {"done": True, "status_mode": "INTERNAL_ERROR",
                           "language": "pt"}

    def test_error_first_frame_still_carries_prelude(self):
        # Engine died before any chunk: prelude, error, done.
        events = [("I hit a problem generating the reply. Please try again.",
                   True, "INTERNAL_ERROR", "en")]
        msgs = _parse(_sse_serialise(events))
        assert msgs[0] == {"language": "en"}
        assert "error" in msgs[1] and msgs[1]["reply"]
        assert msgs[2]["status_mode"] == "INTERNAL_ERROR"
