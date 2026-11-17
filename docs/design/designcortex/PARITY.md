# Approved reference: Aranha escuro

The four source files are byte-for-byte copies; SHA-256 pins live in `tests/test_ui_parity.py`.
Never update them merely to make a UI change pass. `tests/ui_parity.py` is the importable inventory.

- **exact:** retain normalized tags, attributes, children and Portuguese text, including all
  fifteen symbols. Navigation/modes compare ordered button values, labels and symbol references,
  allowing selected state to change.
- **real:** retain the slot, placement and appearance; use the listed feed or an honest empty/unavailable
  state. No seeded projects or demo metrics. Later feed tests must prove provenance.
- **secondary:** anchors reserve containers for controls absent from the prototype; `today` records
  routes/templates and `placement` the decision. A host does not prove a control works. Background
  approval design remains a separate task. Class selectors identify slots without prototype ids.

Reviewer capture procedure (not run by this task):

1. Serve the folder locally, e.g. `python -m http.server 8765 --bind 127.0.0.1 --directory docs/design/designcortex`,
   and open `http://127.0.0.1:8765/` in a real browser.
2. Set `prefers-reduced-motion: reduce` emulation **before reloading**: the prototype checks
   `#reduce-motion`, adds `body.reduced-motion`, and canvas time stays zero. Dark is the default:
   `body` must not have `light`. Use idle state, intensity 70%, zoom 100%, identical device scale.
3. Capture all six `SCREENS` at both `VIEWPORTS`: home, chat, projects, Central,
   Memória detail, appearance × desktop 1440×900 and mobile 390×844 (12 primary captures).
   Reload before each `clicks`/`mobile_clicks` path; chat uses Retomar projeto for deterministic content.
4. Check every `visible` anchor, scrolling content/dialogs for below-fold elements.
   `sidebar` anchors are visible on desktop; on mobile take a supplementary capture
   after `.mobile-menu` opens the drawer. Dialog screenshots show only the foreground. Also test
   Retomar central/focus restoration, Escape and close without losing the conversation.
5. A later task runs `check(served_html, js_text)` (or serialized rendered DOM with
   no JS argument) and records `[]`, then supplies side-by-side reference/live captures
   for the full matrix. The parser only reads the prototype's literal JS metric table;
   it is not a JS interpreter. Prefer rendered DOM for a different implementation. Verify real feeds,
   empty states and secondary actions separately; CSS, canvas and visibility require browser review.

Open edges: “idea” awaits Pedro; profile/model/project adapters are later work. Greeting periods are
05–11 Bom dia, 12–17 Boa tarde, otherwise Boa noite; omit a missing name. Accessibility text
(`aria-label` and `title` values) is outside `exact` and must describe the real control.
The new UI must give `check(...) == []` AND `demo_leftovers(...) == []` on its rendered DOM.
Only the appearance panel (`#controls` and its `#open-appearance` option) allows preview wording;
its state control is never system truth. PNG/backend/UI changes are intentionally excluded.
