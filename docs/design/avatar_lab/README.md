# Cortex avatar lab

An offline chat mock at `/docs/design/avatar_lab/`, with the repository as the
web root. It reuses the two real chat stylesheets without loading application
scripts. The lab sends no messages and persists no settings.
Navigation and sphere remain static examples. The composer creates local user
bubbles using text nodes: Enter sends, Shift+Enter adds a line, and IME composition
does not send prematurely. The send button also works.

## Placement and size

The duck is at the **top-left of the chat area**, beside the sidebar. The sphere
stays horizontally centered. A 230px minimum stage clears the largest duck and
its status label. On phones up to 600px wide, the duck stays upper-left and the
centered 120px sphere sits below it, in a 340px minimum stage. Messages start below
both figures. The existing shell hides the sidebar at 800px.
On these phones the document scrolls naturally, so wrapped controls cannot squeeze
the stage or make the composer overlap messages in a short viewport.

The selector offers **96 / 128 / 144 px**, defaulting to 96. Relative to the
approximate 96-cell drawing grid, these map to 2 / 2.67 / 3 device pixels per big
pixel on a 2× display. Normal image resampling is retained: the actual PNG grid
is irregular and slightly soft, so this is an approximate crispness comparison.
**Comparar tamanhos** shows all three labelled at their actual sizes, with a
common bottom edge. On phones this row scrolls horizontally rather than shrinking
the ducks. Every copy shares state, eye patches, gesture selection and timing.
CSS animation times are aligned when copies are revealed and after changes.
Searching rain shares one capped canvas clock, including newly revealed copies.
The selected dock size does not change these three sizes.

**Repouso = original** freezes every live duck and shows a stripped rest component
beside the original PNG at the selected size. Effects, clips and transforms are
removed. Closing it restores the selected state. The two comparisons can coexist.

## Untouched source and measured clips

Every duck image loads `/src/ui/static/novo/avatar/duckcortex.png` directly. Nothing edits,
re-encodes or makes a raster copy of that asset. Its SHA-256 remains
`a848bd285577cdb4eaf901aa15bc8798f4a76af6e46f3869d47a0ad5d650925e`.

`REGIONS` records native file-pixel bounds as `[x, y, width, height]`:

| Feature | Rectangle |
| --- | --- |
| Head, including antenna | 250, 24, 680, 682 |
| Raised wing | 778, 594, 256, 310 |
| Left eye on screen | 510, 390, 132, 125 |
| Right eye on screen | 732, 358, 100, 104 |
| Antenna light | 436, 46, 52, 62 |
| Visible ear disc | 270, 418, 143, 192 |

These are the measurements from task 133. `HEAD` follows the head contour while
excluding the raised wing; `WING` avoids the beak and torso. An inverse even-odd
clip removes precisely that contour from the base while its same-PNG copy moves.
Eye patches and antenna/ear effects move inside the head group. No ear or eye is
invented. Ending or cancelling a gesture resets eyes, part and gesture attributes;
no inline transform, clip, patch or racket remains from the gesture. The baseline
rest returns to the original source with the existing ambient float/light/blink.
The frozen comparison is the exact still original with all ambient effects off.

## Racket: new grid artwork from task 133

`RACKET` in `avatar.js` is a readable **15-column × 30-row** character map.
Dots are transparent. Each other cell creates one inline SVG `rect`, with width
and height **1254 / 96 = 13.0625 file pixels**, in `viewBox="0 0 1254 1254"` with
`shape-rendering="crispEdges"`. Its origin is grid cell **66, 21**. This gives a
15 × 30 CSS-pixel racket at 96px duck size. Pedro can replace the rows and palette
names without replacing or processing the duck PNG.

All colours below were sampled from the source's decoded RGBA bytes at the listed
native coordinates. Source alpha is 253/255 at each point. Overlays and racket
use the sampled RGB, as task 133's overlays did. Tests independently decode the
PNG scanlines and check every sample; no imaging dependency or raster output is
needed.

| Sample | File coordinate | RGB | Use |
| --- | --- | --- | --- |
| yellow | 460, 75 | 253, 239, 87 | Antenna; racket inner frame |
| visor | 660, 390 | 6, 14, 56 | Existing eye patches |
| red | 330, 445 | 224, 30, 74 | Ear/wing effect; racket outer frame |
| cyan | 580, 420 | 129, 249, 251 | Existing eye cues |
| outline | 445, 35 | 5, 12, 49 | Antenna's navy outline; racket outline/string grid |
| lime | 700, 820 | 196, 239, 7 | Tennis ball; racket grip alternating with yellow |
| blue | 620, 210 | 165, 197, 251 | Helmet light blue; racket string-bed cells |

The racket size, handle, origin and replaceable cell map are unchanged for task
133c. The 2.6s celebration now swings entirely on the raised red wing's side:

| Timeline | Racket and wing choreography |
| --- | --- |
| 0–20% | Racket hidden behind the body, translated −12% / +36%, rotation 0°; wing at rest |
| 28% | Hidden racket travels below the head to +8% / +36%, rotation +30° |
| 31–32% | Racket lifts to zero translation, +30°, beside the head; fades in above the body (2), under the wing (3) |
| 46% | +18° backswing, staying on the outside of the head |
| 64% | +65° forehand, outward and down |
| 78% | +40° settle |
| 90–91% | +30° hold and fade out while the wing still holds the grip |
| 94–100% | Hidden racket retraces the lower route; wing returns to rest |

Wing and racket share the shoulder origin (65.3%, 64%) and identical transforms
throughout 31–91%, including both fades. The grip therefore stays registered to
the wing's raised tip. No inward negative rotation remains. Even the invisible
entry and exit keep the racket's occupied-cell bounds clear of the entire `HEAD`
bounding box. The SVG is absent from the rendered rest picture.

## Motion and gestures

Amplitudes below are intentionally readable at 96px. Percentage translations scale
with the selected size; values in parentheses describe the 96px version.

| State | Motion |
| --- | --- |
| Em repouso | Existing 1.5px float; slow antenna pulse; 160ms blink every 4.6s |
| Pensando | Head tilt −5° over 3.4s; existing 1.1s antenna pulse |
| Buscando na web | Head turns ±5° and ±10° in Y, with ±1.5% horizontal travel (1.44px), over 3.8s |
| Lendo | Head nod 3° / 2% down (1.92px) over 3.2s; existing 0.8/1.5px eye glances |
| Escrevendo | Existing raised-wing tap 3° over 1.45s |
| Lembrando | Head tilt −4° / 1.5% up (1.44px) over 3.6s; existing 180ms closures and antenna glow |
| Ouvindo | Head tilt −4° over 3.4s; existing visible-ear pulse |
| Respondendo | Head nod 3° / 2% down over 2.6s; existing two-beat 1.5px pose bounce |
| Processando | Head scale 1–1.025 over 3s; sampled ear cue in reduced motion |

A **wave** takes 1.8s: the clipped red wing makes three back-and-forth swings,
−18° / +12° around its shoulder, and returns to zero. It plays once on page open
when motion is allowed, on a greeting, or with **Dar oi**. A greeting plus happiness
queues the wave first, then the racket. Explicit gestures are serial (queue capped
at eight); an explicit message/button reaction settles a simulated busy state into
Em repouso before playing. Changing state cancels any active/queued gesture.

Only in **Em repouso**, a seeded Mulberry32 generator schedules a gesture after
25–60s of available rest. The small set is **look around** (1.4s, same ±5° head
turn), **hop** (0.9s, 5% up = 4.8px), **wave** (1.8s, same three swings), and
**stretch** (1.6s, −3° pose tilt / 2% up = 1.92px). Whole-pose hop/stretch transform
the untouched source rather than drawing any new body part. The previous random
gesture is excluded before choosing the next. Ambient float pauses during gestures.
The next random wait starts after completion, with no accumulated missed gestures.
**Gesto aleatório agora** uses exactly these eligibility and non-repeat rules.
The default seed comes from the page-open time; `cortexAvatarLab.setSeed(integer)`
resets the generator and previous random selection for repeatable Node tests.

Both the OS motion preference and lab checkbox cancel all gestures, timers and
queues, including welcome, manual and message reactions. States retain sampled
static light cues and their labels. The OS preference always wins. Hiding the tab
or freezing the original comparison also cancels gestures/timers. Returning starts
a fresh wait rather than replaying missed gestures. CSS movement is paused while
the tab is hidden. Gesture duration constants also set CSS duration properties so
the cleanup timer and visible animation agree.

## Deterministic message signals

`SIGNALS` holds the PT/EN greeting, positive, emoji, negation and negative-rank
lists together. Matching folds case and accents and respects word boundaries.
A greeting must start the message, ignoring punctuation: oi / olá / ola / bom dia /
boa tarde / boa noite / e aí / eai / hey / hi / hello. Substrings such as hiato,
oito or helloWorld and a later “diga oi” do not count.

Happiness matches the documented words, repeated ha/kk laughter, or the listed
emoji. “Boa” requires an immediate exclamation mark, so boa noite/tarde and bare
boa are not positive. A preceding negation within four words in the same sentence
cancels that positive, including “não ficou ótimo”, “não está ótimo”, “nem perfeito”,
“nada de perfeito”, “not really awesome” and negated emoji. Another unnegated
positive in another sentence can still count. This conservative four-word scope is
a deliberate heuristic, not sentiment understanding. “Top 10 piores” (and “top 10
worst”) is negative via the nearby negative-rank list. No model or network is used.

## Reply status head (task 134)

The fake conversation ends with a live reply bubble. After the existing 1.8s
welcome wave, **Resposta de exemplo → Turno automático** cycles thinking,
searching, reading, writing, remembering and processing, holding each for 3.8s.
It then delivers a sample answer at 45ms per character, removes the cursor at
completion, waits 4s and repeats. All text is a local example: no weather lookup,
file reading or file writing occurs.

Choose **Fixar: …** to hold one status while its CSS animation continues.
**Chegar texto** delivers the first character immediately from any held status.
The first character removes the head and dots from the paragraph; the task 131
`reply-cursor` span follows the text until completion. Status labels contain no
emoji. The `reply-dots` span contains exactly three `i` children and has
`aria-hidden="true"`, just as `replyProgress` in the real chat creates. Both the
text node and indicator stay attached across updates of the same kind, so their
animations do not restart with every status or character.

`#reply-head` is an independent component, not a small whole duck. Its one image
loads `/src/ui/static/novo/avatar/duckcortex.png` by URL. `configureArt` supplies the same `HEAD`
polygon and sampled colours as the corner. `REGIONS.head` maps the 1254px file
behind a 28px-high head: scale 28/682, offset −250/682 × 28 horizontally and
−24/682 × 28 vertically. The contour excludes the wing and includes the antenna.
The fixed 32 × 32px inline slot, 8px text gap and 36px paragraph line height
reserve room for motion without changing the line box between status and text.
No raster is created. Theme colours come from the existing chat CSS; sampled duck
colours stay identical in dark and light themes.

| Reply state | Animation at 28px | Reduced-motion cue (held still) |
| --- | --- | --- |
| Pensando | −3° tilt / 3.4s; antenna opacity .15–.9 / 1.6s | −4° tilt, bright antenna (.9) |
| Buscando | ±.8px sideways, ±4° roll, ±16° Y turn / 3.8s | +4° roll / +14° Y turn, ear outline (.75) |
| Lendo | .9px down and +4° nod / 3.2s | 1px down / +5°, softened eyes (visor patch .45) |
| Escrevendo | Two .65px upward bobs / 1.2s | 1px up / −2°, bright ear outline (.9) |
| Lembrando | −3° tilt / 3.6s; soft 180ms eye-closure envelope; slow antenna opacity .15–.9 / 3.6s | −3°, closed eyes (sampled visor), antenna .45 |
| Processando | Scale 1–1.035 / 3s (about .98px height change) | Scale .96, antenna .25 and ear .3 |

Both `body.reduced-motion` and the OS media query consume the same six still-cue
sets directly in CSS. The OS preference always wins. Reduced motion also cancels
demo timers: select held states to compare them; **Chegar texto** shows the full
answer immediately. Hiding the tab pauses motion and all timers. Returning starts
a fresh wait at the current status (or resumes at the next character), without
catch-up. The original comparison hides/pauses the reply demo and restores it when
closed, keeping the original comparison entirely still.

The demo and all corner copies share one state value. Selecting the corner's
**Estado** while a demo is active holds the matching reply state. Selecting idle,
listening or speaking hides the demo because they have no working-reply label.
Explicit greeting/happiness reactions also stop the demo before the existing
gestures. **Sem demonstração** hides the reply and leaves the corner available
for its existing controls. Only states are synchronized; animation clocks differ
deliberately. Writing bobs the reply head because the wing cannot appear in a
head-only crop. The existing corner writing gesture remains its wing tap.

## Wiring into the real UI

Task 135 retains the server's `stage` field and treat it as the state source:
`thinking → thinking`, `web_search → searching`, `reading → reading`,
`writing → writing`, `remembering → remembering`; every unknown stage maps to
`thinking` (the explicit task 135 decision supersedes the lab draft). No server change is needed. Build the head from that mapped stage,
not by parsing the label. Continue using the server's localized `label` and detail
text, stripping just a known leading status emoji plus optional variation selector
and whitespace, for example
`label.replace(/^\s*(?:🧠|🔍|📖|✍)\uFE0F?\s*/u, '')`.
Leave labels without that prefix intact; never remove emoji inside a file name,
search query or later text. Render the cleaned label with a text node.

Mount the clipped-original head before that text node and keep task 131's existing
three-dot indicator after it. Update head state and text in place on each status
event. On the first `text` event, detach the head and replace the dots with the
existing cursor; subsequent tokens must not reinsert either status element.
Remove all transient indicators on final, error, Stop and replacement turns,
including reconciliation paths. Send the same mapped state to the corner duck
when that component is integrated. Reuse the real theme tokens, body motion toggle
and OS fallback; no `respondendo` head belongs in a streaming reply. The lab demonstrates the same lifecycle offline; task 135 wires it into production.

## Searching code rain (tasks 134b / 134c)

Only **Buscando na web** and the reply's **buscando: …** status have code rain.
Each live figure lazily prepends one decorative canvas below its PNG group: rain
is at stack level 0, the entire source/moving group at 1, with an isolated figure
stack. The duck halo is **150% wide and 125% high**, offset **−25% on both axes**:
it is horizontally centred and clipped at the feet so it ends before the status
label. Its CSS sizes at 96 / 128 / 144px are **144 × 120 / 192 × 160 / 216 × 180px**.
The dock starts 48px from the top and left, reserving the largest 36px overflow
with 12px left over. Its bottom remains inside the existing 230px stage, above
chat text. Preview figures reserve 36px top/side gutters inside the horizontal
scrollport and stop the rain before their captions. Only the canvas has an
elliptical **closest-side** radial mask: opaque through **70%**, fading to zero
at **100%** of its outer radius. The original source layers keep their own clips.

The glyph table contains **0, 1, 2, 3, 7, <, >, /, =**, each five rows of three
binary cells, painted solely as unit rectangles. No font, extra image or source
pixel is used. The duck canvas is **144 × 120 logical cells**: one cell corresponds to
1254/96 native pixels and scales from 1 CSS px at 96px to 1.5 at 144px. Nearest
scaling preserves blocks. **24 columns** have a **6-cell pitch**, a 6-cell glyph
row pitch and **eight-glyph trails** (48 cells). Speeds repeat **12 / 16 / 20 cells per second**;
the glyph selection changes twice per second. Integer positions keep the cells
on-grid. Staggered heads wrap after the full trail has left the bottom.

The head uses a **40 × 40px circular dark badge**, with **seven columns** at a
5px pitch, **five-glyph trails** and speeds
**6 / 8 / 10 CSS px per second**. Its deliberate one-CSS-pixel cells are about
twice its native head grid (~0.54px), since exact native cells would reduce each
3 × 5 glyph to an unreadable 1.6 × 2.7px speck. Its PNG is still the exact same
measured clip and 28px height, inset 6px from the badge edges. The badge is
absolutely positioned at −4px within the unchanged **32px inline slot**. Its
right edge stops 4px before the text (the slot retains its 8px right margin);
the 36px paragraph line height stays unchanged. The 5px paragraph top margin
clears the badge's 4px upward overflow below the author. The aura stays
stationary behind the moving figure. Its ground is the sampled navy outline,
RGB **5, 12, 49**, including in the light theme.

**Cor da chuva** defaults to **Verde Matrix** (`green`), with a base glyph colour of
RGB **57, 255, 112** for the first trailing glyph. **Ciano do visor** (`cyan`)
uses `SAMPLES.cyan`, RGB **129, 249, 251**. Column heads mix 65% toward white
(green heads are **186, 255, 205**). Remaining trail RGB factors are
**1 / .8 / .65 / .5 / .4 / .3 / .22**; the badge uses the first five colours.
Head alpha is **.95**; trail alpha fades linearly to one eighth for the duck,
one fifth for the badge. On the light chat ground, the duck uses dark green
**22, 119, 55** or teal **24, 116, 124**, with a .75 RGB multiplier for its
inkier column heads. The dark head badge retains the neon palette in both
themes to preserve contrast. Palette changes repaint in place.

A single 50ms timeout caps rain at **20fps**, only while visible and searching;
existing CSS motion and reply/gesture timers retain their behavior. All visible
size copies use the identical logical time and pattern. Tables, figure records,
canvas buffers and palettes are reused; drawing does not allocate nodes, arrays
or paths per frame. Hiding the tab stops the clock, with no catch-up on return.
Leaving searching clears and hides every affected canvas immediately, including
the detached reply head, closed size previews and frozen original comparison.
Re-entry reuses the same canvas. The still original never receives one.

Both the OS preference and `body.reduced-motion` (including the lab checkbox)
paint a fixed phase at faint **.26 alpha**, then stop the rain timer. A body-class
observer also handles class changes made outside the checkbox. Theme and colour
changes repaint the static cue without starting falling motion. This is a
deliberate addition to the earlier CSS-only lab, bounded to the searching aura.

## Verification limits

Tests cover local assets, the source hash, measured regions and exact colour
samples, controls, deterministic message signals, seeded timing/exclusion rules,
queue order, startup restrictions, comparison synchronization, safe composer text,
and cleanup after every gesture/state. Reply tests add the original-URL head clip,
all six animation/still-cue sets in both reduced modes, exact emoji-free labels,
task 131 markup, automatic cycling, held states, shared corner state, paused
timers, stable indicators and complete status removal on first/final text. The
fake DOM runs the actual lab script under Node, without a browser. The racket
collision regression reads the
actual SVG rect map, `HEAD` contour bounds, CSS transform origins and keyframes.
It models percentage translation, rotation and CSS ease-in-out interpolation at
every 1% of each transform segment, plus exact 20/45/65% frames and endpoints.
It checks even hidden frames and confirms matching wing/racket transforms during
the visible interval. The test fails at 45% with the pre-133c inward backswing.
This bounding-box check is deliberately stricter than contour intersection.
The task explicitly forbids servers,
browsers, desktop control and network, so visual inspection was not run. The
reviewer should judge contour seams, grip alignment, size preference and reply-head
legibility/alignment at 28px in both themes in a browser; these remain the visual
uncertainties.
Task 134b adds a fake canvas recorder for glyph rectangles, lazy creation/reuse,
search-only activation and full clearing, stacking rules, both palettes and their
theme variants, capped shared timing, comparison synchronization, hidden-tab pause
and static patterns for OS preference, checkbox and direct body-class changes.
The existing asset and hash checks still protect the original PNG and offline
scope. Rain mask softness and tiny-head legibility remain visually unverified
under the same no-browser constraint.
Task 134c adds a measurable visibility regression under `node -e`: record every
unit `fillRect` over 40 × 50ms frames, map it through the actual CSS placement and
radial mask, and discard anything inside the full figure rectangle or below
**.15 effective alpha**. The intended outer ring excludes the last 10% of the
ellipse radius, where deliberate edge fading dominates. Require **≥25%** of its
cells lit over two seconds and **≥5%** in every animated frame, including entry.
These bounds prevent both a hidden effect and an occasional single-glyph flash.
The pre-134c implementation fails both figures at **0%** coverage under this
conservative opaque-box model. The new duck reaches **30.5%** (minimum frame
**5.3%**) at all three sizes; the badge reaches **38.3%** (minimum **9.3%**).
Both colours and themes are checked. A separate geometry regression reads the
CSS bounds, dock stage, preview gutters, inline slot and paragraph spacing to
check label, caption, neighbouring figure and chat-text clearance, including the
entire masked canvas envelope. These arithmetic checks do not replace a browser
review of mask softness, PNG motion or perceived glyph contrast.

## Production integration (task 135)

The sole original PNG now lives in the app static directory. The lab uses the
repository-root URL documented above; production uses the app's
`/ui/static/novo/avatar/duckcortex.png` URL. The move preserves its bytes and hash.
No derived raster or second source file is created.

Production `src/ui/static/novo/avatar.js` retains the measured regions, contour,
samples, replaceable racket map and gesture/rain implementation. It owns the
production `SIGNALS` lists and detector. The isolated lab deliberately keeps its
approved copy so it runs without application code or effects; tests compare the
two lists and exercise the same positive/negative cases. There are no production
lab controls or fake replies, and rain is fixed to Verde Matrix.

The decorative corner is a child of the real stage and appears only in chat.
It is 144px above 800px and 96px at or below 800px. Above 1100px it sits 48px from
the stage's top and left, with a 228px stage minimum. At 800px it sits at
24px left / 32px top with a 196px stage minimum. Compact desktops (801–1100px)
reserve a separate upper-left row (216px top padding, 432px minimum stage).
Phones up to 512px, including 375px, reserve a smaller row (156px top padding,
340px minimum stage). The sphere retains its existing responsive dimensions and
horizontal centre; only these narrow stage rows move it down to clear the duck
and its entire rain envelope. Stage space scrolls with the conversation and the
composer stays in its existing footer flow. Home/welcome layout is untouched.

`core.js.setLive(next, avatarState = next)` forwards listening and other live
states. SSE passes the distinct duck state in the second argument, retaining
remembering while the sphere thinks. `core.js.preview` forwards the selected
preview and restores the duck's independent live state on close. Corner state
changes are idempotent so repeated tokens/statuses do not restart animations.

Only an accepted, non-card composer submission calls `onSent`; loaded history,
regeneration, card decisions and replies never run message detection. As in the
lab, explicit reactions settle the duck into rest, serialize wave before racket,
and are cancelled by the next actual state change. A real SSE stage therefore
takes priority over a sent-text reaction; reactions are never delayed until after
the answer. The first appearance of chat triggers one welcome wave per page
load, subject to motion/visibility preferences, and conversation switches do not
replay it. A body-class observer tracks chat visibility, theme and the shell's
motion setting; the OS preference always wins. Leaving chat or hiding the tab
cancels gestures and rest timers without accumulating missed activity.

Each live reply owns one clipped 28px head, registered with the shared rain
clock. Status updates reuse its node; first token and every exit dispose its
canvas record and remove the head immediately. Final paragraphs/history have
no transient progress class or indicators. Tests run the actual avatar, core and
conversation modules together under Node; no server or browser was used during
implementation. A reviewer still needs to inspect seams, rain contrast and
perceived spacing in the real browser, especially 375px and compact desktops.
