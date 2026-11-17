"""Served-script contracts for conversation-scoped, read-only effect marks."""
import re

from tests.test_ui_novo import (  # noqa: F401
    ASSETS, SECONDARY, HEADERS, client, ui_app, conversa, pendente,
)
from tests.ui_parity import Document, check, demo_leftovers


def test_marks_read_and_refresh_contract(pendente):
    effects_read = "read('/ui/effects' + (cid ? '?conversation_id=' + encodeURIComponent(cid) : ''))"
    assert effects_read in pendente
    refresh = pendente.split("async function refresh() {", 1)[1].split(
        "\ndocument.addEventListener", 1)[0]
    clear = "$('#messages').querySelectorAll('.effect-mark').forEach(element => element.remove());"
    assert refresh.index(clear) < refresh.index("read('/ui/pending'")
    # Invalidate the cache too: a save event during/after a failed read must not restore it.
    assert refresh.index("effectRead = null;") < refresh.index(effects_read)
    assert (refresh.index("version !== sequence || cid !== chat.activeId")
            < refresh.index("effects.status === 'fulfilled'")
            < refresh.index("effectRead = {cid, data: effects.value};")
            < refresh.index("mark();"))
    assert "renderNotices(effects.value);" in refresh
    assert "document.addEventListener('cortex-turn-saved', mark);" in pendente


def test_marks_target_reply_then_preview_and_group_in_order(pendente):
    assert "function mark() {" in pendente
    mark = pendente.split("function mark() {", 1)[1].split("\nasync function read", 1)[0]
    for fragment in (
        "querySelectorAll('.effect-mark').forEach(element => element.remove())",
        "const cid = chat.activeId", "effectRead?.cid === cid ? effectRead.data : null",
        "if (!data?.available || !Array.isArray(data.conversation)) return;",
        "querySelectorAll('.message.assistant')", "new Map()",
        "for (const effect of data.conversation)",
        "effect.turn?.conversation_id === cid",
        "row.dataset.turnId === effect.turn.turn_id",
        "const fallback = !row;", "if (!row && effect.asked_turn?.conversation_id === cid)",
        "row.dataset.turnId === effect.asked_turn.turn_id", "if (!row) continue;",
        "groups.get(row).push({label: effect.label, fallback})",
        "for (const [row, effects] of groups)", "effects.map(effect => effect.label)",
        "effects.every(effect => effect.fallback)", "node('div', null, 'effect-mark')",
        "node('span', text + '.')", "row.querySelector('.message-content').append(marker)",
        "action(marker, 'Ver', () => { $('#settings').click(); $('#option-audit').click(); })",
        "'Não sei se isto aconteceu: '", "'Não sei se estas '", "' ações aconteceram: '",
        "labels.join(', ')", "' (a resposta do “sim” se perdeu)'",
    ):
        assert fragment in mark
    assert mark.index("effect.turn?.conversation_id === cid") < mark.index("const fallback = !row;")
    assert mark.index("const fallback = !row;") < mark.index("effect.asked_turn?.conversation_id === cid")
    assert "fetch(" not in mark and "read(" not in mark
    assert "[data-turn-id" not in pendente
    assert not re.search(r"querySelector(?:All)?\([^\n]*turn_id", pendente)
    assert "element.textContent = text;" in pendente


def test_saved_reply_exposes_id_and_announces_after_render(conversa):
    saved = re.split(r"^(?:async )?function ",
                     conversa.split("function saved(live, turn) {", 1)[1],
                     maxsplit=1, flags=re.M)[0]
    assert "if (typeof turn.id === 'string') live.closest('.message').dataset.turnId = turn.id;" in saved
    assert (saved.index("live.innerHTML = renderMarkdown(turn.reply);")
            < saved.index("document.dispatchEvent(new CustomEvent('cortex-turn-saved'));"))
    assert saved.rstrip().endswith("document.dispatchEvent(new CustomEvent('cortex-turn-saved'));\n}")
    assert "// Never substitute an older or voice reply." in conversa
    assert "Delivery ids differ from saved ids." not in conversa


def test_marks_are_muted_and_wrap(client):
    response = client.get(SECONDARY)
    assert response.status_code == 200
    rule = re.search(r"^\.effect-mark\s*\{([^}]*)\}", response.text, re.M)
    assert rule
    for fragment in ("var(--muted)", "font-size: 11px", "display: flex", "align-items: center",
                     "flex-wrap: wrap", "gap: 6px", "margin-top:"):
        assert fragment in rule.group(1)
    assert ".effect-mark button" in response.text


def test_marks_preserve_shell_parity_and_script_order(client):
    response = client.get('/ui/novo', headers=HEADERS)
    shell = client.get(ASSETS[1])
    assert response.status_code == shell.status_code == 200
    assert check(response.text, shell.text) == demo_leftovers(response.text) == []
    scripts = [node.attrs['src'] for node in Document(response.text).nodes if node.tag == 'script']
    assert scripts == ['/ui/static/novo/' + name + '.js' for name in (
        'avatar', 'core', 'conversa', 'quarto', 'projetos', 'central', 'pendente', 'microfone', 'anexos', 'contatos', 'shell')]
