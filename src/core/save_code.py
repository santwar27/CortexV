"""Save recent channel code as inert files, without involving the model.

Payloads are UTF-8 encodings of the exact text between complete fence lines.
Publication uses the PDF publisher's macOS RENAME_EXCL primitive: a completed
temporary directory becomes visible in one move, never replacing any entry.
"""
from contextlib import ExitStack
from dataclasses import dataclass
import base64
import errno
import hashlib
import hmac
import os
from pathlib import Path
import re
import secrets
import stat
import unicodedata
from urllib.parse import quote, unquote

from src.core.dispatch.normalisation import strip_courtesy, strip_vocatives
from src.core.i18n import t
from src.core.pdf_publish import _open_dir, _rename_exclusive


ROOT_PARTS = ('Documents', 'Cortex', 'Codigo')
MAX_REPLIES = 5
_DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
_ALLOWED = {'.html', '.css', '.js', '.json', '.md', '.txt'}
_DEFAULTS = {'html': 'index.html', 'css': 'styles.css', 'js': 'script.js',
             'javascript': 'script.js', 'json': 'codigo.json', 'md': 'codigo.md',
             'markdown': 'codigo.md', 'txt': 'codigo.txt', '': 'codigo.txt'}
_CODE = r'(?:c[óo]digo|code|html|css|site)'
_OBJECT = r'(?:(?:esse|este|aquele|this|that|o)\s+' + _CODE + r')'
_PT_VERB = r'(?:salv[ae]r?|grav[ae]r?|cri[ae]r?)'
_CONTAINER = r'(?:um\s+)?(?:arquivo|doc|documento)'
_REQUEST = re.compile(
    r'(?:(?:me\s+)?' + _PT_VERB + r'\s+(?:' + _OBJECT + r'|'
    + _CONTAINER + r'\s+com\s+' + _OBJECT + r'|'
    r'(?:isso|isto)\s+em\s+' + _CONTAINER + r'\s+(?:de|com)\s+' + _CODE + r')'
    r'|save\s+' + _OBJECT + r'|put\s+' + _OBJECT + r'\s+in\s+a\s+file)'
    r'(?:\s+(?:em\s+' + _CONTAINER + r'|in\s+a\s+file))?'
    r'(?:\s+(?:em|como|as|to)\s+(?P<name>.+?))?'
    r'(?:\s+(?:para|pra)\s+(?:que\s+eu\s+possa\s+|eu\s+)?'
    r'abrir\s+o\s+preview(?:\s+dele)?)?\s*[.!?]?', re.I)
_OPEN = re.compile(r'^[ \t]{0,3}(?P<fence>`{3,}|~{3,})(?P<lang>[^\r\n`]*)\r?\n', re.M)
_FINDER = re.compile(
    r'(?:mostra no finder|abre no finder|abre a pasta (?:do site|do c[óo]digo)|'
    r'show (?:it )?in finder|open the folder)\s*[.!?]?', re.I)
_SAVED_FOLDER = re.compile(
    r'^(?:Salvei |Saved |Salvei \d+ arquivos em |Saved \d+ files in )'
    r'~/Documents/Cortex/Codigo/([^/:\r\n]+)(?:/[^/\r\n]+\.|: )')
_LOCATION = re.compile(
    r'(?:onde est[áa] o arquivo|onde (?:voc[êe] )?salvou|'
    r'onde ficou (?:o arquivo|o site)|qual (?:[ée] )?o link|'
    r'(?:passa|manda|me d[áa]) o link|cad[êe] (?:o arquivo|o link)|'
    r'where is the file|where did you save it|(?:send|give) me the link|'
    r"what['’]s the link)\s*[.!?]?", re.I)
_WEB_SUFFIX = re.compile(r'\.(html?|css|js)(?:\.txt)?$', re.I)
_HTML_START = re.compile(r'\s*(?:<!doctype\s+html\b|<html\b)', re.I)


def is_finder_request(text: str) -> bool:
    text = strip_courtesy(strip_vocatives((text or '').strip()))
    return _FINDER.fullmatch(text) is not None


def is_saved_location_request(text: str) -> bool:
    text = strip_courtesy(strip_vocatives((text or '').strip()))
    return _LOCATION.fullmatch(text) is not None


def _last_saved_receipt(history):
    for message in reversed(history):
        if message.get('role') == 'assistant':
            content = message.get('content') or ''
            match = _SAVED_FOLDER.match(content)
            if match:
                return match[1], content
    return None, ''


def last_saved_folder(history: list) -> str | None:
    """Share Finder's receipt lookup; never consult another channel."""
    return _last_saved_receipt(history)[0]


def folder_link_key(folder: str) -> str:
    """A URL-safe capability for one folder, separate from login credentials."""
    api_key = os.getenv('CORTEX_API_KEY')
    if not api_key:
        raise ValueError('saved-code link secret is not configured')
    secret = hmac.digest(api_key.encode('utf-8'), b'cortex-codigo-v1', hashlib.sha256)
    digest = hmac.digest(secret, folder.encode('utf-8'), hashlib.sha256)
    return base64.urlsafe_b64encode(digest).rstrip(b'=').decode('ascii')


def _safe_component(name):
    return (isinstance(name, str) and bool(name) and not name.startswith('.')
            and '..' not in name and not any(c in name for c in '/\\:')
            and not any(unicodedata.category(c).startswith('C') for c in name))


def _open_saved_folder(folder, stack):
    """Read-only counterpart of _root; pin every component without links."""
    if not _safe_component(folder):
        raise ValueError('invalid saved folder')
    root_fd = _open_dir(Path.home().joinpath(*ROOT_PARTS), stack)
    fd = os.open(folder, _DIR_FLAGS, dir_fd=root_fd)
    stack.callback(os.close, fd)
    return fd


def _open_saved_file(folder_fd, name, stack):
    if not _safe_component(name) or Path(name).suffix.lower() not in _ALLOWED:
        raise ValueError('invalid saved filename')
    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=folder_fd)
    stack.callback(os.close, fd)
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        raise ValueError('saved file must be regular')
    return fd


def read_saved_file(folder: str, name: str) -> bytes:
    """Read the verified descriptor, never re-open a checked pathname."""
    with ExitStack() as stack:
        folder_fd = _open_saved_folder(folder, stack)
        fd = _open_saved_file(folder_fd, name, stack)
        with os.fdopen(os.dup(fd), 'rb') as source:
            return source.read()


def _site_link(folder: str, html: str, lang: str, url: str | None = None) -> str:
    url = url or (f'/ui/codigo/{folder_link_key(folder)}/'
                  + quote(folder, safe='') + '/' + quote(html, safe=''))
    return t(lang, f' [Abrir o site]({url})', f' [Open the site]({url})')


def saved_location(history: list, lang: str) -> str | None:
    """Return disk facts for the last receipt, or leave ordinary routing alone."""
    folder, receipt = _last_saved_receipt(history)
    if folder is None:
        return None
    destination = '~/Documents/Cortex/Codigo/' + folder
    link = re.search(r'\[(?:Abrir o site|Open the site)\]'
                     r'\((/ui/codigo/[A-Za-z0-9_-]+/[^)\r\n]+)\)', receipt.split('\n', 1)[0])
    html_url = None
    html_name = None
    if link:
        parts = link[1].split('/')
        if (len(parts) == 6 and unquote(parts[4]) == folder
                and _safe_component(unquote(parts[5])) and parts[5].lower().endswith('.html')):
            html_url, html_name = link[1], unquote(parts[5])
    try:
        with ExitStack() as stack:
            folder_fd = _open_saved_folder(folder, stack)
            names = ([html_name] if html_name else
                     sorted(os.listdir(folder_fd), key=lambda n: (not n.lower().endswith('.html'), n)))
            for name in names:
                try:
                    with ExitStack() as file_stack:
                        _open_saved_file(folder_fd, name, file_stack)
                except (OSError, ValueError):
                    continue
                reply = t(lang, f'Está em {destination}/{name}.', f"It's in {destination}/{name}.")
                if name.lower().endswith('.html'):
                    reply += _site_link(folder, name, lang, html_url)
                return reply
    except (OSError, ValueError):
        return t(lang, f'A pasta {destination} não existe mais.',
                 f'The folder {destination} no longer exists.')
    return t(lang, 'Não achei arquivos salvos nessa pasta.',
             "I couldn't find saved files in that folder.")


def show_saved_in_finder(history: list, lang: str) -> str:
    """Use only this channel's last literal save receipt, including after reload.

    UI persists receipts in its chat store. save_last_code also records them
    in engine history for voice/terminal, whose deterministic replies otherwise
    bypass engine.chat. No global last-folder slot can leak across channels.
    """
    from src.tools.macos_controller import _run

    folder = last_saved_folder(history)
    if folder is None:
        return t(lang, 'Ainda não salvei código nesta conversa.',
                 "I haven't saved code in this conversation yet.")
    destination = '~/Documents/Cortex/Codigo/' + folder
    try:
        with ExitStack() as stack:
            folder_fd = _open_saved_folder(folder, stack)
            # HTML first; otherwise the first regular saved kind by filename.
            names = sorted(os.listdir(folder_fd), key=lambda n: (not n.lower().endswith('.html'), n))
            for name in names:
                try:
                    with ExitStack() as file_stack:
                        _open_saved_file(folder_fd, name, file_stack)
                except (OSError, ValueError):
                    continue
                target = Path.home().joinpath(*ROOT_PARTS, folder, name)
                ok, _ = _run(['open', '-R', str(target)], timeout=10, lang=lang)
                return (t(lang, 'Abri a pasta no Finder do Mac.',
                          "I opened the folder in the Mac's Finder.") if ok else
                        t(lang, 'Não consegui abrir a pasta no Finder do Mac.',
                          "I couldn't open the folder in the Mac's Finder."))
    except (OSError, ValueError):
        return t(lang, f'A pasta {destination} não existe mais.',
                 f'The folder {destination} no longer exists.')
    return t(lang, 'Não achei arquivos salvos nessa pasta.',
             "I couldn't find saved files in that folder.")


@dataclass(frozen=True)
class SaveRequest:
    name: str | None


def parse_request(text: str) -> SaveRequest | None:
    """Require a save verb and a deictic code object, never a code mention."""
    text = strip_courtesy(strip_vocatives((text or '').strip()))
    match = _REQUEST.fullmatch(text)
    return SaveRequest(match.group('name')) if match else None


def _basename(value: str, default: str = 'codigo') -> str:
    value = unicodedata.normalize('NFC', value.strip().strip('`*\"\''))
    value = value.replace('\\', '/').rsplit('/', 1)[-1]
    value = ''.join(c for c in value if not unicodedata.category(c).startswith('C')
                    and c not in '/:')
    value = ' '.join(value.split()).replace('..', '').lstrip('. ').rstrip(' .')
    # Bound encoded size too, leaving room for extensions and numeric suffixes.
    while len(value.encode('utf-8')) > 160:
        value = value[:-1]
    return value or default


def _label(prefix: str) -> str | None:
    lines = prefix.rstrip().splitlines()
    if not lines:
        return None
    line = lines[-1].strip().rstrip(':').strip()
    if line.startswith('**') and line.endswith('**'):
        return line[2:-2]
    if line.startswith('`') and line.endswith('`'):
        return line[1:-1]
    # Recognise bare filenames and path-like labels, without taking prose.
    # Even a hidden/absolute label supplies only a sanitised basename.
    if re.fullmatch(r'(?:[^\s`*]+\.[^\s`*]+|[./\\][^\s`*]+)', line):
        return line
    return None


def _blocks(reply: str):
    cursor = 0
    while opening := _OPEN.search(reply, cursor):
        fence = opening.group('fence')
        closing = re.compile(r'^[ \t]{0,3}' + re.escape(fence[0]) +
                             '{' + str(len(fence)) + r',}[ \t]*\r?$', re.M)
        end = closing.search(reply, opening.end())
        if end is None:
            return  # An incomplete final block is never guessed or repaired.
        yield (opening.group('lang').strip().lower(),
               _label(reply[cursor:opening.start()]), reply[opening.end():end.start()])
        cursor = end.end()


def _recent_blocks(history: list):
    replies = 0
    for message in reversed(history):
        if message.get('role') != 'assistant':
            continue
        replies += 1
        if replies > MAX_REPLIES:
            break
        blocks = list(_blocks(message.get('content') or ''))
        if blocks:
            return blocks
    return []


def _files(blocks, requested):
    result = []
    used = set()
    override_used = False
    for language, label, content in blocks:
        override = requested and not override_used and (len(blocks) == 1 or language == 'html')
        if override:
            override_used = True
        name = _basename(requested if override else label or _DEFAULTS.get(language, 'codigo.txt'))
        if Path(name).suffix.lower() not in _ALLOWED:
            name += '.txt'
        elif language not in _DEFAULTS and not name.lower().endswith('.txt'):
            name += '.txt'
        original = Path(name)
        number = 2
        # Case-insensitive uniqueness also works on macOS default filesystems.
        while name.casefold() in used:
            name = f'{original.stem}-{number}{original.suffix}'
            number += 1
        used.add(name.casefold())
        result.append((name, content.encode('utf-8')))
    return result


def _root(stack):
    fd = _open_dir(Path.home(), stack)
    for part in ROOT_PARTS:
        try:
            os.mkdir(part, 0o755, dir_fd=fd)
        except FileExistsError:
            pass
        fd = os.open(part, _DIR_FLAGS, dir_fd=fd)
        stack.callback(os.close, fd)
    return fd


def _write_files(folder_fd, files):
    verified = []
    for name, content in files:
        fd = os.open(name, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                     0o644, dir_fd=folder_fd)
        try:
            remaining = memoryview(content)
            while remaining:
                count = os.write(fd, remaining)
                if count <= 0:
                    raise OSError('write did not advance')
                remaining = remaining[count:]
            os.fsync(fd)
            os.lseek(fd, 0, os.SEEK_SET)
            readback = bytearray()
            while chunk := os.read(fd, 65536):
                readback.extend(chunk)
            if readback != content:
                raise OSError('code readback mismatch')
            verified.append(name)
        finally:
            os.close(fd)
    os.fsync(folder_fd)
    return verified


def _publish(files, stem):
    with ExitStack() as stack:
        root_fd = _root(stack)
        staging = '.cortex-code-' + secrets.token_hex(16)
        os.mkdir(staging, 0o700, dir_fd=root_fd)
        # We own only this new, flat staging folder. Never chmod or clean up
        # any unrelated/stale folder, including a staging-name collision.
        folder_fd = None
        committed = False
        try:
            folder_fd = os.open(staging, _DIR_FLAGS, dir_fd=root_fd)
            verified = _write_files(folder_fd, files)
            for number in range(1, 100):
                name = stem if number == 1 else f'{stem} ({number})'
                try:
                    _rename_exclusive(root_fd, staging, name)
                except OSError as error:
                    if error.errno in (errno.EEXIST, errno.ENOTEMPTY):
                        continue
                    raise
                committed = True
                # A failure after commit cannot turn a real save into a lie.
                try:
                    os.fsync(root_fd)
                except OSError:
                    pass
                return name, verified
            raise OSError('all numbered folders occupied')
        finally:
            if folder_fd is None:
                os.rmdir(staging, dir_fd=root_fd)
            else:
                try:
                    if not committed:
                        for entry in os.listdir(folder_fd):
                            os.unlink(entry, dir_fd=folder_fd)
                        os.rmdir(staging, dir_fd=root_fd)
                finally:
                    os.close(folder_fd)


def _save_failure(lang):
    return t(lang, 'Não consegui salvar o código — nada foi gravado.',
             "I couldn't save the code — nothing was written.")


def publish_code(files: list[tuple[str, bytes]], stem: str, lang: str) -> str:
    """Publish prepared basenames/bytes with the one exclusive code publisher."""
    try:
        if (not _safe_component(stem) or not files
                or any(not _safe_component(name) or Path(name).suffix.lower() not in _ALLOWED
                       for name, _ in files)):
            raise ValueError('invalid code destination')
        # Check link configuration BEFORE publication: a saved file must always
        # have a truthful receipt even when the server secret is missing.
        if any(name.lower().endswith('.html') for name, _ in files):
            folder_link_key(stem)
        folder, names = _publish(files, stem)
    except Exception:
        return _save_failure(lang)
    destination = '~/Documents/Cortex/Codigo/' + folder
    html = next((name for name in names if name.lower().endswith('.html')), None)
    if len(names) == 1:
        reply = t(lang, f'Salvei {destination}/{names[0]}.', f'Saved {destination}/{names[0]}.')
        if html:
            reply += t(lang, ' Abra no navegador para ver.', ' Open it in a browser to view it.')
    else:
        reply = t(lang, f'Salvei {len(names)} arquivos em {destination}: {", ".join(names)}.',
                  f'Saved {len(names)} files in {destination}: {", ".join(names)}.')
        if html:
            reply += t(lang, f' Abra o {html} no navegador para ver.',
                       f' Open {html} in a browser to view it.')
    if html:
        reply += _site_link(folder, html, lang)
    return reply


def _web_kind(filename: str, content: str) -> str | None:
    match = _WEB_SUFFIX.search((filename or '').strip())
    if match:
        return 'html' if match[1].lower() in ('html', 'htm') else match[1].lower()
    # Detect HTML inside an optional opening fence without trimming the body.
    opening = _OPEN.match(content.lstrip())
    body = content.lstrip()[opening.end():] if opening else content
    return 'html' if _HTML_START.match(body) else None


def save_web_document(filename: str, content: str, lang: str, *,
                      request_text: str = "") -> str | None:
    """Route web tool payloads to code; None means ordinary text-note semantics.

    Only a complete outer fence is removed. Leading/trailing body whitespace,
    CRLFs and every other payload byte survive except explicit HTML repairs.
    """
    language = _web_kind(filename, content)
    if language is None:
        return None
    try:
        opening = _OPEN.match(content.lstrip())
        if opening:
            source = content.lstrip()
            fence = opening['fence']
            closing = re.search(r'^[ \t]{0,3}' + re.escape(fence[0]) +
                                '{' + str(len(fence)) + r',}[ \t]*\r?$',
                                source[opening.end():], re.M)
            if closing is None or source[opening.end() + closing.end():].strip():
                return _save_failure(lang)
            content = source[opening.end():opening.end() + closing.start()]
        raw = _WEB_SUFFIX.sub('', (filename or '').strip())
        # Reuse the saver's basename rules, with a site-specific unnamed fallback.
        basename = _basename(raw, '')
        if not _WEB_SUFFIX.search((filename or '').strip()) and basename:
            basename = Path(basename).stem
        stem = basename or 'site'
        name = basename + '.' + language if basename else _DEFAULTS[language]
        notes = []
        if language == 'html':
            from src.core.web_repairs import repair_html, fixes_line, example_warning
            requested = re.search(r'(?<![\w./-])([A-Za-z0-9_-]+\.html)\b', request_text, re.I)
            if requested:
                name = requested[1]
            content, changes = repair_html(content)
            incomplete = '' if re.search(r'</html\s*>', content, re.I) else t(
                lang,
                'O código parece incompleto (sem o fim da página). Peça de novo com menos seções.',
                'The code looks incomplete (the end of the page is missing). Ask again with fewer sections.')
            notes = [line for line in (fixes_line(changes, lang), incomplete,
                                      example_warning(content, request_text, lang)) if line]
        reply = publish_code([(name, content.encode('utf-8'))], stem, lang)
    except Exception:
        return _save_failure(lang)
    if reply == _save_failure(lang):
        return reply
    if notes:
        reply += '\n' + '\n'.join(notes)
    # A longer fence keeps code containing literal fence lines recoverable.
    fence = '`' * max(3, max((len(m[0]) + 1 for m in re.finditer(r'`+', content)), default=0))
    return reply + '\n\n' + fence + language + '\n' + content + (
        '' if content.endswith('\n') else '\n') + fence


def save_last_code(text: str, history: list, lang: str) -> str:
    """Save only this channel's most recent fenced reply; report disk facts."""
    try:
        request = parse_request(text)
        if request is None:
            return _save_failure(lang)
        blocks = _recent_blocks(history)
        if not blocks:
            return t(lang, 'Não achei código nas últimas respostas — me peça o código de novo.',
                     "I couldn't find code in the last replies — ask me for the code again.")
        files = _files(blocks, request.name)
        stem = (_basename(Path(_basename(request.name)).stem) if request.name else
                'site' if any(language == 'html' for language, _, _ in blocks) else 'codigo')
        reply = publish_code(files, stem, lang)
    except Exception:
        return _save_failure(lang)
    if reply == _save_failure(lang):
        return reply
    # Voice/terminal deterministic replies bypass engine.chat. Keep complete
    # turns for the next model call. UI receives a disposable history copy;
    # its durable user/receipt pair is recorded once by the chat route.
    history.extend([{'role': 'user', 'content': text},
                    {'role': 'assistant', 'content': reply}])
    return reply
