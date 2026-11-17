"""
Tool specifications for the Cortex orchestrator's function-calling layer.

Format follows the OpenAI / Ollama convention: each tool has a name,
a natural-language description (the LLM uses this to decide when to
call it), and a JSON Schema for its arguments. The orchestrator's
`_execute_tool` method dispatches the actual call when the LLM picks a
tool.

Keep descriptions concrete and example-rich — the LLM reads them at
inference time, so quality of phrasing directly affects accuracy.
"""

TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "read_document",
            "description": (
                "Open, read or summarise a named document in the vault "
                "('abre o roadmap.md') or personal notes/dossier. "
                "Use find_file when asked where a file is or its location is unknown. "
                "Answer general questions, code help, creative writing and chat directly."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": (
                            ".txt/.pdf/.docx filename."
                        ),
                    },
                    "query": {
                        "type": "string",
                        "description": (
                            "Focus/question; omit for summary."
                        ),
                    },
                },
                "required": ["filename"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_file",
            "description": (
                "Find where a file is or search if its location is unknown "
                "('cadê meu currículo'). Return home paths via Spotlight. "
                "Use read_document to open, read or summarise a named document. "
                "No general questions/chat."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Filename key term only; omit arquivo/file/meu and extension."
                        ),
                    },
                    "content": {
                        "type": "boolean",
                        "description": (
                            "Default false matches names. Use true to search contents "
                            "only when the user describes what files say."
                        ),
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file_at",
            "description": (
                "Read a path from find_file or the user; never invent one. "
                "Use read_document for vault files by name. Confirm before opening."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "Exact absolute path; .pdf/.txt/.md/.docx only."
                        ),
                    },
                    "question": {
                        "type": "string",
                        "description": (
                            "User's own words; omit for summary."
                        ),
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_document",
            "description": (
                "Append to the vault only when asked to save, store or record. "
                "Web code gets a new site folder. Answer questions/chat directly."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": (
                            "Create if absent. Notes use .txt; web code keeps .html/.css/.js."
                        ),
                    },
                    "content": {
                        "type": "string",
                        "description": (
                            "Refined, structured professional prose; no chat filler."
                        ),
                    },
                },
                "required": ["filename", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_emails",
            "description": (
                "Read recent Gmail messages or mail from a sender. No send/edit/delete."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "count": {
                        "type": "integer",
                        "description": (
                            "1-25; default 5."
                        ),
                    },
                    "query": {
                        "type": "string",
                        "description": (
                            "Gmail syntax; omit for inbox."
                        ),
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": (
                "Use for any request to send, write, reply to or forward an email, "
                "even polite or indirect requests ('pode mandar', 'manda um email'). "
                "Put the recipient address in to. Propose via Gmail for confirmation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {
                        "type": "string",
                        "description": "One recipient email address.",
                    },
                    "subject": {
                        "type": "string",
                        "description": "Subject.",
                    },
                    "body": {
                        "type": "string",
                        "description": (
                            "Plain text; user's language; concise, professional."
                        ),
                    },
                },
                "required": ["to", "subject", "body"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_events",
            "description": (
                "Read primary Google Calendar events. Resolve relative times "
                "against the current date and local time given in the conversation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "time_min": {
                        "type": "string",
                        "description": (
                            "ISO 8601 range start; default now (UTC)."
                        ),
                    },
                    "time_max": {
                        "type": "string",
                        "description": (
                            "ISO 8601 range end; default time_min + 7 days."
                        ),
                    },
                    "max_results": {
                        "type": "integer",
                        "description": (
                            "1-50; default 10."
                        ),
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_event",
            "description": (
                "Propose an event in primary Google Calendar for confirmation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": (
                            "EXACT user title, spelling/language/punctuation; never "
                            "translate, pluralise or paraphrase."
                        ),
                    },
                    "start_iso": {
                        "type": "string",
                        "description": (
                            "ISO 8601 with offset. Resolve relative times against the current "
                            "date and local time given in the conversation."
                        ),
                    },
                    "end_iso": {
                        "type": "string",
                        "description": (
                            "ISO 8601 with offset. Resolve relative times against the current "
                            "date and local time given in the conversation; default duration is "
                            "1 hour if only start is given."
                        ),
                    },
                    "location": {
                        "type": "string",
                        "description": "Physical/virtual location.",
                    },
                    "description": {
                        "type": "string",
                        "description": "Details/agenda.",
                    },
                    "attendees": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Attendee email addresses.",
                    },
                },
                "required": ["summary", "start_iso", "end_iso"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_reminder",
            "description": (
                "Propose a reminder in macOS Reminders' default list for confirmation. "
                "iCloud syncs to iPhone/iPad."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": (
                            "EXACT user wording/language/accents/case; no "
                            "translation/paraphrase; prefer short imperative."
                        ),
                    },
                    "due_iso": {
                        "type": "string",
                        "description": (
                            "ISO 8601 with offset; resolve relative times against the current "
                            "date and local time given in the conversation."
                        ),
                    },
                },
                "required": ["title"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search changing facts: news/scores/weather/prices/rates/promotions. "
                "Stable knowledge: answer directly. Needs an internet lease; "
                "errors ask for authorisation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Natural-language query without search/pesquise/procure/busque."
                        ),
                    },
                },
                "required": ["query"],
            },
        },
    },
]
