# CORTEX shell helper — pretty curl-er.
#
# Source this once from your ~/.zshrc to get a `cx` function that
# sends a query to the Cortex API and prints the response cleanly
# (no \u00xx escapes, with status_mode header).
#
# Install:
#   echo 'source $HOME/cortex_project/scripts/cortex.sh' >> ~/.zshrc
#   source ~/.zshrc
#
# Use:
#   cx meus emails
#   cx agenda evento Reunião teste amanhã das 17h às 18h
#   cx sim
#   cx não
#
# Configuration: the function reads CORTEX_API_KEY from the project's
# .env file automatically. Override host/key for remote use:
#   CORTEX_HOST=100.75.236.81:8000 CORTEX_KEY=<your-key> cx meus emails

cx() {
    if [ "$#" -eq 0 ]; then
        echo "Usage: cx <text>"
        return 1
    fi
    local host="${CORTEX_HOST:-localhost:8000}"
    local key="${CORTEX_KEY}"

    # Fall back to reading CORTEX_API_KEY from the project's .env
    # (gitignored — secrets never get committed).
    if [ -z "$key" ]; then
        local env_file="$HOME/cortex_project/.env"
        if [ -f "$env_file" ]; then
            key=$(awk -F= '/^CORTEX_API_KEY=/{ sub(/^[^=]*=/, ""); print; exit }' "$env_file")
        fi
    fi

    if [ -z "$key" ]; then
        echo "ERROR: CORTEX_KEY not set and .env missing CORTEX_API_KEY." >&2
        echo "Either export CORTEX_KEY or fill .env from .env.example." >&2
        return 1
    fi

    local text="$*"

    python3 - "$host" "$key" "$text" <<'PY'
import json
import sys
import urllib.request

host, key, text = sys.argv[1], sys.argv[2], sys.argv[3]
req = urllib.request.Request(
    f"http://{host}/ask",
    method="POST",
    headers={"X-Cortex-Key": key, "Content-Type": "application/json"},
    data=json.dumps({"user_input": text}).encode("utf-8"),
)
try:
    with urllib.request.urlopen(req, timeout=120) as r:
        body = json.loads(r.read().decode("utf-8"))
except Exception as e:
    print(f"[ERROR] {e}", file=sys.stderr)
    sys.exit(1)

mode = body.get("status_mode", "?")
response = body.get("response", "")
# ANSI dim grey for the status mode header
print(f"\033[2m[{mode}]\033[0m")
print(response)
PY
}
