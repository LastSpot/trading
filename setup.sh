#!/usr/bin/env bash
# Platform: macOS / Linux — run: ./setup.sh
# Requires Bash 4+ (associative arrays). macOS ships Bash 3.2, so re-exec
# under Homebrew Bash if available.
if (( BASH_VERSINFO[0] < 4 )); then
    for candidate in /opt/homebrew/bin/bash /usr/local/bin/bash; do
        if [[ -x "$candidate" ]]; then
            exec "$candidate" "$0" "$@"
        fi
    done
    echo "[✗] Bash 4+ required. Install via 'brew install bash' and re-run." >&2
    exit 1
fi
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
cd "$SCRIPT_DIR"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m'

info()  { echo -e "${GREEN}[✓]${NC} $*"; }
warn()  { echo -e "${YELLOW}[!]${NC} $*"; }
error() { echo -e "${RED}[✗]${NC} $*"; }
step()  { echo -e "\n${BOLD}── $* ──${NC}"; }

status_label() {
    case "$1" in
        ok) echo "done" ;;
        partial) echo "partial (some steps skipped)" ;;
        *) echo "skipped" ;;
    esac
}

python_status="skipped"

setup_poetry() {
    step "Python environment (Poetry)"

    if ! command -v python3 &>/dev/null; then
        error "python3 not found on PATH"
        return 1
    fi
    info "Found python3: $(command -v python3)"

    if ! command -v poetry &>/dev/null; then
        info "Installing latest stable Poetry..."
        if ! curl -sSL https://install.python-poetry.org | python3 -; then
            warn "Official Poetry installer failed; trying pipx..."
            if command -v pipx &>/dev/null; then
                pipx install poetry || return 1
            else
                error "Could not install Poetry. Install pipx or Poetry manually."
                return 1
            fi
        fi
        export PATH="$HOME/.local/bin:$PATH"
    else
        info "Poetry already installed: $(command -v poetry)"
    fi

    if ! command -v poetry &>/dev/null; then
        error "Poetry is not available on PATH after install"
        return 1
    fi

    poetry config virtualenvs.in-project true --local

    if [[ ! -f "$SCRIPT_DIR/pyproject.toml" ]]; then
        local default_name
        default_name="$(basename "$SCRIPT_DIR")"
        local project_name=""
        read -rp "Project name [${default_name}]: " project_name
        if [[ -z "$project_name" ]]; then
            project_name="$default_name"
        fi

        info "Initializing Poetry project: $project_name"
        poetry init \
            --name "$project_name" \
            --python "^3.11" \
            --dependency "numpy:*" \
            --dependency "pandas:*" \
            --dependency "yfinance:*" \
            --no-interaction

        if [[ -f "$SCRIPT_DIR/pyproject.toml" ]]; then
            if grep -q '^package-mode = false' "$SCRIPT_DIR/pyproject.toml"; then
                info "package-mode already set to false"
            elif grep -q '^\[tool\.poetry\]' "$SCRIPT_DIR/pyproject.toml"; then
                sed -i.bak '/^\[tool\.poetry\]$/a\
package-mode = false' "$SCRIPT_DIR/pyproject.toml"
                rm -f "$SCRIPT_DIR/pyproject.toml.bak"
                info "Set package-mode = false in pyproject.toml"
            else
                if grep -q '^\[build-system\]' "$SCRIPT_DIR/pyproject.toml"; then
                    sed -i.bak '/^\[build-system\]/i\
\
[tool.poetry]\
package-mode = false\
' "$SCRIPT_DIR/pyproject.toml"
                    rm -f "$SCRIPT_DIR/pyproject.toml.bak"
                else
                    {
                        echo ""
                        echo "[tool.poetry]"
                        echo "package-mode = false"
                    } >> "$SCRIPT_DIR/pyproject.toml"
                fi
                info "Set package-mode = false in pyproject.toml"
            fi
        else
            error "poetry init did not create pyproject.toml"
            return 1
        fi
    else
        info "Found existing pyproject.toml — skipping init"
    fi

    info "Installing dependencies..."
    poetry install

    info "Activate with: poetry shell"
    info "Run backtest: poetry run python research/v2_trend_trio/v2_research.py"
    return 0
}

if setup_poetry; then
    python_status="ok"
else
    warn "Poetry setup failed"
fi

step "Setup complete"
echo ""
echo "  Poetry: $(status_label "$python_status")"
echo ""
