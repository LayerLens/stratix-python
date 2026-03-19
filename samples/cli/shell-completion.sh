#!/usr/bin/env bash
# LayerLens CLI — Shell Completion Installer
# Installs tab-completion for the LayerLens CLI scripts in bash or zsh.
#
# Usage:
#   chmod +x shell-completion.sh
#   ./shell-completion.sh install        # auto-detect shell
#   ./shell-completion.sh install bash   # force bash
#   ./shell-completion.sh install zsh    # force zsh
#   ./shell-completion.sh uninstall
#
# After install, restart your shell or run: source ~/.bashrc  (or ~/.zshrc)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPLETION_DIR="$HOME/.layerlens/completions"

# LayerLens CLI commands and their subcommands
COMMANDS="traces evaluations judges replay exports"
JUDGES_SUBS="list create test"
EXPORTS_FORMATS="csv json parquet"

# ── Generate bash completion function ─────────────────────────────────────────
generate_bash_completion() {
    cat <<'BASH_COMP'
# LayerLens CLI bash completion — auto-generated
_layerlens_complete() {
    local cur="${COMP_WORDS[COMP_CWORD]}"
    local prev="${COMP_WORDS[COMP_CWORD-1]}"
    local script="${COMP_WORDS[0]##*/}"
    script="${script%.sh}"

    case "$script" in
        judges)     COMPREPLY=($(compgen -W "list create test" -- "$cur")) ;;
        exports)    COMPREPLY=($(compgen -W "csv json parquet --type --output" -- "$cur")) ;;
        replay)     [[ "$prev" == "--model" ]] && COMPREPLY=() || COMPREPLY=($(compgen -W "--model" -- "$cur")) ;;
        *)          COMPREPLY=() ;;
    esac
}

for _cmd in traces evaluations judges replay exports; do
    complete -F _layerlens_complete "${_cmd}.sh"
    complete -F _layerlens_complete "$_cmd"
done
BASH_COMP
}

# ── Generate zsh completion function ──────────────────────────────────────────
generate_zsh_completion() {
    cat <<'ZSH_COMP'
# LayerLens CLI zsh completion — auto-generated
_layerlens_judges() {
    _arguments '1:command:(list create test)' '*:args:'
}
_layerlens_exports() {
    _arguments '1:format:(csv json parquet)' '--type[Export type]:type:(traces evaluations)' '--output[Output file]:file:_files'
}
_layerlens_replay() {
    _arguments '1:trace_id:' '--model[Model override]:model:'
}
compdef _layerlens_judges judges.sh judges
compdef _layerlens_exports exports.sh exports
compdef _layerlens_replay replay.sh replay
ZSH_COMP
}

# ── Install ───────────────────────────────────────────────────────────────────
cmd_install() {
    local shell_name="${1:-}"
    if [[ -z "$shell_name" ]]; then
        shell_name="$(basename "$SHELL")"
    fi

    mkdir -p "$COMPLETION_DIR"
    echo "Installing completions for: $shell_name"

    case "$shell_name" in
        bash)
            generate_bash_completion > "$COMPLETION_DIR/layerlens.bash"
            local rc="$HOME/.bashrc"
            local source_line="source $COMPLETION_DIR/layerlens.bash"
            if ! grep -qF "$source_line" "$rc" 2>/dev/null; then
                echo "" >> "$rc"
                echo "# LayerLens CLI completions" >> "$rc"
                echo "$source_line" >> "$rc"
            fi
            echo "Installed: $COMPLETION_DIR/layerlens.bash"
            echo "Added source line to $rc"
            ;;
        zsh)
            generate_zsh_completion > "$COMPLETION_DIR/_layerlens"
            local rc="$HOME/.zshrc"
            local fpath_line="fpath=($COMPLETION_DIR \$fpath)"
            if ! grep -qF "$fpath_line" "$rc" 2>/dev/null; then
                echo "" >> "$rc"
                echo "# LayerLens CLI completions" >> "$rc"
                echo "$fpath_line" >> "$rc"
                echo "autoload -Uz compinit && compinit" >> "$rc"
            fi
            echo "Installed: $COMPLETION_DIR/_layerlens"
            echo "Updated fpath in $rc"
            ;;
        *)
            echo "ERROR: Unsupported shell '$shell_name'. Use bash or zsh." >&2
            exit 1 ;;
    esac

    echo "Restart your shell or source your rc file to activate."
}

# ── Uninstall ─────────────────────────────────────────────────────────────────
cmd_uninstall() {
    echo "Removing $COMPLETION_DIR ..."
    rm -rf "$COMPLETION_DIR"
    echo "NOTE: Manually remove LayerLens source lines from ~/.bashrc or ~/.zshrc."
    echo "Done."
}

# ── Dispatch ──────────────────────────────────────────────────────────────────
case "${1:-help}" in
    install)   shift; cmd_install "$@" ;;
    uninstall) cmd_uninstall ;;
    *)
        echo "Usage: shell-completion.sh [install [bash|zsh] | uninstall]" >&2
        exit 1 ;;
esac
