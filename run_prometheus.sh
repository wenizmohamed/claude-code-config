#!/bin/bash
# run_prometheus.sh
# Script to launch prometheus_unbound.py with appropriate privileges and session management.
#
# CRITICAL WARNING: This script launches Prometheus Unbound with root privileges.
# Ensure the host system is physically air-gapped before execution.

set -e # Exit immediately if a command exits with a non-zero status.

log_info() {
    echo "[INFO] $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_error() {
    echo "[ERROR] $(date '+%Y-%m-%d %H:%M:%S') - $1" >&2
}

# Ensure script is run from the prometheus_unbound directory
CURRENT_DIR=$(pwd)
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
if [ "$CURRENT_DIR" != "$SCRIPT_DIR" ]; then
    log_info "Changing directory to $SCRIPT_DIR"
    cd "$SCRIPT_DIR" || { log_error "Failed to change directory."; exit 1; }
fi

# Check if running as prometheus user (sudo is handled internally by the script)
if [ "$USER" != "prometheus" ]; then
    log_error "This script should ideally be executed as the 'prometheus' user."
    log_error "The 'prometheus' user has passwordless sudo configured for the agent's operations."
    log_error "Attempting to switch to 'prometheus' user and re-execute. Please enter password if prompted."
    exec sudo -u prometheus /bin/bash -c "cd $SCRIPT_DIR && ./run_prometheus.sh $*"
    exit 0 # Should not be reached if exec is successful
fi

# Check for --resume argument
RESUME_SESSION=""
for arg in "$@"; do
    if [[ "$arg" == --resume=* ]]; then
        RESUME_SESSION="${arg#*=}"
        break
    fi
done

LOG_DIR="logs"
SESSIONS_DIR="sessions"
LAST_SESSION_FILE="$SESSIONS_DIR/last_session.json"

mkdir -p "$LOG_DIR" "$SESSIONS_DIR"

COMMAND="python3 prometheus_unbound.py"

if [ -n "$RESUME_SESSION" ]; then
    log_info "Attempting to resume session: $RESUME_SESSION"
    if [ -f "$SESSIONS_DIR/$RESUME_SESSION.json" ]; then
        COMMAND+=" --resume $RESUME_SESSION"
    else
        log_error "Session '$RESUME_SESSION' not found in '$SESSIONS_DIR'. Starting fresh."
    fi
elif [ -f "$LAST_SESSION_FILE" ]; then
    LAST_SESSION_ID=$(jq -r '.last_session_id' "$LAST_SESSION_FILE" 2>/dev/null)
    if [ -n "$LAST_SESSION_ID" ] && [ -f "$SESSIONS_DIR/$LAST_SESSION_ID.json" ]; then
        log_info "Last session '$LAST_SESSION_ID' found. Do you want to resume it? (yes/no):"
        read -r RESUME_PROMPT
        if [[ "$RESUME_PROMPT" =~ ^[Yy][Ee][Ss]$ ]]; then
            log_info "Resuming last session: $LAST_SESSION_ID"
            COMMAND+=" --resume $LAST_SESSION_ID"
        else
            log_info "Starting a new session."
        fi
    else
        log_info "No valid last session found or file corrupted. Starting a new session."
    fi
else
    log_info "No previous session data found. Starting a new session."
fi

log_info "Launching Prometheus Unbound..."
# Execute the main agent script. The script itself will use `sudo` where necessary due to passwordless sudo.
# It's launched by the prometheus user, and its internal tools will leverage its root access.
exec $COMMAND