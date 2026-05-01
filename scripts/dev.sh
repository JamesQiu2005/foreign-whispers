#!/usr/bin/env bash
# scripts/dev.sh — manage the three local services (Chatterbox, API, Frontend)
# without Docker, on Apple Silicon.
#
#   ./scripts/dev.sh start    [api|chatterbox|frontend|all]   default: all
#   ./scripts/dev.sh stop     [api|chatterbox|frontend|all]   default: all
#   ./scripts/dev.sh restart  [api|chatterbox|frontend|all]   default: all
#   ./scripts/dev.sh status
#   ./scripts/dev.sh logs     api|chatterbox|frontend         tail -f
#
# Notes:
#   - All three run in the background; PIDs are tracked under /tmp/fw-*.pid.
#   - Logs at /tmp/fw-{api,chatterbox,frontend}.log.
#   - Idempotent: starting an already-running service is a no-op.
#   - .env is sourced for FW_HF_TOKEN; UID/GID lines are skipped (read-only in bash).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

API_PORT=8080
CHATTERBOX_PORT=8020
FRONTEND_PORT=8501

API_LOG=/tmp/fw-api.log
CHATTERBOX_LOG=/tmp/fw-chatterbox.log
FRONTEND_LOG=/tmp/fw-frontend.log

API_PID=/tmp/fw-api.pid
CHATTERBOX_PID=/tmp/fw-chatterbox.pid
FRONTEND_PID=/tmp/fw-frontend.pid

# ── helpers ────────────────────────────────────────────────────────────────

c_green()   { printf "\033[0;32m%s\033[0m" "$1"; }
c_red()     { printf "\033[0;31m%s\033[0m" "$1"; }
c_yellow()  { printf "\033[0;33m%s\033[0m" "$1"; }

is_running() {
  local pidfile="$1"
  [[ -f "$pidfile" ]] && kill -0 "$(cat "$pidfile")" 2>/dev/null
}

port_listening() {
  lsof -i ":$1" -sTCP:LISTEN 2>/dev/null | grep -q LISTEN
}

wait_for_port() {
  local port="$1" name="$2" timeout="${3:-60}"
  for ((i = 0; i < timeout; i++)); do
    port_listening "$port" && return 0
    sleep 1
  done
  echo "$(c_red "✗")  $name didn't bind to :$port within ${timeout}s. See log."
  return 1
}

# Source .env safely (skip read-only UID/GID).
load_env() {
  if [[ -f "$ROOT/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source <(grep -vE '^(UID|GID)=' "$ROOT/.env" | grep -vE '^\s*#' | grep -vE '^\s*$')
    set +a
  fi
  export FW_HF_TOKEN="${FW_HF_TOKEN:-${HF_TOKEN:-}}"
}

# ── start ──────────────────────────────────────────────────────────────────

start_chatterbox() {
  if is_running "$CHATTERBOX_PID" || port_listening "$CHATTERBOX_PORT"; then
    echo "$(c_yellow "•")  chatterbox already up on :$CHATTERBOX_PORT"
    return 0
  fi
  echo "→  starting chatterbox-mps on :$CHATTERBOX_PORT (first run loads model, ~30s)"
  (
    cd "$ROOT/tools/chatterbox_server"
    nohup uv run python serve.py > "$CHATTERBOX_LOG" 2>&1 &
    echo $! > "$CHATTERBOX_PID"
  )
  wait_for_port "$CHATTERBOX_PORT" chatterbox 90 \
    && echo "$(c_green "✓")  chatterbox ready (PID $(cat "$CHATTERBOX_PID"))"
}

start_api() {
  if is_running "$API_PID" || port_listening "$API_PORT"; then
    echo "$(c_yellow "•")  api already up on :$API_PORT"
    return 0
  fi
  echo "→  starting Foreign Whispers API on :$API_PORT"
  load_env
  (
    cd "$ROOT"
    FW_WHISPER_MODEL="${FW_WHISPER_MODEL:-base}" \
    FW_HF_TOKEN="${FW_HF_TOKEN:-}" \
      nohup uv run uvicorn api.src.main:app \
        --host 0.0.0.0 --port "$API_PORT" \
        > "$API_LOG" 2>&1 &
    echo $! > "$API_PID"
  )
  wait_for_port "$API_PORT" api 30 \
    && echo "$(c_green "✓")  api ready (PID $(cat "$API_PID"))"
}

start_frontend() {
  if is_running "$FRONTEND_PID" || port_listening "$FRONTEND_PORT"; then
    echo "$(c_yellow "•")  frontend already up on :$FRONTEND_PORT"
    return 0
  fi
  echo "→  starting Next.js frontend on :$FRONTEND_PORT"
  (
    cd "$ROOT/frontend"
    API_URL="http://localhost:$API_PORT" \
    PORT="$FRONTEND_PORT" \
    HOSTNAME=0.0.0.0 \
      nohup pnpm dev > "$FRONTEND_LOG" 2>&1 &
    echo $! > "$FRONTEND_PID"
  )
  wait_for_port "$FRONTEND_PORT" frontend 60 \
    && echo "$(c_green "✓")  frontend ready (PID $(cat "$FRONTEND_PID"))    → http://localhost:$FRONTEND_PORT"
}

# ── stop ───────────────────────────────────────────────────────────────────

stop_one() {
  local name="$1" pidfile="$2" pattern="$3"
  if is_running "$pidfile"; then
    local pid
    pid=$(cat "$pidfile")
    echo "→  stopping $name (PID $pid)"
    kill "$pid" 2>/dev/null || true
    sleep 1
    kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
    rm -f "$pidfile"
  fi
  # Belt-and-braces: also kill any orphan matching the pattern (e.g. PID file lost).
  pkill -f "$pattern" 2>/dev/null || true
  echo "$(c_green "✓")  $name stopped"
}

stop_chatterbox() { stop_one chatterbox "$CHATTERBOX_PID" "tools/chatterbox_server/.*serve\.py"; }
stop_api()        { stop_one api        "$API_PID"        "uvicorn api\.src\.main:app"; }
stop_frontend()   { stop_one frontend   "$FRONTEND_PID"   "next dev|pnpm dev"; }

# ── status ─────────────────────────────────────────────────────────────────

status() {
  printf "%-12s %-8s %-8s %s\n" "service" "port" "pid" "health"
  printf "%-12s %-8s %-8s %s\n" "-------" "----" "---" "------"
  for triple in \
      "api $API_PORT $API_PID http://localhost:$API_PORT/healthz" \
      "chatterbox $CHATTERBOX_PORT $CHATTERBOX_PID http://localhost:$CHATTERBOX_PORT/health" \
      "frontend $FRONTEND_PORT $FRONTEND_PID http://localhost:$FRONTEND_PORT"; do
    set -- $triple
    name=$1; port=$2; pidfile=$3; url=$4
    pid="-"
    [[ -f "$pidfile" ]] && pid=$(cat "$pidfile")
    if curl -sf "$url" -o /dev/null --max-time 2 2>/dev/null; then
      health="$(c_green "up")"
    elif port_listening "$port"; then
      health="$(c_yellow "binding")"
    else
      health="$(c_red "down")"
      pid="-"
    fi
    printf "%-12s %-8s %-8s %s\n" "$name" "$port" "$pid" "$health"
  done
}

# ── tail logs ──────────────────────────────────────────────────────────────

tail_log() {
  case "$1" in
    api)        tail -f "$API_LOG"       ;;
    chatterbox) tail -f "$CHATTERBOX_LOG" ;;
    frontend)   tail -f "$FRONTEND_LOG"  ;;
    *) echo "usage: $0 logs api|chatterbox|frontend" >&2; exit 1 ;;
  esac
}

# ── dispatch ──────────────────────────────────────────────────────────────

cmd="${1:-status}"
target="${2:-all}"

case "$cmd" in
  start)
    case "$target" in
      api)        start_api ;;
      chatterbox) start_chatterbox ;;
      frontend)   start_api; start_frontend ;;  # frontend needs api
      all)        start_chatterbox; start_api; start_frontend ;;
      *) echo "usage: $0 start [api|chatterbox|frontend|all]" >&2; exit 1 ;;
    esac
    echo
    status
    ;;
  stop)
    case "$target" in
      api)        stop_api ;;
      chatterbox) stop_chatterbox ;;
      frontend)   stop_frontend ;;
      all)        stop_frontend; stop_api; stop_chatterbox ;;
      *) echo "usage: $0 stop [api|chatterbox|frontend|all]" >&2; exit 1 ;;
    esac
    ;;
  restart)
    "$0" stop "$target"
    "$0" start "$target"
    ;;
  status)  status ;;
  logs)    tail_log "$target" ;;
  *) echo "usage: $0 {start|stop|restart|status|logs} [target]" >&2; exit 1 ;;
esac
