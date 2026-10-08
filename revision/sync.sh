#!/usr/bin/env bash
# Copy ONLY the final-epoch adapter of each run from the training machine (L20) to the eval pod,
# and pull responses/ back. Run on the L20 server. Small: ~150-200 MB per 7-8B run.
#
#   POD=root@<ip> PORT=<ssh port> bash revision/sync.sh push      # adapters + data + evalsets -> pod
#   POD=root@<ip> PORT=<ssh port> bash revision/sync.sh pull      # pod responses/ -> here
#
# Safe to run repeatedly (rsync only sends what changed). Put it in cron/watch during the run:
#   watch -n 600 'POD=... PORT=... bash revision/sync.sh push'
set -euo pipefail
cd "$(dirname "$0")/.."
: "${POD:?set POD=user@host}"; PORT="${PORT:-22}"; DEST="${DEST:-/workspace/persona-safe-ft}"
SSH="ssh -p $PORT -o StrictHostKeyChecking=accept-new"

case "${1:-push}" in
  push)
    for run in outputs_*; do
      [ -d "$run" ] || continue
      [ -f "$run/DONE" ] || continue          # only finished runs (train.py writes DONE at the end)
      last=$(ls -d "$run"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1) || true
      [ -n "$last" ] || continue
      rsync -az -e "$SSH" --relative "./$last" "./$run/DONE" "$POD:$DEST/"
    done
    rsync -az -e "$SSH" ./data ./evalsets "$POD:$DEST/" 2>/dev/null || true
    echo "pushed finished adapters ($(ls outputs_*/DONE 2>/dev/null | wc -l) runs) + data + evalsets"
    ;;
  pull)
    rsync -az -e "$SSH" "$POD:$DEST/responses/" ./responses/
    echo "pulled responses/"
    ;;
  *) echo "usage: sync.sh push|pull"; exit 1 ;;
esac
