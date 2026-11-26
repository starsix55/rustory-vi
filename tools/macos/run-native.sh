#!/bin/bash
# Runs the game natively and keeps a log you can attach to a tester report.
# Usage: tools/macos/run-native.sh [--headless] [--seconds N] [--console series-x|series-s] <game folder or .exe>
#   --headless            no window; frames go to an offscreen layer
#   --seconds N           stop the run after N seconds (default: run until you close it)
#   --console MODEL       console model reported to the game (default: series-s on <=16 GB Macs, else series-x)
# Extra VICTORY_* variables in your environment are passed through
# (see docs/native-host.md, "Bring-up diagnostics").
# VICTORY_SWAP_LIMIT_MB=N stops the run (and says so in the log) when system
# swap passes N MiB: a safety guard, not a successful test result.
set -uo pipefail
cd "$(dirname "$0")/../.."

headless=()
seconds=0
console=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --headless) headless=(--headless); shift ;;
    --seconds) seconds="$2"; shift 2 ;;
    --console) console="$2"; shift 2 ;;
    --console=*) console="${1#--console=}"; shift ;;
    -h|--help) sed -n 2,9p "$0"; exit 0 ;;
    *) break ;;
  esac
done
[[ $# -eq 1 ]] || { sed -n 2,9p "$0"; exit 2; }

ram_gb=$(( $(sysctl -n hw.memsize) / 1073741824 ))
if [[ -z "$console" ]]; then
  # Series X reports 13.5 GiB of title memory, which exceeds Metal's working
  # set on 16 GB Macs (#8). Default <=16 GB Macs to Series S (7.6 GiB).
  if (( ram_gb <= 16 )); then
    console="series-s"
  else
    console="series-x"
  fi
fi

target="$1"
if [[ -d "$target" ]]; then
  exe=$(find "$target" -maxdepth 1 -iname 'game_*master.exe' | head -1)
  [[ -n "$exe" ]] || { echo "No game_*master.exe found in $target"; exit 2; }
else
  exe="$target"
fi
[[ -f "$exe" ]] || { echo "Not a file: $exe"; exit 2; }

bin=target/x86_64-apple-darwin/release/victory
[[ -x "$bin" ]] || { echo "victory is not built yet: run tools/macos/native-setup.sh first"; exit 2; }

run_dir="runs/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$run_dir"
log="$run_dir/victory.log"
{
  echo "# $(sw_vers -productName) $(sw_vers -productVersion), $(sysctl -n machdep.cpu.brand_string), ${ram_gb} GB, console=${console}"
  echo "# commit $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
} > "$log"

export VICTORY_GUESS_GRAPHICS="${VICTORY_GUESS_GRAPHICS:-1}"
# Press Enter (the pad's A) every 2 s from 4 s to 80 s unless VICTORY_KEYS is set.
export VICTORY_KEYS="${VICTORY_KEYS:-$(seq 4 2 80 | sed 's/$/:Enter/' | paste -sd, -)}"
echo "Running $exe (console=$console)"
echo "Log: $log"

"$bin" run --execute --console "$console" ${headless[@]+"${headless[@]}"} --unimplemented zero "$exe" >> "$log" 2>&1 &
pid=$!
if (( seconds > 0 )); then
  ( sleep "$seconds"; kill "$pid" 2>/dev/null ) &
fi
guard=""
if [[ -n "${VICTORY_SWAP_LIMIT_MB:-}" ]]; then
  # Opt-in safety: stop this run (never report it as a success) before system
  # swap makes the whole Mac unresponsive. Parses `used = 12345.67M`.
  (
    while kill -0 "$pid" 2>/dev/null; do
      used=$(sysctl -n vm.swapusage | awk '{print $6}')
      used=${used%M}
      used=${used%.*}
      if [[ "$used" =~ ^[0-9]+$ ]] && (( used > VICTORY_SWAP_LIMIT_MB )); then
        echo "swap guard: system swap ${used} MiB > ${VICTORY_SWAP_LIMIT_MB} MiB; stopping the run" >> "$log"
        kill "$pid" 2>/dev/null
        break
      fi
      sleep 2
    done
  ) &
  guard=$!
fi
wait "$pid"
rc=$?
if [[ -n "$guard" ]]; then
  kill "$guard" 2>/dev/null
  wait "$guard" 2>/dev/null
fi

echo
echo "Exit code $rc. Last lines of the log:"
sed 's/\x1b\[[0-9;]*m//g' "$log" | tail -15 | cut -c1-200
echo
echo "To report: open a Tester report issue and attach $log"
