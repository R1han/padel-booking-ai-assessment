#!/usr/bin/env bash
#
# race.sh
# Issues N concurrent booking requests against a single slot and counts responses.
# Expected outcome: exactly one 201, with the remainder 409.
#
# Usage:
#   BASE_URL=http://localhost:3000 SLOT_ID=slot_alquoz_pc01_20260810_1800 ./tests/race.sh
#
# Optional:
#   N=20          number of concurrent requests
#   DURATION=60   duration_min sent with each request

set -uo pipefail

BASE_URL="${BASE_URL:-http://localhost:3000}"
SLOT_ID="${SLOT_ID:?SLOT_ID must be set}"
DURATION="${DURATION:-60}"
N="${N:-20}"

WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT

echo "POST ${BASE_URL}/api/v1/bookings"
echo "slot: ${SLOT_ID}   concurrency: ${N}"
echo

for i in $(seq 1 "$N"); do
  curl -s --max-time 30 \
    -o "${WORKDIR}/body_${i}.json" \
    -w '%{http_code}' \
    -X POST "${BASE_URL}/api/v1/bookings" \
    -H 'Content-Type: application/json' \
    -d "{\"slot_ids\":[\"${SLOT_ID}\"],\"user_id\":\"usr_race_${i}\",\"duration_min\":${DURATION}}" \
    > "${WORKDIR}/code_${i}.txt" 2>/dev/null &
done
wait

created=0
conflict=0
other=0

for i in $(seq 1 "$N"); do
  code="$(cat "${WORKDIR}/code_${i}.txt" 2>/dev/null || echo "000")"
  case "$code" in
    201) created=$((created + 1)) ;;
    409) conflict=$((conflict + 1)) ;;
    *)
      other=$((other + 1))
      echo "  unexpected ${code}: $(head -c 200 "${WORKDIR}/body_${i}.json" 2>/dev/null)"
      ;;
  esac
done

echo
echo "201 Created   ${created}   (expected 1)"
echo "409 Conflict  ${conflict}   (expected $((N - 1)))"
echo "other         ${other}   (expected 0)"
echo

if [ "$created" -eq 1 ] && [ "$other" -eq 0 ]; then
  echo "PASS"
  exit 0
fi

echo "FAIL"
exit 1