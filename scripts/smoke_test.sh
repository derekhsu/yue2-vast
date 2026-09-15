#!/bin/bash
# YuE2 smoke test — run against a deployed instance.
# Usage: MUSIC_API_KEY=xxx ./smoke_test.sh [base_url]
#   base_url defaults to http://127.0.0.1:7862 (run on the instance over SSH,
#   or point it at the Caddy endpoint https://<host>:8787 for external check).
set -euo pipefail

BASE="${1:-http://127.0.0.1:7862}"
: "${MUSIC_API_KEY:?set MUSIC_API_KEY}"
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT

pass() { echo "PASS: $*"; }
fail() { echo "FAIL: $*"; exit 1; }

# --- 1. /health ----------------------------------------------------------------
code=$(curl -s -o "$OUT/health.json" -w '%{http_code}' "$BASE/health")
[ "$code" = "200" ] || fail "/health -> $code ($(cat "$OUT/health.json"))"
grep -q '"ready"' "$OUT/health.json" || fail "/health not ready: $(cat "$OUT/health.json")"
pass "/health ready"

# --- 2. auth: no key -> 401 ------------------------------------------------------
code=$(curl -s -o /dev/null -w '%{http_code}' -X POST "$BASE/v1/audio/speech" \
    -H 'Content-Type: application/json' \
    -d '{"input":"[Verse]\nhi","instructions":"pop"}')
[ "$code" = "401" ] || fail "no-key request -> $code (expected 401)"
pass "auth rejects missing key"

# --- 3. /v1/models ---------------------------------------------------------------
code=$(curl -s -o "$OUT/models.json" -w '%{http_code}' \
    -H "Authorization: Bearer $MUSIC_API_KEY" "$BASE/v1/models")
[ "$code" = "200" ] || fail "/v1/models -> $code"
grep -q 'yue2' "$OUT/models.json" || fail "/v1/models missing yue2"
pass "/v1/models lists yue2"

# --- 4. short generation (1 verse + 1 chorus) ------------------------------------
cat > "$OUT/req.json" << 'EOF'
{
  "input": "[Verse]\nStreetlights hum a quiet tune\nEmpty roads beneath the moon\n\n[Chorus]\nCarry on, the night is young\nEvery ending is a song",
  "instructions": "English, gentle acoustic pop, soft female vocal, warm guitar, 80 BPM",
  "seed": 42,
  "cot": "full",
  "id": "smoke-test",
  "response_format": "flac"
}
EOF

echo "generating (this takes ~1-3 min on a 3090/4090)..."
code=$(curl -s -o "$OUT/song.flac" -w '%{http_code}' \
    -X POST "$BASE/v1/audio/speech" \
    -H "Authorization: Bearer $MUSIC_API_KEY" \
    -H 'Content-Type: application/json' \
    --max-time 900 \
    -d @"$OUT/req.json" \
    -D "$OUT/headers.txt")
[ "$code" = "200" ] || fail "generation -> $code: $(head -c 500 "$OUT/song.flac")"

# --- 5. validate output -----------------------------------------------------------
head -c 4 "$OUT/song.flac" | grep -q 'fLaC' || fail "output is not FLAC"
size=$(stat -f%z "$OUT/song.flac" 2>/dev/null || stat -c%s "$OUT/song.flac")
[ "$size" -gt 100000 ] || fail "FLAC too small: ${size}B"
grep -qi 'X-Seed: 42' "$OUT/headers.txt" || fail "X-Seed header missing/wrong"
pass "FLAC generated (${size} bytes, seed=42)"

echo "ALL CHECKS PASSED"
