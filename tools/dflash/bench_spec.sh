#!/bin/bash
# Speculative decoding against plain decoding, same binary, same card.
#
#   tools/dflash/bench_spec.sh VV_CLI MODEL DRAFTER OUT.csv AUDIO... [-- CLI ARGS]
#
# For every audio file: one plain run, then one run per --draft-block in
# BLOCKS (default "2 3 4 5 8"), each RUNS times (default 3, best kept). A
# row records decode tokens/s, RTF, tokens per drafted block and whether the
# transcript is byte-for-byte the plain one. Run on a card nothing else uses.
set -u
CLI=$1; MODEL=$2; DRAFT=$3; OUT=$4; shift 4
AUDIO=(); EXTRA=()
while [ $# -gt 0 ]; do
    if [ "$1" = "--" ]; then shift; EXTRA=("$@"); break; fi
    AUDIO+=("$1"); shift
done
BLOCKS=${BLOCKS:-"2 3 4 5 8"}
RUNS=${RUNS:-3}
TMP=$(mktemp -d)
echo "audio,mode,block,decode_tok_s,rtf,tokens_per_block,identical" > "$OUT"

run() {   # $1 = output json, rest = extra args; prints "tok/s rtf tpb"
    local out=$1; shift
    local log
    log=$("$CLI" --model "$MODEL" --output "$out" "${EXTRA[@]}" "$@" 2>&1)
    local tps rtf tpb
    tps=$(grep -oE "Decode speed +[0-9.]+" <<<"$log" | grep -oE "[0-9.]+$")
    rtf=$(grep -oE "RTF +[0-9.]+" <<<"$log" | grep -oE "[0-9.]+$")
    tpb=$(grep -oE "([0-9.]+) tokens (per cycle|each)" <<<"$log" | grep -oE "^[0-9.]+")
    echo "${tps:-0} ${rtf:-0} ${tpb:-0}"
}

best() {  # best (highest tok/s) of RUNS runs; $1 = json, rest = args
    local json=$1; shift
    local top="0 0 0"
    for _ in $(seq 1 "$RUNS"); do
        local r
        r=$(run "$json" "$@")
        if awk -v a="${r%% *}" -v b="${top%% *}" 'BEGIN{exit !(a>b)}'; then top=$r; fi
    done
    echo "$top"
}

for a in "${AUDIO[@]}"; do
    name=$(basename "$a")
    read -r tps rtf _ <<<"$(best "$TMP/plain.json" --audio "$a")"
    echo "$name,plain,0,$tps,$rtf,1,yes" >> "$OUT"
    for b in $BLOCKS; do
        read -r tps rtf tpb <<<"$(best "$TMP/spec.json" --audio "$a" --draft "$DRAFT" --draft-block "$b")"
        same=no
        cmp -s "$TMP/plain.json" "$TMP/spec.json" && same=yes
        echo "$name,spec,$b,$tps,$rtf,$tpb,$same" >> "$OUT"
    done
done
rm -rf "$TMP"
column -s, -t < "$OUT"
