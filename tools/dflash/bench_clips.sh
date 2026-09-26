#!/bin/bash
# Speculative decoding over a set of clips: total decode time with and
# without the drafter, tokens per drafted block, and how many transcripts
# came out byte-for-byte the same.
#
#   tools/dflash/bench_clips.sh VV_CLI MODEL DRAFTER CLIPS.tsv [N] [-- CLI ARGS]
#
# CLIPS.tsv is the corpus manifest (id <tab> wav path [<tab> hotwords]); the
# first N clips (default all) are used. Run on a card nothing else uses.
set -u
export LC_ALL=C
CLI=$1; MODEL=$2; DRAFT=$3; LIST=$4; shift 4
N=0
if [ $# -gt 0 ] && [ "$1" != "--" ]; then N=$1; shift; fi
EXTRA=()
if [ $# -gt 0 ] && [ "$1" = "--" ]; then shift; EXTRA=("$@"); fi
TMP=$(mktemp -d)

sum_plain=0; tok_plain=0; sum_spec=0; tok_spec=0; blocks=0; btok=0; same=0; total=0
while IFS=$'\t' read -r id wav hot; do
    [ -z "$id" ] && continue
    [ "$N" -gt 0 ] && [ "$total" -ge "$N" ] && break
    hw=()
    [ -n "${hot:-}" ] && hw=(--hotwords "$hot")
    lp=$("$CLI" --model "$MODEL" --audio "$wav" "${hw[@]}" --output "$TMP/p.json" "${EXTRA[@]}" 2>&1)
    ls=$("$CLI" --model "$MODEL" --audio "$wav" "${hw[@]}" --output "$TMP/s.json" --draft "$DRAFT" "${EXTRA[@]}" 2>&1)
    read -r mp tp <<<"$(grep -oE "Autoregressive decode +[0-9.]+ ms +\([0-9]+ tokens" <<<"$lp" | grep -oE "[0-9.]+" | tr '\n' ' ')"
    read -r ms ts <<<"$(grep -oE "Autoregressive decode +[0-9.]+ ms +\([0-9]+ tokens" <<<"$ls" | grep -oE "[0-9.]+" | tr '\n' ' ')"
    nb=$(grep -oE "(spec: |stream: )[0-9]+ (cycles|drafted blocks)" <<<"$ls" | grep -oE "[0-9]+" | head -1)
    tpb=$(grep -oE "[0-9.]+ tokens (per cycle|each)" <<<"$ls" | grep -oE "^[0-9.]+" | head -1)
    btok=$(awk -v a="$btok" -v b="${tpb:-0}" -v c="${nb:-0}" 'BEGIN{print a+b*c}')
    sum_plain=$(awk -v a="$sum_plain" -v b="${mp:-0}" 'BEGIN{print a+b}')
    sum_spec=$(awk -v a="$sum_spec" -v b="${ms:-0}" 'BEGIN{print a+b}')
    tok_plain=$((tok_plain + ${tp:-0}))
    tok_spec=$((tok_spec + ${ts:-0}))
    blocks=$((blocks + ${nb:-0}))
    cmp -s "$TMP/p.json" "$TMP/s.json" && same=$((same + 1))
    total=$((total + 1))
    printf '%-40s plain %8.1f ms  spec %8.1f ms  %5s tokens\n' "$id" "${mp:-0}" "${ms:-0}" "${tp:-0}"
done < "$LIST"
rm -rf "$TMP"
awk -v sp="$sum_plain" -v ss="$sum_spec" -v tp="$tok_plain" -v ts="$tok_spec" \
    -v b="$blocks" -v bt="$btok" -v same="$same" -v n="$total" 'BEGIN{
    printf "clips %d, identical %d\n", n, same;
    printf "plain: %d tokens in %.1f s = %.1f tok/s\n", tp, sp/1000, tp/(sp/1000);
    printf "spec:  %d tokens in %.1f s = %.1f tok/s (%.2fx), %.2f tokens per block\n",
           ts, ss/1000, ts/(ss/1000), (ts/(ss/1000))/(tp/(sp/1000)), (b>0? bt/b : 0);
}'
