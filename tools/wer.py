#!/usr/bin/env python3
"""Word error rate of vv_cli transcripts against a reference transcript.

    tools/wer.py REF.json HYP.json [HYP.json ...]

Both sides are vv_cli --output JSON. Words are lower-cased with punctuation
stripped (apostrophes kept). For every hypothesis this prints the WER
(substitutions + deletions + insertions over reference words), how many
aligned words carry a different speaker than in the reference, the mean
shift of the segment start of aligned words, and the segment counts.
Standard library only.
"""
import json
import re
import sys

WORD = re.compile(r"[a-z0-9']+")


def words(doc):
    """[(word, speaker, segment start)] in transcript order."""
    out = []
    for seg in doc.get("segments") or []:
        for w in WORD.findall(seg.get("text", "").lower()):
            out.append((w, seg.get("speaker"), float(seg.get("start", 0.0))))
    if not out:
        out = [(w, None, 0.0) for w in WORD.findall(doc.get("text", "").lower())]
    return out


def align(ref, hyp):
    """Levenshtein on words; returns (errors, [(i, j) matched pairs])."""
    n, m = len(ref), len(hyp)
    prev = list(range(m + 1))
    back = [bytearray(m + 1) for _ in range(n + 1)]   # 0 diag 1 up 2 left
    for j in range(1, m + 1):
        back[0][j] = 2
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        back[i][0] = 1
        ri = ref[i - 1][0]
        bi = back[i]
        for j in range(1, m + 1):
            d = prev[j - 1] + (ri != hyp[j - 1][0])
            u = prev[j] + 1
            l = cur[j - 1] + 1
            if d <= u and d <= l:
                cur[j] = d
            elif u <= l:
                cur[j] = u
                bi[j] = 1
            else:
                cur[j] = l
                bi[j] = 2
        prev = cur
    pairs = []
    i, j = n, m
    while i > 0 or j > 0:
        b = back[i][j]
        if b == 0:
            if ref[i - 1][0] == hyp[j - 1][0]:
                pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif b == 1:
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    return prev[m], pairs


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    ref_doc = json.load(open(sys.argv[1]))
    ref = words(ref_doc)
    print("ref %-40s %6d words %4d segments" %
          (sys.argv[1].split("/")[-1], len(ref), len(ref_doc.get("segments") or [])))
    for path in sys.argv[2:]:
        try:
            doc = json.load(open(path))
        except (OSError, ValueError) as e:
            print("hyp %-40s unreadable: %s" % (path.split("/")[-1], e))
            continue
        hyp = words(doc)
        err, pairs = align(ref, hyp)
        spk = sum(1 for i, j in pairs if ref[i][1] != hyp[j][1])
        dt = [abs(ref[i][2] - hyp[j][2]) for i, j in pairs]
        mean_dt = sum(dt) / len(dt) if dt else 0.0
        print("hyp %-40s WER %6.2f%% (%d/%d)  speaker %d/%d  start shift "
              "%.3f s  %d segments" %
              (path.split("/")[-1], 100.0 * err / max(1, len(ref)), err,
               len(ref), spk, len(pairs), mean_dt,
               len(doc.get("segments") or [])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
