#!/usr/bin/env python3
"""Cut the DFlash training corpus into clips: WAV files plus a manifest.

The drafter learns to predict what the *target* writes, so the text of these
datasets is never used -- only their audio, in the lengths and shapes the
runtime meets: long multi-speaker recordings (AMI meetings, VoxConverse,
earnings calls) cut into windows of 10 s to 15 min, and short read-speech
utterances (LibriSpeech, FLEURS in eight languages, SOVA) joined into clips of
10 s to 5 min. `gen` lines get the id and the path, tab-separated, and the
line order is shuffled so that any prefix of the manifest is a fair sample.

    build_corpus.py --src /mnt/hdd/vv3-dflash/datasets \
                    --out /mnt/hdd/vv3-dflash/clips --hours 140
"""
import argparse
import glob
import io
import os
import random

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

# Hours of clips per source, as fractions of --hours; eval gets held-out
# recordings (or held-out speakers) of the same sources.
MIX = {
    "ami": 0.22,
    "voxconverse": 0.11,
    "earnings22": 0.18,
    "librispeech": 0.20,
    "fleurs": 0.22,
    "sova": 0.07,
}
FLEURS = ["ru_ru", "cmn_hans_cn", "es_419", "de_de", "fr_fr", "ja_jp",
          "ko_kr", "pt_br"]


def clip_len(rng, cap=900.0):
    """Seconds: a quarter short, the bulk a few minutes, a tail of long."""
    r = rng.random()
    if r < 0.25:
        return rng.uniform(8, 40)
    if r < 0.70:
        return rng.uniform(40, 240)
    return min(cap, rng.uniform(240, 900))


def decode(audio):
    data, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32",
                       always_2d=True)
    return data.mean(axis=1), sr


def write(out_dir, cid, pcm, sr, rows):
    path = os.path.join(out_dir, cid + ".wav")
    if not os.path.exists(path):
        sf.write(path, np.clip(pcm, -1, 1), sr, subtype="PCM_16")
    rows.append((cid, path, len(pcm) / sr))


def windows(rng, pcm, sr, starts, budget_s, prefix, out_dir, rows):
    """Cut consecutive windows out of one long recording, each starting at
    an utterance start when the recording has them."""
    total = len(pcm) / sr
    t, k, used = 0.0, 0, 0.0
    starts = sorted(s for s in starts if s < total - 5) if starts else []
    while used < budget_s and t < total - 5:
        if starts:
            nxt = [s for s in starts if s >= t]
            if not nxt:
                break
            t = max(0.0, nxt[0] - rng.uniform(0.0, 0.8))
        L = min(clip_len(rng), total - t)
        if L < 5:
            break
        a, b = int(t * sr), int((t + L) * sr)
        write(out_dir, f"{prefix}_{k:03d}", pcm[a:b], sr, rows)
        used += L
        k += 1
        t += L + rng.uniform(5, 60)
    return used


def long_form(rng, files, budget_h, name, out_dir, rows, ts_key=True):
    got = 0.0
    files = list(files)
    rng.shuffle(files)
    for f in files:
        for batch in pq.ParquetFile(f).iter_batches(batch_size=4):
            for row in batch.to_pylist():
                if got >= budget_h * 3600:
                    return got
                pcm, sr = decode(row["audio"])
                starts = row.get("timestamps_start") if ts_key else None
                rid = os.path.splitext(os.path.basename(
                    row["audio"].get("path") or f"r{len(rows)}"))[0]
                per = min(budget_h * 3600 - got, rng.uniform(0.3, 0.7) *
                          len(pcm) / sr)
                got += windows(rng, pcm, sr, starts, per, f"{name}_{rid}",
                               out_dir, rows)
    return got


def joined(rng, utterances, budget_h, name, out_dir, rows, cap=300.0):
    """Join utterances (pcm, sr, group) into clips; a clip keeps to one
    group (speaker/chapter or language) with prob 0.7, mixes otherwise."""
    got, k = 0.0, 0
    by_group = {}
    for u in utterances:
        by_group.setdefault(u[2], []).append(u)
    groups = list(by_group)
    while got < budget_h * 3600 and groups:
        L = min(clip_len(rng, cap), cap)
        g = rng.choice(groups)
        pieces, dur, sr0 = [], 0.0, None
        mixed = rng.random() > 0.7
        while dur < L:
            src = by_group[rng.choice(groups) if mixed else g]
            pcm, sr, _ = src[rng.randrange(len(src))]
            if sr0 is None:
                sr0 = sr
            if sr != sr0:
                continue
            pieces.append(pcm)
            gap = np.zeros(int(sr * rng.uniform(0.2, 0.9)), np.float32)
            pieces.append(gap)
            dur += len(pcm) / sr + len(gap) / sr
        pcm = np.concatenate(pieces[:-1])
        write(out_dir, f"{name}_{k:05d}", pcm, sr0, rows)
        got += len(pcm) / sr0
        k += 1
    return got


def utterances(files, limit_s, rng, group_fn, max_rows=None):
    out, got = [], 0.0
    files = list(files)
    rng.shuffle(files)
    for f in files:
        for batch in pq.ParquetFile(f).iter_batches(batch_size=256):
            for row in batch.to_pylist():
                pcm, sr = decode(row["audio"])
                out.append((pcm, sr, group_fn(row)))
                got += len(pcm) / sr
                if got >= limit_s or (max_rows and len(out) >= max_rows):
                    return out
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hours", type=float, default=140.0)
    ap.add_argument("--eval-hours", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    train_dir = os.path.join(a.out, "train")
    eval_dir = os.path.join(a.out, "eval")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)
    train, ev = [], []
    S = a.src
    H, E = a.hours, a.eval_hours

    # Long-form, multi-speaker.
    ami = sorted(glob.glob(f"{S}/diarizers-community__ami/ihm/train-*.parquet"))
    ami_ev = sorted(glob.glob(f"{S}/diarizers-community__ami/ihm/test-*.parquet"))
    print("ami", long_form(rng, ami, H * MIX["ami"], "ami", train_dir, train) / 3600, flush=True)
    print("ami eval", long_form(rng, ami_ev[:1], E * 0.25, "amiev", eval_dir, ev) / 3600, flush=True)

    vc = sorted(glob.glob(f"{S}/diarizers-community__voxconverse/data/dev-*.parquet"))
    vc_ev = sorted(glob.glob(f"{S}/diarizers-community__voxconverse/data/test-*.parquet"))
    print("voxconverse", long_form(rng, vc, H * MIX["voxconverse"], "vc", train_dir, train) / 3600, flush=True)
    print("voxconverse eval", long_form(rng, vc_ev[:1], E * 0.2, "vcev", eval_dir, ev) / 3600, flush=True)

    ea = sorted(glob.glob(f"{S}/distil-whisper__earnings22/full/test-*.parquet"))
    print("earnings22", long_form(rng, ea[1:], H * MIX["earnings22"], "earn", train_dir, train, ts_key=False) / 3600, flush=True)
    print("earnings22 eval", long_form(rng, ea[:1], E * 0.15, "earnev", eval_dir, ev, ts_key=False) / 3600, flush=True)

    # Read speech, joined.
    ls = sorted(glob.glob(f"{S}/openslr__librispeech_asr/clean/train.100/*.parquet"))
    u = utterances(ls, H * MIX["librispeech"] * 3600 * 1.1, rng,
                   lambda r: f"{r['speaker_id']}-{r['chapter_id']}")
    print("librispeech", joined(rng, u, H * MIX["librispeech"], "ls", train_dir, train) / 3600, flush=True)
    del u
    ls_ev = sorted(glob.glob(f"{S}/openslr__librispeech_asr/clean/validation/*.parquet"))
    u = utterances(ls_ev, E * 0.15 * 3600 * 1.2, rng,
                   lambda r: f"{r['speaker_id']}-{r['chapter_id']}")
    print("librispeech eval", joined(rng, u, E * 0.15, "lsev", eval_dir, ev) / 3600, flush=True)
    del u

    per_lang = H * MIX["fleurs"] / len(FLEURS)
    for lang in FLEURS:
        fs = sorted(glob.glob(f"{S}/google__fleurs/parquet-data/{lang}/train-*.parquet"))
        if not fs:
            print("fleurs missing", lang, flush=True)
            continue
        u = utterances(fs, per_lang * 3600 * 1.1, rng, lambda r, l=lang: l)
        print("fleurs", lang, joined(rng, u, per_lang, f"fl{lang[:2]}", train_dir, train, cap=240) / 3600, flush=True)
        del u
        if lang in ("ru_ru", "cmn_hans_cn"):
            fv = sorted(glob.glob(f"{S}/google__fleurs/parquet-data/{lang}/validation-*.parquet"))
            u = utterances(fv, E * 0.1 * 3600 * 1.2, rng, lambda r, l=lang: l)
            print("fleurs eval", lang, joined(rng, u, E * 0.1, f"flev{lang[:2]}", eval_dir, ev, cap=180) / 3600, flush=True)
            del u

    sv = sorted(glob.glob(f"{S}/bond005__sova_rudevices/data/train-*.parquet"))
    if sv:
        u = utterances(sv, H * MIX["sova"] * 3600 * 1.1, rng, lambda r: "ru")
        print("sova", joined(rng, u, H * MIX["sova"], "sova", train_dir, train, cap=180) / 3600, flush=True)
        del u

    rng.shuffle(train)
    for name, rows in (("train", train), ("eval", ev)):
        with open(os.path.join(a.out, f"{name}.tsv"), "w") as f:
            for cid, path, _ in rows:
                f.write(f"{cid}\t{path}\n")
        print(name, len(rows), "clips", sum(r[2] for r in rows) / 3600, "h", flush=True)


if __name__ == "__main__":
    main()
