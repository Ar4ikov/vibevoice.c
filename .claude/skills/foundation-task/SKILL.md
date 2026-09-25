---
name: foundation-task
description: Run a large piece of TurboQwen work end to end — issue by the repo template, branch, implementation against the C rules, evidence-backed verification, PR with results, CI watched to green. Use for anything that adds a backend, a kernel family, a format or a CLI surface, rather than a one-line fix.
---

# Foundation task

A foundation task is one issue, one branch, one pull request, and one claim
per number in it. The repo's history is the standard to match: read the last
three merged PRs before writing yours.

## 1. The issue comes first

`.github/ISSUE_TEMPLATE/task.yml` asks for **Why / What / How**, and means it:

- **Why** — what is wrong today and what it costs, measured. "The encoder
  takes 180 ms of a 1.9 s request", not "the encoder is slow". If the cost
  cannot be measured yet, measure it before filing.
- **What** — the change as its user sees it: flags, formats, endpoints,
  behaviour. Say what is out of scope and what must not regress (transcripts
  are usually the thing that must not regress).
- **How** — the approach, where it lands in the tree, and how it will be
  verified. End with a literal **Done when:** line naming the tests, the
  benchmark and the reference.

```bash
gh issue create --repo Ar4ikov/TurboQwen --title "..." --body-file <(cat)
```

Branch from `master`, named for the work (`metal-backend`, `awq-fast-path`).
Another agent may be working the same repo from another machine: `git fetch`
before branching and before pushing, and keep to your own files.

## 2. While implementing

The rules that get code rejected here are in `CLAUDE.md` §6; the ones that
bite most often:

- C11, `snake_case`, `vv_` on every public symbol, `vv_status_t` back from
  every function, Doxygen on public headers, comments in English.
- No `malloc`/`free` (use `vv_alloc`/`vv_free`), no globals, no `printf` for
  errors (`vv_log`), no allocation or synchronisation in a hot path.
- Backends live behind `include/vibevoice/device.h`. One backend per binary;
  `src/device/device_none.c` is the shape a new one must fill. If an op has
  no equivalent on the hardware, **decline it** (return `VV_ERR_UNSUPPORTED`)
  and say so in the docs — do not emulate it silently.
- Summation order is a compatibility surface. A reduction that changes order
  moves timestamps; if a test compares bits, keep the order and say why in a
  comment.

## 3. Verification is the deliverable

A claim without a command that produced it does not go in the PR.

- `ctest --test-dir build --output-on-failure`, and again with
  `VV_TEST_MODEL=<checkpoint>` so the tests that need weights actually run.
- Transcripts: compare against the reference for the same checkpoint on
  `jfk.wav`, a 30 s file and a long file. Character-identical is the bar when
  the change is not meant to alter arithmetic; when it is, say which words or
  timestamps moved and by how much.
- Benchmarks: state the machine, the checkpoint, the file, and how many runs.
  Re-run anything surprising — on a shared or memory-pressured box the same
  binary can differ 2x (check `vm_stat` / `nvidia-smi` first).
- Run what you did not touch as well: the CPU path, the other quantisations,
  `serve`, `chat`, `mic`. A backend PR that never started the server is not
  finished.

## 4. The pull request

Body structure the repo uses:

```
Closes #N.

<the problem in one paragraph, with the number that motivated it>

### <What changed, in sections a reader can skim>

### Same bits as before      <- or: what moved, and by how much

### Results (machine, checkpoint, date)
| ... | before | after |

### How it was verified
### Not checked
```

**"Not checked" is mandatory.** List the platforms, checkpoints and paths you
could not run, so a reviewer knows the shape of the hole.

```bash
gh pr create --repo Ar4ikov/TurboQwen --base master --title "..." --body-file ...
```

Then watch it: `gh pr checks <N> --watch`. CI (`.github/workflows/container.yml`)
builds the Linux container on every PR; a Metal- or MSVC-only change still has
to leave that build green. Fix what it finds in the same branch — do not open
a second PR for your own red CI.

## 5. Finishing

- `CHANGELOG.md` gets an Unreleased entry; `README.md` and `CLAUDE.md` get the
  new flag, format or platform. `docs/` gets the design if the change has one
  (`docs/METAL.md`, `docs/BITNET.md`, `docs/STREAMING.md` are the models).
- Leave the numbers that surprised you in a memory, not only in the PR.
