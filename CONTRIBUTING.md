# Contributing

The rules here are the ones this repository actually runs on, not a wish
list. They are short because most of what matters is in
[CLAUDE.md](CLAUDE.md) (the project's context file: architecture, numbers,
and the bugs that were expensive to find).

## The shape of a change

**One issue, one branch, one pull request.** Open the issue first, using the
[task template](.github/ISSUE_TEMPLATE/task.yml): *why* (what is wrong today
and what it costs, measured if it can be measured), *what* (the change as its
user sees it, and what must not regress), *how* (the approach and how it will
be verified). Someone who was not in the room should be able to pick it up.

Then a pull request that says `Closes #N`, with a CHANGELOG entry under
`## Unreleased` in the same PR. Cut releases with the workflow, never by
editing the version by hand — see [docs/RELEASING.md](docs/RELEASING.md).

## Code

C11, CUDA C in `.cu`. `snake_case`, `vv_` on every public symbol. Functions
return `vv_status_t`; there are no exceptions and no `errno` conventions.
`vv_alloc` / `vv_free`, never bare `malloc`. No allocations in the decode
loop. No globals beyond the documented ones. `vv_log()`, never `printf`, for
anything a user should see. Doxygen on public API, in English.

Comments explain *why*, and are worth writing where the answer is not local:
a numeric order that must not be re-associated, a lock that exists because of
a specific race, a constant that came from a measurement. Match the density
of the file you are editing.

The runtime is C. Python belongs in `tools/` only, for one-off conversion and
reference dumps — never in the inference path.

## Numbers

Claims about speed, memory or quality come with the machine, the input and
the command. "Faster" is not a result; "prefill 283 → 991 GFLOP/s on a 5900X,
best of three" is. If a change is supposed to leave output identical, say so
and show the comparison — several changes in this repository are bit-exact on
purpose, and `tools/compare_ref.py` exists to prove it against PyTorch.

## Tests

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Tests that need weights skip themselves without `VV_TEST_MODEL`, so a clean
`ctest` passes anywhere. Add a test that fails before your change and passes
after; if the change is bit-exact, assert that, not "close enough".

Pull requests build the container image and run CodeQL (`c-cpp`, `actions`,
`python`). Both must be green.

## Reporting

Bugs and ideas: an issue, with the template. Security problems: privately,
see [SECURITY.md](SECURITY.md).
