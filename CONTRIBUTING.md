# Contributing to Slate

Thank you for your interest in contributing. Slate is a small, opinionated codebase, so
we have a small, opinionated set of expectations. Reading this document end-to-end will
take less than ten minutes and will save reviewer cycles.

## Ground rules

1. **Stay within the layering.** Slate's L0–L10 stack only allows downward
   dependencies. A change to L3 (operators) must not pull in anything from L4 (modules)
   or above. If you find yourself needing to, that is a signal that an interface needs
   to be lifted into a lower layer.

2. **Every operator ships with a gradient check.** Any new `op_*_forward` /
   `op_*_backward` pair MUST have a corresponding entry in `tests/test_gradcheck.c`
   that compares the analytic backward against a finite-difference numerical gradient.
   The default tolerance is `1e-3` relative; if your op cannot meet that, write a
   comment explaining why.

3. **No third-party C/C++ dependencies in the core.** The `src/` tree must compile
   with nothing beyond a C11 standard library and pthreads / Win32 threads. Tools in
   `tools/` and tests are not bound by this rule.

4. **No silent allocations in hot paths.** Allocations during the forward or backward
   pass of training must go through an `slate_arena_t`. Calling `malloc` inside an op
   kernel will fail review.

5. **Disclosure of AI assistance.** Slate accepts AI-assisted contributions, but you
   must (a) be able to explain every line you submit to a reviewer without AI help,
   (b) take responsibility for maintaining the code you contribute, and (c) note in
   the PR description if a significant portion was AI-drafted. PRs that appear to be
   the verbatim output of a language model without human review will be closed.

## Development workflow

### Setting up

```bash
git clone https://github.com/eating-and-drinking/slate.git
cd slate
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug \
    -DSLATE_SANITIZE=address \
    -DSLATE_BUILD_TESTS=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

### Running tests

```bash
ctest --test-dir build --output-on-failure
# Or run individual binaries:
./build/tests/slate_test_gradcheck
./build/tests/slate_test_autograd
```

### Formatting

```bash
# requires clang-format 14+
find src include tests examples -name '*.c' -o -name '*.h' | xargs clang-format -i
```

CI will reject PRs that fail `clang-format --dry-run --Werror`.

## Submitting a change

### Issues

- **Bugs**: please file an issue using the bug-report template. Include the platform,
  compiler version, build options, and a minimal reproducer.
- **Feature requests**: please open a discussion before writing code. Slate is small
  by design; not every reasonable feature belongs in core.

### Pull requests

1. Open the PR against the `main` branch.
2. Use a present-tense, imperative summary line (`add Adafactor optimizer`, not
   `Added Adafactor optimizer`).
3. The PR description should answer three questions:
   - **What** does this change do?
   - **Why** is it needed (link an issue or design doc if applicable)?
   - **How** is it tested?
4. PR descriptions and review responses must be written by humans. We do not
   accept LLM-drafted prose in either.
5. CI must pass before review.

### Commit messages

```
add Adafactor optimizer

Adafactor reduces optimizer-state memory for Adam from 2x weights to
roughly 0x, by factoring the second-moment estimate over the matrix's
rows and columns instead of storing it per element. This is essential
for streaming-training of 7B-class models on 16 GB devices, where
AdamW's optimizer state alone would not fit even with sub-module
streaming.

Closes #N
```

## Code style cheatsheet

- C11, C++17 if needed (only for opt-in test harnesses)
- 4-space indentation, no tabs (except in Makefiles)
- 100-column soft limit
- `snake_case` for functions and variables
- `slate_` prefix for all public symbols
- `SLATE_` prefix for all public macros
- Opaque pointer pattern for objects with internal state (`typedef struct slate_foo
  slate_foo_t;`)
- Errors returned as `slate_status_t`, never via `errno` or by `return NULL` without
  context
- Comments describe **why**, not **what**

## Review expectations

Reviews focus on:

1. **Correctness** — does the test prove the behavior?
2. **Layering** — does the change respect L0–L10?
3. **Memory discipline** — does it allocate where it shouldn't?
4. **Honesty about scope** — does the PR description match the diff?

Reviewers will ask questions. Please respond to them in your own words. If you used an
AI tool to help understand the code, that is fine; if you used it to write your
response, please rephrase in your own voice before sending.

## Reporting security issues

Do **not** open a public issue for security vulnerabilities. See [`SECURITY.md`](SECURITY.md)
for the responsible-disclosure process.

## Code of conduct

This project follows the Contributor Covenant; see [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md).
