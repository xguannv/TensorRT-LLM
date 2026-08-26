<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Sync a Rubin Branch and Import a PR as One Commit

Use this guide to rebase Rubin-only changes onto an exact `main` commit, import a PR as one
signed-off squash commit, and publish the compiled artifacts. Run commands from the repository
root. Read `AGENTS.md`, `CLAUDE.local.md` when present, and `CODING_GUIDELINES.md` before resolving
code conflicts.

## 1. Preserve local work

Inspect the branch and working tree before changing history:

```bash
git status --short --branch
git log --graph --decorate --oneline -30
```

Never use `git reset --hard` or a blanket clean. If tracked edits must be moved aside, stash only
those paths so untracked build artifacts remain untouched:

```bash
git stash push -m "preserve-before-rubin-sync" -- path/to/edited-file ...
```

Record the created stash and restore that exact stash only after the rebase and cherry-pick.

## 2. Rebase Rubin changes onto an exact main commit

Set the requested commit and new branch name:

```bash
BASE=d09405681f755d09d520f75a1b5378d7ed699610
NEW_BRANCH=rubin-v1.3.0rc22
```

Fetch main, verify the commit, and inspect the Rubin range:

```bash
git fetch origin main
git cat-file -e "${BASE}^{commit}"
OLD_BASE=$(git merge-base HEAD origin/main)
git log --graph --decorate --oneline --boundary "${OLD_BASE}..HEAD"
```

Confirm that the range contains only the Rubin commits that should move, then replay it:

```bash
git switch -c "${NEW_BRANCH}"
git rebase --onto "${BASE}" "${OLD_BASE}"
```

Resolve non-API conflicts against the newer main implementation, stage them, and run
`git rebase --continue`. API conflicts require the human-decision workflow in section 8. When one
resolution preserves both main and Rubin contracts without an interface tradeoff, take that compatible
union automatically; ask the human only when the API directions genuinely compete. Skip a commit only after verifying that its complete change
is already in the new base. Verify the result:

```bash
git merge-base --is-ancestor "${BASE}" HEAD
git log --graph --decorate --oneline "${BASE}..HEAD"
git diff --check "${BASE}..HEAD"
```

## 3. Fetch and verify the PR series

Set the PR and fork details. Dedicated remote-tracking refs avoid adding a persistent remote:

```bash
PR=14047
EXPECTED_COMMITS=22
PR_REPO=NVIDIA/TensorRT-LLM
PR_URL="https://github.com/${PR_REPO}/pull/${PR}"
FORK_URL=https://github.com/lowsfer/TensorRT-LLM.git
FORK_BRANCH=kvCacheManagerV2-cpp
PR_REF="refs/remotes/pr/${PR}"
FORK_REF="refs/remotes/lowsfer/${FORK_BRANCH}"

git fetch origin "refs/pull/${PR}/head:${PR_REF}"
git fetch "${FORK_URL}" "${FORK_BRANCH}:${FORK_REF}"
```

`PR_URL` must be the canonical PR page in the target repository, not a fork branch, compare page,
or bare `#<number>` reference. Verify that GitHub's PR ref and the fork branch agree and that the
series is linear:

```bash
test -n "$(git ls-remote origin "refs/pull/${PR}/head")"
test "$(git rev-parse "${PR_REF}")" = "$(git rev-parse "${FORK_REF}")"
PR_BASE=$(git merge-base origin/main "${PR_REF}")
test "$(git rev-list --count "${PR_BASE}..${PR_REF}")" -eq "${EXPECTED_COMMITS}"
test "$(git rev-list --merges --count "${PR_BASE}..${PR_REF}")" -eq 0
git log --reverse --oneline "${PR_BASE}..${PR_REF}"
```

Stop and investigate if the count or refs differ; the PR may have been updated.

## 4. Make one signed-off squash commit

```bash
TARGET_BRANCH=$(git branch --show-current)
git switch -c "pr-${PR}-squash" "${PR_BASE}"
git merge --squash "${PR_REF}"
git diff --cached --check
git diff --cached --stat
PR_TITLE='[None][feat] KVCacheManagerV2 C++ translation'
git commit -s \
    -m "${PR_TITLE} (#${PR})" \
    -m "Pull-Request: ${PR_URL}"
SQUASH_COMMIT=$(git rev-parse HEAD)
```

Use the actual PR title, number, and canonical URL. The `Pull-Request` trailer preserves a clickable,
unambiguous source even after the squash commit is cherry-picked or moved to another remote. Preserve the original sign-off if possible, otherwise let
Git add the DCO sign-off through `-s`; never type it manually. Let all hooks finish. If hooks modify
files, re-stage them and commit again.

For merge-request from gitlab, preserve the original commit message, merge-request number and DCO sign-off, drop the merge commit , keep only the actual change commit.

## 5. Cherry-pick onto Rubin

```bash
git switch "${TARGET_BRANCH}"
git cherry-pick "${SQUASH_COMMIT}"
```

Preserve the PR's behavior while adapting conflicts to newer Rubin/main APIs. Before continuing,
check conflict markers and syntax-check resolved Python files:

```bash
rg -n '^(<<<<<<<|=======|>>>>>>>)'
python3 -m py_compile path/to/resolved.py ...
git diff --check
git add path/to/resolved-file ...
git cherry-pick --continue
```

Restore the exact stash created in step 1 and verify the result:

```bash
git stash list
git stash pop stash@{0}
git status --short --branch
git show -s --format='%H%n%P%n%B' HEAD
test "$(git show -s --format='%(trailers:key=Pull-Request,valueonly)' HEAD)" = "${PR_URL}"
git rev-list --count "${PR_BASE}..${PR_REF}"
git rev-list --merges --count "${PR_BASE}..${PR_REF}"
```

Do not push unless requested. Push branches to the user-specified fork, normally `origin`; PRs
target the NVIDIA repository.

## 6. Locate and attribute a commit

Start with the exact object rather than searching only by subject. Record its graph position,
changed paths, and every branch that contains it:

```bash
COMMIT=8ddfaf7f2866f22c5b83a9693d4b2d256e074592

git cat-file -e "${COMMIT}^{commit}"
git show -s --format='commit=%H%nparents=%P%nauthor=%an <%ae>%ndate=%aI%nsubject=%s%nbody=%b' \
    "${COMMIT}"
git rev-list --children --all | rg "^${COMMIT} "
git branch -a --contains "${COMMIT}"
git show --stat --name-status "${COMMIT}"
```

The parent is in `%P`; the line from `git rev-list --children` identifies direct children. Branch
containment answers where the commit is reachable, while `--name-status` answers where its change
lives in the source tree.

For a rebased, cherry-picked, or rewritten commit, search for candidates by subject, then compare
the actual patches. A matching title is evidence for a candidate, not proof of equivalence:

```bash
SUBJECT=$(git show -s --format=%s "${COMMIT}")
git log --all --format='%H %D %s' --fixed-strings --grep="${SUBJECT}"

CANDIDATE=<candidate-commit>
git range-diff "${CANDIDATE}^!" "${COMMIT}^!"
git show "${CANDIDATE}" | git patch-id --stable
git show "${COMMIT}" | git patch-id --stable
```

Equal stable patch IDs establish patch equivalence for ordinary non-merge commits. Different patch
IDs do not necessarily mean unrelated work: conflict resolution or adaptation to a new base may
change the patch. Use `git range-diff` and inspect the changed paths in that case. To check whether
the change is already represented on GitHub `main`, refresh the remote before using `git cherry`:

```bash
git fetch origin main
git cherry -v origin/main HEAD
```

`-` means Git found a patch-equivalent commit upstream; `+` means it did not. This is a patch test,
not a PR lookup.

For the internal `ftp/tekit` GitLab project, use the commit-to-merge-request API. Keep shell tracing
disabled and never print the token:

```bash
set +x
source ~/.bashrc
test -n "${GLPAT:-}"

GITLAB_PROJECT_ID=68442
GITLAB_API=https://gitlab-master.nvidia.com/api/v4
curl --fail-with-body --silent --show-error \
    --header "PRIVATE-TOKEN: ${GLPAT}" \
    "${GITLAB_API}/projects/${GITLAB_PROJECT_ID}/repository/commits/${COMMIT}/merge_requests" \
    | jq -r '.[] | "!\(.iid)\t\(.state)\t\(.title)\t\(.web_url)"'
```

An empty array means GitLab has no MR directly associated with that SHA. If the commit message names
an original source SHA, repeat the query with that SHA. For a squash or heavily reorganized commit,
there may be no authoritative one-to-one MR mapping; report constituent MRs only after comparing
their commits and patches.

## 7. Compile and publish

`compile.sh` builds the SBSA Rubin wheel and container image under `build_images/<short-commit>/`,
then copies both to
`/lustre/share/coreai_comparch_trtllm/lizhiz/rubin_dsv4/<short-commit>/`:

```bash
./compile.sh batch-spx
```

The first argument selects the Slurm partition. The optional second argument is forwarded to
`build_wheel.py`, for example `./compile.sh batch-spx --clean`. Override the destination when needed:

```bash
PUBLISH_ROOT=/another/shared/path ./compile.sh batch-spx
```

Verify the publication:

```bash
HASH=$(git rev-parse --short HEAD)
find "/lustre/share/coreai_comparch_trtllm/lizhiz/rubin_dsv4/${HASH}" \
    -maxdepth 1 -type f \( -name '*.sqsh' -o -name '*.whl' \) -print
```

## 8. Human decisions and live conflict ledger

### API-conflict workflow

Treat renamed, added, removed, reordered, or reinterpreted parameters; public data-structure
changes; caller/callee ownership changes; and competing interface evolution as API conflicts.
For every API conflict:

1. Stop before editing or staging the API and compare the Rubin source tip, selected `main`, and
   rebased worktree across declarations, definitions, bindings or schemas, active callers, and tests.
2. If a method preserves both main and Rubin behavior and signatures without an API tradeoff, take
   that compatible union automatically. Record why it is compatible before staging; no human choice
   is required.
3. If the directions genuinely compete (including changed semantics, ownership, incompatible
   signatures, or removal versus retention), show the human the choices and consequences and do not
   edit or stage until the human chooses.
4. Record every result below, keyed by repository-relative filename and function names. For a
   compatible union, record `Human direction: not required - compatible union`; otherwise record the
   human choice.
5. Apply only the recorded direction, validate the whole contract, and immediately update the
   entry with the exact edit and evidence.

Maintain this ledger during the sync. Record every conflict decision, including non-API conflicts.
Do not rewrite history when a resolution changes: mark the old resolution `superseded` and append
its replacement.

Each entry must contain: conflict, human direction, resolution, evidence, and status. Use a
directory path only for an inseparable generated-file bundle.

### Sync: `user/lizhiz/rubin-dsv4-k3-latest` onto `origin/main` (2026-08-18)

Context: main `0eeda343df8cf44e743d00d1b666b08a43fde83e`; DSV4 source
`gitlab/user/lizhiz/rubin-dsv4-latest` at `bb0249bb`; old base
`43d6fa410c3296b364c5ea762248c223940c8785`. `gitlab/feat/rubin-k3-latest` is an ancestor of the
selected DSV4 source.

#### `tensorrt_llm/_torch/attention_backend/sparse/dsa.py`

- Conflict: Rubin modified the old monolithic module; main replaced it with `sparse/dsa/`.
- Human direction: not an API conflict.
- Resolution: keep the main package migration and port still-needed Rubin behavior into the package.
- Evidence: the package is the active post-rebase import layout.
- Status: applied; branch guard pending.

#### `tensorrt_llm/_torch/attention_backend/sparse/dsa/indexer.py`

- Conflict: Rubin's final DSV4 workspace reservation lived in the removed monolithic module.
- Human direction: not an API conflict.
- Resolution: port the full-workspace reservation into main's package implementation.
- Evidence: Python syntax validation passed.
- Status: applied; branch guard pending.

#### `tensorrt_llm/_torch/attention_backend/sparse/params.py`

- Conflict: the package port needs the workspace-budget helper in the shared sparse layer.
- Human direction: not an API conflict.
- Resolution: place the helper in this shared package file.
- Evidence: Python syntax validation passed.
- Status: applied; branch guard pending.

#### `cpp/tensorrt_llm/kernels/trtllmGenKernels/fmha/cubin/`

- Conflict: the rebase produced a committed FMHA inventory exactly matching DSV4 (11,761 files) and
  containing all Rubin/K3 files, but it omits 836 files from current main. The omitted set includes
  SM100f refreshes associated with main bug fixes such as `b4b92c9f5f`.
- Human direction: union the DSV4/Rubin inventory with all 836 current-main-only SM100f files.
- Resolution: supersede the earlier blanket deletion choice. Restore the main-only paths on top of
  the complete DSV4 inventory, materialize their LFS objects, and validate the combined bundle.
- Evidence: after application at HEAD `f44bea29e2`, inventory is 12,597; main-only, DSV4-only, and
  Rubin/K3-only versus HEAD are all zero. All 12,597 LFS objects materialized; restored pointer-stub
  count is zero. Obsolete DSV4-only build job `436918` was canceled.
- Status: applied; replacement full build and branch guard pending.

#### `cpp/tensorrt_llm/kernels/trtllmGenKernels/blockScaleMoe/runner.cu`

- Conflict: an initial old-DSV4 bundle restoration appeared to remove SiTu support; history showed
  SiTu is supported on Rubin/K3 and main.
- Human direction: preserve Rubin SiTu support and use the newer-main-first union.
- Resolution: supersede the old-bundle restoration. Keep the standalone DeepSeek-FP8 SiTu rejection
  while preserving SiTu through fused native BMM cubins.
- Evidence: Rubin/K3 and main history; coherent BMM bundle compiled.
- Status: compile-validated; branch guard pending.

#### `cpp/tensorrt_llm/kernels/trtllmGenKernels/blockScaleMoe/batchedGemm/`

- Conflict: mixing individual generated files caused an incoherent schema with missing rank,
  world-size, multicast, and slice-K members.
- Human direction: preserve Rubin SiTu/SM107 support using a coherent union, not the older DSV4
  bundle.
- Resolution: restore the complete `gitlab/feat/rubin-k3-latest` bundle and materialize its LFS
  objects. Its inventory contains all BMM paths from DSV4 and main.
- Evidence: 1,288 files; BMM extraction and aggregator/runner compilation succeeded.
- Status: compile-validated; branch guard pending.

#### `cpp/tensorrt_llm/thop/attentionOp.cpp`

- Conflict: HEAD mixes Rubin's `compressed_kv_cache_pool_ptr`, `uses_spcompress`, and `ugpu_id` with
  main's `aux_kv_cache_pool_ptr` body references and newly used `max_num_sequences`. Active Python
  callers supply main's generalized pointer name and `max_num_sequences`; Rubin tips retain the old
  pointer name. Main commit `a87741ef3d` intentionally performed the generalizing rename.
- Human direction: use main's `aux_kv_cache_pool_ptr` and optional `max_num_sequences`, while
  preserving Rubin's `uses_spcompress`, `ugpu_id`, DSV4 epilogue, rope, and KV-norm parameters.
- Resolution: rename the remaining implementation-side compressed-pool parameter and uses to main's
  generalized name, add the optional maximum-sequence parameter, and retain every Rubin/DSV4-only
  parameter and behavior.
- Evidence: job `436847` failed on undefined `aux_kv_cache_pool_ptr` and `max_num_sequences`; all
  contract layers and rename history were compared. Implementation/build validation pending.
- Status: applied; compile and branch guard pending.

#### `cpp/tensorrt_llm/thop/attentionOp.h`

- Conflict: HEAD's header uses `aux_kv_cache_pool_ptr` but lacks `max_num_sequences`, disagreeing with
  the implementation and active Python caller contract.
- Human direction: use main's `aux_kv_cache_pool_ptr` and optional `max_num_sequences`, while
  preserving Rubin's `uses_spcompress`, `ugpu_id`, DSV4 epilogue, rope, and KV-norm parameters.
- Resolution: add the optional main parameter to this declaration; retain the existing main pointer
  name and every Rubin/DSV4-only parameter.
- Evidence: direct Rubin/K3, DSV4, main, and HEAD comparison; implementation/build validation pending.
- Status: applied; compile and branch guard pending.


#### `cpp/tensorrt_llm/nanobind/thop/bindings.cpp`

- Conflict: after the approved attention API merge, the nanobind registration still exported Rubin's
  old `compressed_kv_cache_pool_ptr` keyword and had no annotation for main's optional
  `max_num_sequences`, causing job `436938` to fail the argument-count static assertion.
- Human direction: inherited from the explicit attention API decision above: use main's generalized
  `aux_kv_cache_pool_ptr` and optional `max_num_sequences`, while preserving Rubin/DSV4-only API
  arguments and their ordering.
- Resolution: rename only the exported pool-pointer keyword and insert the optional maximum-sequence
  annotation immediately after `force_prepare_spec_dec_tree_mask`, matching the merged C++ signature.
- Evidence: direct comparison of HEAD, `origin/main`, and `gitlab/feat/rubin-k3-latest`; job `436938`
  reached 847/887 before failing at the nanobind annotation-count check.
- Status: applied; compile and branch guard pending.

#### `cpp/tensorrt_llm/kernels/communicationKernels/moeAlltoAllKernels.cu`

- Initial conflict: main changed `check_timeout` from a fixed/single-argument check to a phase-aware
  `(start, budget)` API, while Rubin call sites still used the older form.
- Initial human direction: preserve main's dynamic timeout semantics while retaining Rubin's
  all-to-all paths.
- Initial resolution: keep the two-argument main API, restore `kDefaultTimeoutCycles`, and pass
  `ptrs.timeout_cycles` at every Rubin call site.
- Follow-up API conflict: the same merge retained main's declaration and host call sites for
  `moeA2AGetTimeoutCycles(bool)` but omitted its implementation, so job `436976` failed while linking
  `libth_common.so`.
- Human direction: option 1 selected. Preserve main's phase-aware timeout contract by restoring its
  resolver implementation; do not revert the API to Rubin's fixed timeout behavior.
- Follow-up resolution: restore main's exact `moeA2AGetTimeoutCycles(bool)` implementation alongside
  Rubin's all-to-all paths. This completes the already-retained declaration, warmup state, host call
  sites, parameter budgets, and two-argument device checks.
- Evidence: main commit `6ba3de1cbd` introduced the resolver, declaration, warmup state, host call
  sites, parameter budgets, and two-argument device checks as one bug-fix contract. Rubin/K3 and DSV4
  contain none of that contract; job `436976` proved the resolver definition was the only missing link
  symbol.
- Status: option 1 applied; full build and branch guard pending.
Append or update an entry immediately whenever the next conflict is inspected, decided, applied, or
validated.

### Redo sync: `user/lizhiz/rubin-dsv4-k3-latest` onto `origin/main` (2026-08-18)

Context: previous sync attempt retained on `backup/rubin-dsv4-k3-pre-redo-20260818` at
`ce5123dc4dd7bac8e658b6a28f2b2c3a43a12e49`. The redo uses freshly fetched main
`b417fc5454a0bd1d355d791c415ebe0e64ff5544`, DSV4 source
`bb0249bbf55916d44d94f75512aa59e0e5810642`, and old base
`43d6fa410c3296b364c5ea762248c223940c8785`.

- Procedure decision: the earlier attempt is superseded because textual conflict resolution was not
  followed by a complete cross-file contract audit before compilation. During this redo, inspect
  declarations, definitions, bindings or schemas, active callers, tests, and build registration
  before staging each API resolution and before every `git rebase --continue`.
- Existing human API directions above remain recorded and will be reapplied only after fresh
  comparison against the new main tip. Any newly changed direction must be appended, not substituted.
- Status: branch recreated at the source tip; rebase pending.

#### `cpp/cmake/modules/cuda_configuration.cmake`

- Conflict: Rubin enabled automatic SM107 selection at CUDA 13.3 while fresh main now recognizes SM107
  at CUDA 13.4; explicit architecture selection is separate from this default list.
- Human direction: non-API conflict; follow the documented newer-main implementation.
- Resolution: keep main's CUDA 13.4 threshold. Preserve Rubin SM107 support through the explicit
  architecture lists and the branch build's `107-real` selection.
- Evidence: main commit `9f92665449` intentionally introduced SM107 recognition with the 13.4
  threshold; the prior branch build configured explicit `107-real` successfully.
- Status: applied and static-reviewed in rebase commit 1/76.

#### `cpp/include/tensorrt_llm/common/cudaUtils.h`

- Conflict: Rubin used `optional.value_or(getSMVersion())`, which eagerly queries a GPU even when the
  caller supplied an SM value; main replaced it with an explicit `has_value()` branch.
- Human direction: non-API conflict; follow the newer-main implementation while retaining the full
  SM100-family range that includes SM107.
- Resolution: keep main's implementation and signature unchanged.
- Evidence: main commit `9f92665449` documents the no-device failure avoided by the newer code; all
  active callers remain compatible.
- Status: applied and static-reviewed in rebase commit 1/76.

#### `requirements.txt`

- Conflict: Rubin added TensorRT and OpenCV but loosened Torch and Triton and pinned an older
  FlashInfer, while main added current compatibility bounds and newer package pins.
- Human direction: non-API conflict; preserve Rubin-only dependencies without discarding newer-main
  dependency constraints.
- Resolution: keep main's Torch range, FlashInfer 0.6.16, and Triton 3.7.0 pin; add Rubin's
  `tensorrt~=10.16.1` and `opencv-python-headless` entries.
- Evidence: direct main/source/final-source comparison and the Rubin image's local-version Torch and
  Triton packages.
- Status: applied and static-reviewed in rebase commit 1/76.

#### `docker/common/install_base.sh`

- Conflict: the Rubin commit silently added an early unguarded mlx5 symlink plus duplicate libzmq,
  ccache, and CMake setup; fresh main now installs those dependencies and repairs mlx5 through package
  reinstallation.
- Human direction: non-API conflict; retain newer main's safer, idempotent bootstrap.
- Resolution: drop the redundant Rubin block and keep this file at fresh main for this commit.
- Evidence: full `init_ubuntu` comparison against `origin/main` and DSV4 final source.
- Status: applied and static-reviewed in rebase commit 1/76.

#### `docker/common/install_cuda_libs.sh`

- Conflict: a clean textual merge would silently retain Rubin's disabling of the public CUDA-library
  replacement block while main has since added CUDA 13.4 cuBLAS layout compatibility handling.
- Human direction: non-API toolchain choice; preserve the Rubin base-image contract.
- Resolution: retain Rubin's disabled replacement block because the same commit switches
  `Dockerfile.multi` to the internal Rubin development image; do not overwrite its platform CUDA
  stack with public packages.
- Evidence: final DSV4 and latest Rubin tips both retain this behavior, and Docker invokes the block
  after selecting the Rubin base image.
- Status: applied and static-reviewed in rebase commit 1/76.

#### `cpp/tensorrt_llm/runtime/CMakeLists.txt`

- Conflict: commit 1 registers `ugpu/ugpu_internal/ugpu_utils.cpp` before the runtime commit adds it;
  later source commit `0d7b5e2eec` moves that implementation to `ugpu/ugpu_utils.cpp` and updates CMake.
- Human direction: non-API sequencing issue; preserve the source series and require the final static
  audit to verify the post-refactor path exists.
- Resolution: retain the registration at this replay point; validate it again when the owning runtime
  and refactor commits are replayed.
- Evidence: add history points to `2c5ea89a03`; delete/move history points to `0d7b5e2eec`; final DSV4
  source registers the existing public path.
- Status: intentionally deferred within the replay series; final audit required.

#### `cpp/tensorrt_llm/thop/CMakeLists.txt`

- Conflict: commit 1 registers `ugpuAllocator.cpp` before runtime commit `2c5ea89a03` adds it.
- Human direction: non-API sequencing issue; preserve the source series and verify the file when its
  owning commit is replayed.
- Resolution: retain the registration and mark its existence check deferred.
- Evidence: final DSV4 source contains the file; add history points to `2c5ea89a03`.
- Status: intentionally deferred within the replay series; owning-commit audit required.

#### `cpp/tests/unit_tests/runtime/CMakeLists.txt`

- Conflict: commit 1 registers `ugpuLocalizationTest.cu` before test commit `b143262054` adds it.
- Human direction: non-API sequencing issue; preserve the source series and verify the test source at
  its owning commit.
- Resolution: retain the registration and mark its existence check deferred.
- Evidence: final DSV4 source contains the file; add history points to `b143262054`.
- Status: intentionally deferred within the replay series; owning-commit audit required.

#### `cpp/tensorrt_llm/kernels/trtllmGenKernels/batchedGemm/` (redo commit 2)

- Conflict: Replaying `629f900081` produces 603 unmerged paths in batched GEMM: 595 cubins and eight source/config/Python paths across the commit. The coherent Rubin BMM tree contains every fresh-main BMM path plus 627 Rubin-only paths. Initial inspection incorrectly treated `mDtypeSfC` as absent from that coherent tree; the deeper declaration/metadata audit showed that immutable `3c0b90f9d0` independently carries the complete contract even though main commit `1dfb7b134562934783c0e63dc6cb4a0ac462dc87` is not its ancestor.
- Function/symbol scope: `mDtypeSfC` plus the Rubin SM107, Lamport, slice-K, SiTu, and related option/trait/parameter/metadata contracts.
- Human direction: User chose option 1: newer-main-first contract union. Retain the coherent Rubin feature bundle and port the complete fresh-main `mDtypeSfC` contract.
- Resolution: Restore the BMM directory atomically from immutable coherent tree `3c0b90f9d0` (`e08e093137dbd057276105ef82617827c4b1eb80`). Do not mix main cubins with Rubin metadata or splice generated aggregates. This tree retains the full Rubin SM107/Lamport/slice-K/SiTu schema and already ports main's full `mDtypeSfC` constructor, member, dump, defaulting, validation, warning, cubin, and metadata contract.
- Evidence: `origin/main` has 661 BMM paths (639 cubins); the coherent tree has 1,288 paths (1,266 cubins), including all main paths and 627 additions. Its source has all 15 main `mDtypeSfC` contract occurrences, plus `mEnablesGlobalPtxKnobs`, `mNumSlicesForSliceK`, `mNumStagesTmemA`, `mSliceK`, `mPatchPzo`, and `mUseUniqueSfA`. The staged BMM tree has zero diff from `3c0b90f9d0`, 685 metadata `mDtypeSfC` initializers, 135 SiTu metadata occurrences, zero unmerged paths, and zero worktree LFS pointer stubs after hydration.
- Status: applied, staged, and static-reviewed in rebase commit 2/76.

#### `tensorrt_llm/_torch/custom_ops/cute_dsl_custom_ops.py` (redo commit 2)

- Conflict: Fresh main contains active Blackwell MLA decode operator registrations, while Rubin adds active SM107 CuTe DSL and uGPU operators and registrations.
- Function/symbol scope: `cute_dsl_mla_decode_fp8_blackwell`, `cute_dsl_mla_decode_fp16_blackwell`, Rubin NVFP4/BF16/grouped/finalize operators, and uGPU inplace operators.
- Human direction: User chose option 1: contract union. Retain both fresh-main Blackwell MLA registrations and Rubin SM107 CuTe DSL and uGPU operators.
- Resolution: Keep the clean three-way merge before the conflicting tail, then append the complete fresh-main Blackwell MLA block followed by the complete Rubin SM107/uGPU block. Git had interleaved similarly named runner methods when treating the independent append blocks as one conflict; raw union was rejected after syntax validation caught an unmatched signature.
- Evidence: exactly one FP8 and one FP16 Blackwell MLA definition remain; both active callers under `attention_backend/fmha/cute_dsl_mla.py` resolve. Rubin SM107 and uGPU operators remain present. `python3 -m py_compile` passes and no conflict marker remains.
- Status: applied, staged, and static-reviewed in rebase commit 2/76.

#### `tensorrt_llm/_torch/modules/linear.py` (redo commit 2)

- Conflict: Fresh main adds NVFP4 Marlin backend behavior, current LoRA result handling, and fused-bias arguments, while Rubin adds uGPU weight-loading/layout/planner/runtime behavior, inplace execution, and the optional `output_tensor` apply API.
- Function/symbol scope: `LoraLayer`, `add_lora_result`, Marlin selection/execution, fused bias, `split_indice_for_ugpu`, uGPU inplace execution, and `Linear.apply_linear(..., output_tensor=None)`.
- Human direction: User chose option 1: contract union. Retain fresh-main Marlin, LoRA, and fused-bias behavior together with Rubin uGPU loading, execution, and output API.
- Resolution: Union the independent imports and helper definitions; retain main's `_uses_marlin_nvfp4_backend`, fused-bias control, and `add_lora_result` semantics while retaining Rubin's uGPU imports, `split_indice_for_ugpu`, partitioned weight lifecycle, `_run_linear_ugpu`, and optional apply output parameter. Remove the duplicate `LoraLayer` import introduced by the mechanical union.
- Evidence: both `split_indice_for_ugpu` call sites remain; Marlin selection and `add_lora_result` remain; uGPU execution and weight-shard paths remain; `python3 -m py_compile` passes and no conflict marker remains.
- Status: applied, staged, and static-reviewed in rebase commit 2/76.

#### `cpp/tensorrt_llm/nanobind/thop/bindings.cpp` / `initBindings` (redo commit 3)

- Conflict: fresh main extends the public attention binding with `aux_kv_cache_pool_ptr`,
  `max_num_sequences`, `kv_norm_weight`, and `kv_norm_eps`; the Rubin FMHA commit adds
  `uses_spcompress`, `compressed_kv_cache_pool_ptr`, and `ugpu_id` in the same positional API.
- Function/symbol scope: `initBindings`, `attention`, and the exact nanobind argument order.
- Human direction: User chose option 1: newer-main-first contract union, retaining every fresh-main argument and
  adding the Rubin arguments only where the C++ declaration and Python callers support them.
- Evidence: fresh main's DSV4 backend actively populates `SparseRuntimeParams.aux_kv_cache_pool_ptr`
  and fused KV norm arguments; Rubin actively consumes `uses_spcompress` and `ugpu_id`.
- Status: applied, staged, and static-reviewed in rebase commit 3/76.

#### `cpp/tensorrt_llm/thop/attentionOp.cpp` / `attention`, `AttentionOp::run` (redo commit 3)

- Conflict: fresh main carries the DSV4 sparse-pool, sequence-capacity, inverse-rope, epilogue,
  quant-scale, and KV-norm contract, while Rubin adds SM107 spcompress validation and uGPU-aware
  operator selection. `aux_kv_cache_pool_ptr` and Rubin's `compressed_kv_cache_pool_ptr` both feed
  `mRuntimeSparseAttentionParams.sparse_kv_cache_pool` and therefore appear to be two names for the
  same semantic slot rather than independent pools.
- Function/symbol scope: `attention`, `AttentionOp::run`, `TrtllmGenAttentionOp::run`,
  `get_attention_op`, `mUsesSpcompress`, `aux_kv_cache_pool_ptr`,
  `compressed_kv_cache_pool_ptr`, `ugpu_id`, `max_num_sequences`, and `kv_norm_weight`.
- Human direction: User chose option 1: retain the complete fresh-main API and behavior, port Rubin's
  `uses_spcompress` validation and `ugpu_id` routing, and normalize the compressed-pool caller onto
  main's `aux_kv_cache_pool_ptr` instead of exposing two aliases for one pointer.
- Evidence: both conflict sides assign their differently named pointer to the same
  `sparse_kv_cache_pool` field; fresh-main Python callers currently pass only the auxiliary name.
- Status: applied, staged, and static-reviewed in rebase commit 3/76.

#### `tensorrt_llm/_torch/attention_backend/fmha/flashinfer_trtllm_gen.py` / `_trtllm_gen_batch_*_with_kv_cache`, `FlashInferTrtllmGenFmha.is_supported` (redo commit 3)

- Conflict: the Rubin commit updates positional calls for new FMHA ABI slots (`use_fp16_softmax`
  and `uses_spcompress`) but also replaces main's `is_sm_100f` capability helper with an explicit
  SM100/SM103 list.
- Function/symbol scope: `_trtllm_gen_batch_decode_with_kv_cache`,
  `_trtllm_gen_batch_context_with_kv_cache`, and `FlashInferTrtllmGenFmha.is_supported`.
- Human direction: User chose option 1: keep fresh-main capability detection and port only the required new ABI
  placeholders after verifying the rebased generated-op signature.
- Evidence: the source commit's functional additions are three positional placeholders; its SM
  predicate change would regress the newer main helper rather than add Rubin SM107 support.
- Status: applied, staged, and static-reviewed in rebase commit 3/76.

#### `tensorrt_llm/_torch/attention_backend/sparse/rocket/kernels.py` / `triton_bmm` (redo commit 3)

- Conflict: Rubin modifies the former monolithic `sparse/kernel.py`; fresh main has reorganized that
  code into the Rocket package, producing a whole-file conflict during rename mapping.
- Function/symbol scope: `triton_bmm` launch configuration.
- Human direction: User chose option 1: retain the current main Rocket module and port the narrow Rubin rule
  `num_warps=8` for SM107 (`4` otherwise), without restoring the obsolete monolithic layout.
- Evidence: the original Rubin commit changes only imports and this one launch parameter in the
  old sparse kernel file.
- Status: applied, staged, and static-reviewed in rebase commit 3/76.

#### `tensorrt_llm/_torch/attention_backend/sparse/skip_softmax/params.py` / `SkipSoftmaxScheduler.get_runtime_params` (redo commit 3)

- Conflict: fresh main replaced Rubin's `get_kernel_params` return-only API with
  `get_runtime_params`, which updates the shared `SparseRuntimeParams`; Rubin adds
  `uses_spcompress` propagation and a `SkipSoftmaxParams.uses_spcompress` option.
- Function/symbol scope: `SkipSoftmaxScheduler.get_runtime_params`, legacy `get_kernel_params`,
  `SkipSoftmaxKernelParams.uses_spcompress`, and `SkipSoftmaxParams.uses_spcompress`.
- Human direction: User chose option 1: preserve main's runtime-parameter API, add Rubin's `uses_spcompress` field
  and propagate it through that current API; do not resurrect the obsolete method unless an active
  caller remains after the merge.
- Evidence: current main callers use `sparse_runtime_params`; the clean-merged fallback still referred to removed
  `skip_softmax_kernel_params`, so the source behavior must be adapted to the current runtime object.
- Status: applied, staged, and static-reviewed in rebase commit 3/76.

#### `tensorrt_llm/_torch/attention_backend/trtllm.py` / `TrtllmAttentionMetadata`, `TrtllmAttention.forward` (redo commit 3)

- Conflict: fresh main has newer sparse-runtime and DSV4 attention paths, while Rubin adds uGPU
  request/token partitioning, per-uGPU block offsets/workspaces, slice builders, and spcompress
  propagation through the former skip-softmax API.
- Function/symbol scope: `TrtllmAttentionMetadata._ensure_ugpu_workspaces`,
  `build_ugpu_block_offsets`, `_get_generation_only_token_splits`, `_build_ugpu_slices`,
  `_prepare_ugpu_mla_gen_state_from_full`, and `TrtllmAttention.forward`.
- Human direction: User chose option 1: port Rubin's uGPU metadata and slicing features onto the current main class,
  while adapting spcompress to main's `SparseRuntimeParams` contract and preserving all main DSV4
  paths.
- Evidence: Rubin's additions are active infrastructure used throughout its later uGPU attention
  flow; main's `get_runtime_params` call and DSV4 runtime objects have live consumers.
- Status: applied, staged, and static-reviewed in rebase commit 3/76.

#### `tensorrt_llm/_torch/modules/mla.py` / `MLA` projection and context paths (redo commit 3)

- Conflict: the conflict sides differ by about 1,400 lines because fresh main refactored DSA/DSV4
  sparse MLA into dedicated backend modules, while Rubin's older monolithic file still contains
  those paths. The actual source commit adds SM107 NVFP4 backend allowance, strided per-head KV-B
  projection helpers, and Rubin-specific BF16 BMM selection.
- Function/symbol scope: `MLA.__init__`, `_use_kvbproj_strided_out`,
  `_get_kvbproj_head_weights`, `forward_context_default`, and `forward_context`.
- Human direction: User chose option 1: retain the fresh-main refactored MLA structure and port only the actual
  Rubin additions into their current equivalents; do not restore superseded monolithic DSA/DSV4
  implementations.
- Evidence: the source commit's semantic diff is narrow despite the large three-way conflict, and
  fresh main has active dedicated modules under `attention_backend/sparse/deepseek_v4` and `dsa`.
- Status: applied, staged, and static-reviewed in rebase commit 3/76.

#### Commit 3 applied-resolution addendum: `cpp/tensorrt_llm/nanobind/thop/bindings.cpp` / `initBindings`

- Resolution: preserve the complete main binding order and names, insert Rubin's
  `uses_spcompress` and `ugpu_id`, and expose only main's `aux_kv_cache_pool_ptr`.
- Evidence: the binding order now matches `attentionOp.h`; `max_num_sequences`,
  `kv_norm_weight`, and `kv_norm_eps` remain; the obsolete compressed-pointer alias is absent.

#### Commit 3 applied-resolution addendum: `cpp/tensorrt_llm/thop/attentionOp.cpp` / `attention`, `get_attention_op`

- Resolution: retain main's runner API and DSV4 arguments, add Rubin's top-level spcompress and uGPU
  arguments, validate spcompress only on SM107 FP8 context attention, include `ugpu_id` in the
  cached-op key, and pass the single auxiliary sparse-pool pointer through main's runner path.
- Evidence: the Rubin parameters were not threaded through `RunnerBase::run` because neither is
  consumed there; `mUsesSpcompress` is configured before the runner call, and `ugpu_id` is used
  only for operator selection/cache identity. No compressed-pointer alias remains.

#### Commit 3 applied-resolution addendum: `tensorrt_llm/_torch/attention_backend/fmha/flashinfer_trtllm_gen.py`

- Resolution: keep the fresh-main file unchanged. Its refactored FlashInfer wrappers already own the
  current positional ABI, and main's `is_sm_100f` helper correctly includes SM107.
- Evidence: the old Rubin patch targeted wrapper functions no longer present in main; applying its
  explicit SM100/SM103 predicate would exclude Rubin.

#### Commit 3 applied-resolution addendum: `tensorrt_llm/_torch/attention_backend/sparse/rocket/kernels.py` / `triton_bmm`

- Resolution: keep the Rocket package layout and add only Rubin's SM107 launch rule:
  eight warps on SM107 and four otherwise.
- Evidence: this is the complete semantic delta from the source commit's former
  `sparse/kernel.py`; Python compilation passes.

#### Commit 3 applied-resolution addendum: `tensorrt_llm/_torch/attention_backend/sparse/skip_softmax/params.py` / `SkipSoftmaxScheduler.get_runtime_params`

- Resolution: add `SkipSoftmaxParams.uses_spcompress` and propagate it through main's
  `SparseRuntimeParams` replacement path for both enabled and timestep-disabled thresholds.
  Do not restore legacy `get_kernel_params`.
- Evidence: `SparseRuntimeParams` already owns `uses_spcompress`; all active main callers use
  `get_runtime_params`; Python compilation passes.

#### Commit 3 applied-resolution addendum: `tensorrt_llm/_torch/attention_backend/fmha/fallback.py` / `FallbackFmha.forward`

- Resolution: change the clean-merge Rubin reference from removed
  `forward_args.skip_softmax_kernel_params` to
  `forward_args.sparse_runtime_params.uses_spcompress`.
- Evidence: `AttentionForwardArgs` no longer defines `skip_softmax_kernel_params`; the repository
  now has zero references to that removed object and Python compilation passes.

#### Commit 3 applied-resolution addendum: `tensorrt_llm/_torch/attention_backend/trtllm.py` / `TrtllmAttentionMetadata`, `TrtllmAttention.forward`

- Resolution: retain main's current class and sparse hooks while porting Rubin's uGPU fields,
  workspace allocation, speculative-step repartitioning, per-uGPU block offsets, runtime views,
  slice builders, and MLA generation state. Route spcompress through
  `SkipSoftmaxScheduler.get_runtime_params`.
- Evidence: `accumulate` and `UgpuRuntime` imports resolve; no legacy skip-softmax object remains;
  all uGPU helper definitions and call sites remain; Python compilation passes.

#### Commit 3 applied-resolution addendum: `tensorrt_llm/_torch/modules/mla.py` / `_use_kvbproj_strided_out`, `_get_kvbproj_head_weights`, `forward_context_default`, `_bmm_bf16_out`

- Resolution: keep main's refactored MLA/DSV4 backend ownership, port Rubin's cached per-head KV-B
  views and direct strided BMM outputs, and select the Rubin BF16 BMM op on SM107. Do not add the
  source commit's explicit NVFP4 backend list because current `Linear` already has the identical
  default contract.
- Evidence: dedicated main DSA/DSV4 modules remain intact; both fallback and direct-output KV-B paths
  remain; Python compilation passes.

#### `cpp/tensorrt_llm/kernels/trtllmGenKernels/fmha/cubin/` / `kernelMetaInfo.h` (redo commit 3)

- Conflict: 3,221 generated FMHA paths were unmerged. Rubin commit 3 has a coherent 6,849-path
  bundle but omits 836 fresh-main filenames. The preserved pre-redo result incorrectly kept main's
  binary for 16 overlapping filenames that Rubin modified.
- Function/symbol scope: generated FMHA cubins, metadata declarations, SM100/SM103 preservation,
  and Rubin SM107/spcompress/fp16-softmax variants.
- Human direction: User chose option 1: preserve main-only generated coverage while respecting
  Rubin for every overlapping generated artifact and keeping Rubin metadata aligned with Rubin
  binaries.
- Resolution: use the exact Rubin commit-3 bundle for all common/Rubin paths, then add only the 836
  filenames absent from Rubin from `origin/main`. The result is Rubin-wins overlap, main-only
  additive union; no metadata splicing and no generated aggregate mixing.
- Evidence: the staged tree differs from immutable `31d398b984` only by 836 additions, with zero
  modifications or deletions. It contains 7,685 files, zero unmerged paths, zero LFS pointer stubs,
  6,832 metadata symbols, zero metadata symbols lacking a cubin, and 852 unreferenced additive
  cubins (the 836 main-only files plus 16 already-extra files in the Rubin bundle).
- Status: applied, staged, and static-reviewed in rebase commit 3/76.

#### `tensorrt_llm/_torch/custom_ops/torch_custom_ops.py` / `NVFP4GemmUnifiedRunner`, `nvfp4_gemm`, `nvfp4_gemm_inplace` (redo commit 4)

- Conflict: fresh main adds the Marlin NVFP4 backend and current unified-runner behavior, while
  Rubin adds CuTe DSL runner selection, subprocess profiling control, a deprecated compatibility
  signature, and a separate mutating inplace API for partitioned uGPU output.
- Function/symbol scope: `NVFP4GemmUnifiedRunner`, `_make_cutedsl_runner`,
  `should_profile_tactic_in_subprocess`, `nvfp4_gemm`, `nvfp4_gemm_inplace`, and
  `NVFP4GemmInplaceUnifiedRunner`.
- Human direction: User chose option 1: retain main's full backend set including Marlin and port Rubin's CuTe DSL and
  explicit inplace/uGPU APIs without allowing the non-mutating op to mutate an optional output.
- Resolution: retain the current main file and its Marlin path, preserve the three-way merges of Rubin's
  CuTe DSL runner/subprocess profiling additions, and retain Rubin's separate
  `nvfp4_gemm_inplace`/`NVFP4GemmInplaceUnifiedRunner`. The compatibility parameters on
  `nvfp4_gemm` are validation-only and direct callers to the mutating API.
- Evidence: AST comparison reports no main or Rubin symbols missing; the resolved file retains Marlin,
  `_make_cutedsl_runner`, `should_profile_tactic_in_subprocess`, and the inplace runner/API, and passes
  `py_compile` plus `git diff --check`. The prior attempt parsed, but was not reused.
- Status: applied, staged, and static-reviewed in rebase commit 4/76.

#### `tensorrt_llm/_torch/modules/fused_moe/create_moe.py` / `create_moe_backend`, `create_moe` (redo commit 4)

- Conflict: fresh main moved backend selection into `moe_resolution.py`, added the
  `MoEProblem`/deployment eligibility contract, and added MegaMoE/TRTLLM-Gen SiTu parameters.
  Rubin's older file still embeds `get_moe_cls`/legacy resolution and adds SM107 unquantized
  CuTe DSL selection plus `override_quant_config` plumbing.
- Function/symbol scope: `create_moe_backend`, `create_moe`, `resolve_moe_cls`,
  `override_quant_config`, `ActType_TrtllmGen`, `situ_beta`, and
  `trtllm_gen_activation_{type,alpha,beta}`.
- Human direction: User chose option 1: keep main's resolver ownership and every SiTu/public parameter; port only
  Rubin behavior still needed by the current contract, including SM107 CuTe DSL eligibility and
  missing quant-override propagation, without restoring duplicate legacy resolver functions.
- Resolution: retain main's `moe_resolution.py` ownership and complete SiTu/MegaMoE signature; add
  `override_quant_config` only to `create_moe_backend`, forward it to the CuTe DSL constructor family,
  and forward it from the unwrapped `create_moe` fallback. SM107 selection is supplied by the current
  structured `CuteDslFusedMoE.can_implement`, not by restoring legacy `get_moe_cls`.
- Evidence: `ActType_TrtllmGen` was introduced on fresh main by `36922c7283`; it is absent from
  the Rubin source range and final DSV4 source, so dropping main here would recreate the earlier
  unsupported-SiTu failure. AST signature comparison shows every main parameter unchanged plus the one
  Rubin override parameter; the file passes `py_compile`. The prior attempt changed this file by
  230 additions/20 deletions for a 20-addition/1-deletion source delta and resurrected obsolete resolver
  code, so that resolution was rejected.
- Status: applied, staged, and static-reviewed in rebase commit 4/76.

#### `tensorrt_llm/_torch/modules/fused_moe/fused_moe_cute_dsl.py` / `CuteDslFusedMoE.can_implement`, `run_moe` (redo commit 4)

- Conflict: fresh main uses `MoEProblem`, `MoEDeployment`, `MoEEligibility`, and
  `MoERunContext`; Rubin adds the actual SM107 BF16/NVFP4 CuTe DSL runners, uGPU fork/join,
  localized weight shards, inplace FC outputs, and post-load cleanup using the older interfaces.
- Function/symbol scope: `CuteDslFusedMoEBF16Runner`, `CuteDslFusedMoE.can_implement`,
  `run_moe`, `run_moe_bf16`, `_run_moe_{nvfp4,bf16}_ugpu`,
  `post_load_weights_impl`, and `_split_weights_for_ugpu`.
- Human direction: User chose option 1: retain main's eligibility/run-context APIs and port the complete Rubin
  SM107 BF16/NVFP4 and uGPU implementation into those current interfaces.
- Resolution: retain the exact main `MoEProblem`/`MoEDeployment` eligibility and `MoERunContext` run
  signatures; extend structured eligibility to unquantized BF16 on SM107 and NVFP4 on SM107 when the
  internal CuTe DSL dependency exists; port Rubin's BF16 runners, uGPU planner/runtime, localized
  shards, inplace outputs, memset-stream synchronization, and post-load split/release methods.
- Evidence: AST comparison reports all 11 main class methods and all 21 Rubin class methods present,
  with every shared main signature unchanged except the additive `override_quant_config` constructor
  parameter. The file passes `py_compile` and `git diff --check`. The prior attempt retained only one of
  main's five structured-contract symbol occurrences and used the obsolete tuple contract, so it was
  rejected.
- Status: applied, staged, and static-reviewed in rebase commit 4/76.

#### `tensorrt_llm/_torch/modules/fused_moe/fused_moe_trtllm_gen.py` / `TRTLLMGenFusedMoE.can_implement`, `__init__` (redo commit 4)

- Conflict: fresh main owns the new structured eligibility contract and the complete backend-local
  SiTu validation/API; Rubin's narrow change adds SM107 support and stores `model_config`.
- Function/symbol scope: `TRTLLMGenFusedMoE.can_implement`, `ActType_TrtllmGen.SiTu`,
  `_validate_backend_local_activation`, and `model_config`.
- Human direction: User chose option 1: keep main's structured eligibility and SiTu contract, extend its accepted SM
  set to include 107, and retain the Rubin `model_config` state needed by later Rubin changes.
- Evidence: main has 31 SiTu/activation-contract occurrences; both the commit's Rubin side and final
  DSV4 source have none.
- Resolution: start from fresh main, extend structured `can_implement` to SM107, retain the full SiTu
  validation and public parameters, store `model_config`, and pass Rubin's
  `model_config.use_lamport_sync` to the FP4 block-scale MoE call.
- Evidence: the resolved file preserves all six main structured-contract/25 SiTu occurrences, adds the
  Rubin SM107/model/Lamport delta, and passes `py_compile`. The prior attempt's corresponding file
  fails `py_compile` at line 709 (`invalid decimal literal`), so it was rejected rather than copied.
- Status: applied, staged, and static-reviewed in rebase commit 4/76.

#### `cpp/tensorrt_llm/kernels/trtllmGenKernels/blockScaleMoe/runner.cu` / `runner.h` / `Runner::Runner`, `getOptions` (redo commit 4 clean merge audit)

- Conflict/risk: Git merged this pair without a textual conflict. Fresh main supplies the current
  activation/SiTu path; Rubin adds `useLamport` constructor and option propagation. This is the
  exact clean merge that must be reviewed to avoid an unsupported-`ActType::SiTu` compile failure.
- Function/symbol scope: `ActType`, `mActType`, `useLamport`, all `Runner` constructors, and
  both `getOptions` overloads.
- Human direction: User chose option 1: retain the full main activation enum/SiTu behavior and Rubin's Lamport
  parameter end-to-end; validate declaration/definition/call ordering before compilation.
- Resolution: keep main's `isGatedActType` handling and explicit rejection of SiTu only for the
  standalone DeepSeek-FP8 activation path; retain Rubin's `useLamport` after `ActType` in every
  declaration/definition/call, producer flags in PermuteGemm1, and consumer flag in Gemm2.
- Evidence: declaration/definition ordering matches in `runner.h`/`runner.cu`; the MoE constructor
  forwards both `actType` and `useLamport`; the torch runner caches by `(tileN, useLamport)`, tactic
  discovery receives `use_lamport`, and Python forwards `model_config.use_lamport_sync`. This avoids
  the old failure mode of treating `ActType::SiTu` as a Rubin-added enum or replacing main's activation
  contract.
- Status: clean merge reviewed, staged, and accepted in rebase commit 4/76.

#### `cpp/tensorrt_llm/thop/fp4BlockScaleMoe.cpp` / `FP4BlockScaleMoeRunner::getRunner`, `getValidConfigs`, `run`; `tensorrt_llm/_torch/modules/fused_moe/moe_op_backend.py` / `TRTLLMOpBackend.run_fp4_block_scale_moe` (redo commit 4 clean merge audit)

- Conflict/risk: Rubin adds the Lamport selection parameter to the torch custom class and concrete
  Python backend while fresh main already has the compatible tunable-runner caller and SiTu-aware
  activation argument.
- Function/symbol scope: `getRunner`, `getValidConfigs`, `run`, `use_lamport`,
  `FP4BlockScaleMoERunner.get_valid_tactics`, and `FP4BlockScaleMoERunner.forward`.
- Human direction: not required - compatible union. The additive defaulted Lamport parameter does not
  replace or reinterpret main's activation API, and the existing main caller already supplies it.
- Resolution: retain the `(tileN, useLamport)` C++ runner cache, three-element tactic metadata, defaulted
  `use_lamport` concrete-backend parameter, and main's `gated_act_type`/SiTu validation unchanged.
- Evidence: tactic discovery passes `self.use_lamport`; execution selects Lamport only for a tactic marked
  capable and outside tuning mode; C++ constructs the runner with both stored `mActType` and the selected
  Lamport flag. The full changed-Python set passes `py_compile`.
- Status: clean merge reviewed, staged, and accepted in rebase commit 4/76.

#### `cpp/tensorrt_llm/kernels/communicationKernels/moeAlltoAllKernels.cu` / `.h` / `moeA2AGetTimeoutCycles`, `check_timeout`, CFT pointer structs (redo commit 5)

- Conflict: main owns the phase-aware two-argument timeout API and resolver; Rubin adds CFT dispatch/combine paths whose source call sites use the former single-argument timeout and adds CFT constants/types beside main's default budget.
- Function/symbol scope: `moeA2AGetTimeoutCycles`, `check_timeout`, `kDefaultTimeoutCycles`, `DispatchKernelPointers`, `CombineKernelPointers`, `CftPeerLeIds`, and CFT counter constants.
- Human direction: inherited from the previously recorded human decision for this exact API: preserve main's dynamic `(start, budget)` contract and port Rubin CFT paths to it.
- Resolution: retained main's resolver/two-argument macro, unioned `kDefaultTimeoutCycles` with Rubin's CFT constants/types, and passed `ptrs.timeout_cycles` at all three new CFT waits.
- Evidence: all five device wait sites now use the two-argument macro; all four pointer structs retain defaulted timeout fields; the resolver definition/declaration and both host assignments are present; `git diff --check` passes.
- Status: applied, staged, and static-reviewed in rebase commit 5/76.

#### `tensorrt_llm/_torch/model_config.py` / config-utils imports, `UgpuPolicy` (redo commit 5)

- Conflict: main adds Kimi Linear config helpers while Rubin adds `UgpuPolicy`; neither replaces the other.
- Function/symbol scope: `get_kimi_linear_num_attention_layers`, `is_kimi_linear`, and `UgpuPolicy`.
- Human direction: not required - compatible union.
- Resolution: retained all main Kimi/Qwen/Nemotron helpers and added Rubin's `UgpuPolicy` import.
- Evidence: both Kimi helpers have active call sites, `UgpuPolicy` initializes the model-config field, and `model_config.py` passes `py_compile`.
- Status: applied, staged, and static-reviewed in rebase commit 5/76.

#### `tensorrt_llm/_torch/pyexecutor/kv_cache_manager_v2.py` / `KVCacheManagerV2.__init__`, `_prepare_draft_resources` (redo commit 5)

- Conflict: main adds `is_estimating_kv_cache`, stable pool-role representatives, and graceful handling when draft-cache creation returns `None`; Rubin adds `enable_ugpu`, `_fork_join_attn`, and assigned-uGPU invariants. A cleanly merged dummy-cache path places the Rubin invariant before main's nullable saturation result is checked.
- Function/symbol scope: `KVCacheManagerV2.__init__`, `_fork_join_attn`, `_pool_layer_ids_by_role`, `_prepare_draft_resources`, and `_create_kv_cache`.
- Human direction: not required - compatible union.
- Resolution: retained both constructor parameters, both initialization blocks, main's stable pool-role map, and Rubin's fork/join flag. Every merged cache-creation path now checks main's nullable saturation result before asserting the Rubin uGPU assignment, including the dummy/warmup path found during whole-file review.
- Evidence: AST comparison reports no main or Rubin class/function symbols missing; constructor fields and both initialization blocks are present; the changed file passes `py_compile` and `git diff --check`.
- Status: applied, staged, and static-reviewed in rebase commit 5/76.

#### `tensorrt_llm/runtime/kv_cache_manager_v2/_core/_kv_cache_manager.py` / `KVCacheManager.create_kv_cache` (redo commit 5)

- Conflict: main adds the `text_only` per-cache override while Rubin adds optional `ugpu_id` localization; the auto-merged body already consumes both independently.
- Function/symbol scope: `create_kv_cache`, `text_only`, `ugpu_id`, and `ReuseScope`.
- Human direction: not required - compatible union.
- Resolution: retained `text_only` followed by additive `ugpu_id`; localization updates `ReuseScope`, while `text_only` continues into `_KVCache` construction unchanged.
- Evidence: AST comparison reports no main or Rubin symbols missing; both parameters and independent uses are present; the file passes `py_compile`.
- Status: applied, staged, and static-reviewed in rebase commit 5/76.

#### `tensorrt_llm/_torch/attention_backend/sparse/dsa.py` -> `sparse/dsa/indexer.py` / `Indexer._run_fp8_paged_mqa`, `sparse_attn_indexer` (redo commit 6)

- Conflict: Rubin modifies a monolithic DSA file that fresh main deleted after splitting it into the `sparse/dsa/` package. Rubin's exact delta is three integer conversions for max-sequence arguments.
- Function/symbol scope: `Indexer._run_fp8_paged_mqa`, `fp8_fp4_paged_mqa_logits`, `fp8_paged_mqa_logits`, and `cute_dsl_fp8_paged_mqa_logits`.
- Human direction: not required - compatible structural port. Retaining the deleted monolith would duplicate main's active DSA package.
- Resolution: kept main's deletion, ported both paged-MQA `int(max_seq_len)` arguments and the CuTe DSL `int(indexer_max_seq_len)` argument into `sparse/dsa/indexer.py`.
- Evidence: the source commit changes only those three call arguments; all three are present in the active package, no active import targets the deleted file module rather than the package, and `indexer.py` passes `py_compile`.
- Status: applied, staged, and static-reviewed in rebase commit 6/76.

#### `tensorrt_llm/_torch/pyexecutor/model_engine.py` / `_create_cuda_graph_warmup_request` (redo commit 6)

- Conflict: main adds mixed encoder-context CUDA-graph rows and `Sequence`/`Hashable`; Rubin adds per-uGPU longest requests, uGPU assignments, and `accumulate`. The auto-merge runs both allocation implementations, duplicates request IDs/resources, and partially mixes singular/plural longest-request variables.
- Function/symbol scope: `_create_cuda_graph_warmup_request`, `num_mixed_contexts`, `num_long_requests`, `short_request_ids`, `max_seq_len_request_ids`, and `free_warmup_requests`.
- Human direction: not required - compatible union. Mixed contexts occupy short-request slots; one longest generation request per active uGPU occupies the remaining reserved slots.
- Resolution: unioned `accumulate` with main's typing imports and replaced the duplicated allocator with one partition: `num_short_requests + num_long_requests == batch_size`. Mixed contexts occupy a prefix of short IDs; generation short IDs use the remainder; each call receives an equal-length uGPU-ID slice; longest IDs are allocated once with per-uGPU capacity and encoder-output lists sized to `num_long_requests`.
- Evidence: whole-function review shows one short allocation branch and one long allocation; no singular/plural variable splice remains; every actual commit-6 uGPU reorder/split/warmup symbol is present; the file passes `py_compile` and `git diff --check`.
- Status: applied, staged, and static-reviewed in rebase commit 6/76.

#### `tensorrt_llm/_torch/speculative/interface.py` / `SpecMetadata.draft_kv_cache_context` (redo commit 6)

- Conflict: main centralizes cache-layout, FlashMLA, and DSA state in `prepare_attn_metadata_for_draft_replay`/`restore_attn_metadata_after_draft_replay`; Rubin manually swaps older metadata fields and adds temporary uGPU fork-join control.
- Function/symbol scope: `draft_kv_cache_context`, `prepare_attn_metadata_for_draft_replay`, `restore_attn_metadata_after_draft_replay`, `ugpu_enabled`, and `ugpu_kv_cache_block_offsets`.
- Human direction: not required - compatible union. Main owns metadata save/restore; Rubin's uGPU fields are an additive temporary overlay.
- Resolution: preserved main's FlashInfer handling and complete `prepare_attn_metadata_for_draft_replay` helper, overlaid only `ugpu_enabled`/`ugpu_kv_cache_block_offsets`, restored those overlay fields in reverse order, then invoked main's full restore helper in `finally`.
- Evidence: the current method retains the additive `enable_draft_ugpu_forkjoin` argument and every Rubin uGPU branch while main remains sole owner of cache-layout, FlashMLA, and DSA state; the full file retains all non-conflicting Rubin changes and passes `py_compile`.
- Status: applied, staged, and static-reviewed in rebase commit 6/76.

#### `tests/unittest/_torch/executor/test_py_executor.py` / `test_pad_dummy_ctx_caps_to_single_sequence_limit` (redo commit 6)

- Conflict: main tests context-role padding to `max_num_tokens`; Rubin generalizes the test across model/KV-manager single-sequence caps but uses its older role selector. A second broad conflict splices Rubin's new allocation/retry tests into main's legacy-role test body.
- Function/symbol scope: `test_pad_dummy_ctx_caps_to_single_sequence_limit`, `_StubADPExecutor`, `_ADPForwardIntent.CONTEXT`, and `_adp_dummy_is_gen`.
- Human direction: not required - compatible test union.
- Resolution: retained Rubin's three parameterized cap cases with main's `_ADPForwardIntent.CONTEXT`; retained all seven independent retry/cleanup tests added by this commit; retained main's legacy-role and missing-max-token tests; removed only Rubin's duplicate old-role version of the latter.
- Evidence: every test function added by the exact commit diff is present, both main-owned forward-intent tests remain, duplicate function names are absent, and the file passes `py_compile`.
- Status: applied, staged, and static-reviewed in rebase commit 6/76.

#### `3rdparty/patches/cutlass_sm107.patch` / added blank lines (redo commit 6 staged gate)

- Conflict/risk: the Rubin patch contains two added blank lines encoded as `+ `, which fails repository whitespace validation.
- Function/symbol scope: patch hunks at lines 15 and 20; no code symbol or API.
- Human direction: not required - non-API whitespace normalization.
- Resolution: stripped trailing whitespace from the two space-only blank lines; patch code and context are otherwise unchanged.
- Evidence: `git diff --cached --check` and the complete changed-Python `py_compile` gate pass.
- Status: applied, staged, and static-reviewed in rebase commit 6/76.


#### `examples/layer_wise_benchmarks/run.py` / benchmark driver and `main` (redo commit 7)

- Conflict: fresh main and Rubin both evolved the benchmark driver across a broad structural rewrite.
- Function/symbol scope: benchmark argument plumbing, model construction, execution, and `main`.
- Human direction: not required unless inspection finds behavior that cannot coexist. Per user direction, a single implementation satisfying both branches is selected automatically.
- Resolution method: Rubin commit 7 owns the function-scoped structural refactor; port the exact fresh-main delta into that structure. Do not replace the file with the stale prior-rebase version.
- Resolution: kept Rubin commit-7 function-scoped main driver and all new Rubin options; ported fresh-main newer no-choices backend validation into that structure.
- Evidence: the only fresh-main delta since the Rubin parent is present, all main and exact commit-added AST symbols remain, and py_compile passes.
- Status: applied, staged, and static-reviewed in rebase commit 7/76.

#### `jenkins/Build.groovy`, `L0_MergeRequest.groovy`, `L0_Test.groovy`, `current_image_tags.properties` / Rubin CI stages and image keys (redo commit 7)

- Conflict: fresh main has newer CI structure and image tags; Rubin adds GR100/VR200 jobs and Rubin image keys.
- Function/symbol scope: Rubin build/test stage definitions, test-db routing, `LLM_RUBIN_DOCKER_IMAGE`, and `LLM_SBSA_RUBIN_DOCKER_IMAGE`.
- Human direction: not required - compatible union.
- Resolution method: preserve current main CI infrastructure and values, insert Rubin jobs/configuration at the current equivalent extension points, and add only the two Rubin image keys without restoring stale tags.
- Resolution: kept fresh-main versionOverride, Enroot credentials and registry conversion, retry/sidecar arguments, dynamic SBSA multi-GPU gate, and full GB300 metadata; added architecture-matched Rubin builds, image keys, stage-aware mounts, GR100 routing, and VR200 routing. Standard and Rubin build targets coexist on both CPU architectures.
- Evidence: all Rubin build/config/stage keys are present, current helper argument ordering is retained, no conflict markers remain, and git diff --cached --check passes. A Groovy executable is not installed on the host, so parser validation remains part of the later branch guard.
- Status: applied, staged, and static-reviewed in rebase commit 7/76.

#### `tests/integration/defs/accuracy/test_llm_api_pytorch.py`, `tests/integration/defs/conftest.py` / Rubin markers, fixture, and coverage (redo commit 7)

- Conflict: both sides add independent imports and test helpers around the same import/fixture regions.
- Function/symbol scope: `skip_no_rubin`, `skip_pre_rubin`, `nvfp4_gemm_backend`, and Rubin accuracy tests.
- Human direction: not required - compatible union.
- Resolution method: retain main imports/helpers and add the Rubin markers, fixture, and exact commit-added tests; reject obsolete source-base-only symbols that are not part of this commit's delta.
- Resolution: unioned current-main and Rubin imports, retained the current hang-traceback option, then added the Rubin NVFP4 option/fixture and all auto-merged accuracy coverage.
- Evidence: AST comparison reports no main or exact commit-added symbols missing, duplicate definitions are absent, and both files pass py_compile.
- Status: applied, staged, and static-reviewed in rebase commit 7/76.

#### `tests/integration/test_lists/waives.txt` / waiver set (redo commit 7)

- Conflict: main and Rubin independently changed a line-oriented waiver database; the prior rebase result was based on an older main and drops current waivers.
- Function/symbol scope: exact lines added and removed by Rubin commit `b143262054`.
- Human direction: not required - compatible set update.
- Resolution method: start from fresh main, apply only the exact commit delta, retain unrelated current-main waivers, and deduplicate the result. Do not copy the prior resolved file wholesale.
- Resolution: restored the fresh-main file and applied git diff -U0 for b143262054 directly; all unrelated main entries remain, the three old MoE IDs were replaced with Lamport-qualified IDs, and Rubin bringup/VR coverage was added.
- Evidence: the exact patch passed git apply --check before application and the final waiver set has zero duplicate lines.
- Status: applied, staged, and static-reviewed in rebase commit 7/76.

#### `tests/unittest/_torch/modules/moe/test_moe_backend.py` / MoE resolver, routing, uGPU, and Lamport tests (redo commit 7)

- Conflict: the prior rebase resolution contains Rubin coverage but drops seven fresh-main symbols, including shared-expert routing helpers/tests; source-base also has obsolete resolver tests not introduced by this commit.
- Function/symbol scope: `_AppendSharedExpertsRouting`, current MoE resolution tests, commit-added uGPU parameter coverage, and Lamport tests.
- Human direction: not required - compatible test union.
- Resolution method: retain every current-main helper/test and add every test or parameter change from the exact Rubin commit delta. Do not require obsolete source-base-only `get_moe_cls_marlin` tests and do not reuse the lossy prior file.
- Resolution: restored fresh main, applied the ten exact Rubin hunks that still matched, manually placed the four API/import hunks around the fresh-main n_shared_experts contract, and appended the exact Lamport test after the fresh-main newer shared-expert tests.
- Evidence: no fresh-main or exact commit-added AST symbols are missing, including _AppendSharedExpertsRouting and all three main tests previously dropped; duplicate definitions are absent and py_compile passes.
- Status: applied, staged, and static-reviewed in rebase commit 7/76.

#### `tests/unittest/_torch/thop/parallel/test_fp4_linear.py` / uGPU FP4 coverage (redo commit 7)

- Conflict: both branches add imports near the same region while Rubin adds independent uGPU FP4 tests.
- Function/symbol scope: current main imports/tests and commit-added Rubin uGPU test functions.
- Human direction: not required - compatible union.
- Resolution method: union the imports and retain both test sets; verify all exact commit-added functions and no duplicate definitions.
- Resolution: unioned the utility and uGPU imports and retained all auto-merged Rubin FP4 tests beside current-main coverage.
- Evidence: no fresh-main or exact commit-added AST symbols are missing, duplicate definitions are absent, and py_compile passes.
- Status: applied, staged, and static-reviewed in rebase commit 7/76.


#### `tensorrt_llm/_torch/modules/mla.py` -> `attention_backend/sparse/deepseek_v4/module.py` / `create_sparse_attn_weights`, `project_sparse_attn_output` (redo commit 8)

- Conflict: Rubin commit `772dd29d0d` gates DSv4 FP8 o_a_proj allocation and fused inverse-RoPE on `use_cute_dsl_blockscaling_bmm`, but fresh main deleted the monolithic MLA-owned DSv4 implementation and relocated it into the DeepSeek-v4 sparse-attention module.
- Function/symbol scope: `create_sparse_attn_weights`, `project_sparse_attn_output`, `o_a_proj`, and `fused_inv_rope_fp8`.
- Human direction: not required - compatible structural port.
- Resolution: keep both deleted monolithic blocks deleted; apply the exact gate in the relocated implementation. The default BF16 `o_a_proj` remains when the flag is false, and SM100 overrides it with FP8 only when the CuTe DSL block-scaling BMM flag is true; the fused inverse-RoPE path uses the same gate.
- Evidence: the relocated loader dequantizes into the BF16 destination, both gate sites match, the old monolithic symbols remain absent, and the changed Python files pass `py_compile`.
- Status: applied, staged, and static-reviewed in rebase commit 8/76.

#### `tensorrt_llm/_torch/pyexecutor/kv_cache_manager_v2.py` / generation dummy capacity (redo commit 8)

- Conflict/risk: clean merge. Rubin adds one generation token to dummy KV capacity in addition to draft capacity.
- Function/symbol scope: generation branch of dummy request preparation and `new_capacity`.
- Human direction: not required - additive correctness fix.
- Resolution: retain `kv_cache.capacity + _kv_draft + 1` with the already merged saturation handling from earlier commits.
- Evidence: the final staged hunk is the exact `+ 1` delta atop the current saturation-aware resize and warning path; `py_compile` passes.
- Status: applied, staged, and static-reviewed in rebase commit 8/76.

#### `moe_alltoall.py`, `nvlink_one_sided.py` / CFT combine selection (redo commit 8)

- Conflict/risk: clean merge. Rubin disables the CFT combine path when low-precision combine is selected.
- Function/symbol scope: both `moe_a2a_combine` call sites and `use_cft_for_combine`.
- Human direction: not required - compatible guard.
- Resolution: retain the identical `and not use_low_precision_combine` guard at both public and backend call sites.
- Evidence: both staged call sites have the same low-precision exclusion and preserve current argument order; `py_compile` passes.
- Status: applied, staged, and static-reviewed in rebase commit 8/76.


#### Clean commits 9-12 / formatting, speculative event flush, C++ uGPU disable, and CFT dtype (redo commits 9-12)

- Conflict/risk: no textual conflicts, but each patch was reviewed before resolving commit 13.
- Function/symbol scope: `MoeAlltoAll.combine`, `PyExecutor._flush_pending_transfer_responses` and KV-event flush sites, `_create_gpu_cache_tier_config`, and both CFT combine dtype switches.
- Human direction: not required - compatible clean merges.
- Resolution: retain the formatter/license-only MoE change; retain rank-symmetric KV-event flushing and the transceiver guard; retain the backend-specific C++ uGPU disable wrapper and tests; retain `tensorrt_llm::DataType` in both CFT switches.
- Evidence: each rebased patch was inspected in full and `git show --check` passed for commits `143202372a`, `927945c0ab`, `409ead76a8`, and `1270cbbaa5`.
- Status: clean merges reviewed and accepted as rebase commits 9-12/76.

#### `tensorrt_llm/_torch/pyexecutor/kv_cache_manager_v2.py` / backend uGPU adapters, `num_ugpus`, and mirrored draft cache (redo commit 13)

- Conflict: Rubin commit `3bc7071988` adds Python/C++ backend adapters and routes fork-join through the effective `num_ugpus` property; fresh main still references the Python implementation directly and adds a nullable cache-creation retry before the uGPU invariant.
- Function/symbol scope: `_get_num_ugpus`, `_get_kv_cache_ugpu_id`, `_create_backend_kv_cache`, `KVCacheManagerV2.num_ugpus`, constructor `_fork_join_attn`, `_create_kv_cache`, and mirrored draft-cache creation.
- Human direction: not required - compatible union.
- Resolution: use `self.num_ugpus` so the C++ backend reports one effective uGPU; preserve fresh mains `kv_cache is None` retry path and apply the backend-neutral uGPU-id helper only after that check. Retain the exact backend cache-construction adapters and their tests.
- Evidence: only the two backend-private adapter bodies directly access `impl.num_ugpus` or `kv_cache.ugpu_id`; no current-main or exact commit-added AST symbols are missing, the staged stats match the source commit exactly, and changed Python passes `py_compile`.
- Status: applied, staged, and static-reviewed in rebase commit 13/76.


#### `cpp/tensorrt_llm/common/attentionOp.cpp`, `thop/attentionOp.cpp` / `getWorkspaceSizeForContext`, `RunnerBase::getWorkspaceSize`, runtime attention call (redo commit 14)

- Conflict: fresh main fixes cross-attention workspace under-allocation while Rubin commit `9bfface7a4` expands the workspace API to size context and generation from the actual runtime batch and padded lengths. Both address the same allocation boundary through different API shapes.
- Function/symbol scope: `AttentionOp::getWorkspaceSizeForContext`, `RunnerBase::getWorkspaceSize`, its override, `attention`, `padded_num_tokens`, `padded_kv_tokens`, and `cross_kv_length`.
- Human direction: not required - one expanded runtime API satisfies both branches.
- Resolution: take the runtime context/generation signature and padded-buffer formulas from Rubin; compute `cross_kv_length` with fresh mains authoritative `op->isCrossAttention()` check, then pass it through the expanded signature. This preserves mains cross-attention fix without retaining the obsolete `maxCrossKvLength` compatibility parameter.
- Evidence: declaration and override use the same ten-argument order and there is one call; the obsolete maxCrossKvLength parameter is absent, the call uses `op->isCrossAttention()`, padded token bounds cover all changed unfused buffers, both test sets remain, and `git diff --cached --check` passes.
- Status: applied, staged, and static-reviewed in rebase commit 14/76.


#### `fused_tf32_pmap_gemm.cuh` / Rubin fused-MHC optimization (redo commit 15 clean merge)

- Conflict/risk: no textual conflict; the large kernel-only performance patch changes TMA descriptors/barriers, uneven split ownership, and packed FP32x2 math.
- Function/symbol scope: `fma_f32x2`, `mul_f32x2`, fused kernel descriptors, `full_mix`, and `H_TILES_BASE/H_TILES_EXTRA`.
- Human direction: not required - clean Rubin-only kernel optimization.
- Resolution: retain the exact clean merge; no current-main API or caller was displaced.
- Evidence: the complete rebased patch was reviewed and `git show --check dfb832221d` passes. Compile validation is deferred to the batch-xdr build.
- Status: clean merge reviewed and accepted as rebase commit 15/76.

#### `mega_moe/mega_moe_cute_dsl.py` / `MegaMoECuteDsl.can_implement`; custom op SM gate; developer guide (redo commit 16)

- Conflict: fresh main replaced the legacy tuple-returning `can_implement` API with structured `MoEProblem`/`MoEDeployment` eligibility and newer resolution docs; Rubin updates the legacy error text and custom-op gates to recognize SM100-SM109, including SM107.
- Function/symbol scope: `MegaMoECuteDsl.can_implement`, `_is_megamoe_supported_sm`, runner/op construction gates, SM-family tests, and both documentation tables.
- Human direction: not required - the current structured API already uses `is_sm_100f`, whose active definition is SM100-SM109, so it supports Rubin without reverting the API.
- Resolution: keep the structured eligibility method and change only its stale SM error text to SM100-SM109; retain the clean custom-op helper and exact parameterized test. In docs, update only MegaMoE-CuteDSL hardware/NVFP4 cells while preserving current main resolver, FP8, W4A16, and Marlin descriptions.
- Evidence: `is_sm_100f` is defined as 100 <= SM < 110; the exact helper test covers 90, 100, 103, 107, 109, 110, and 120; AST comparison reports no current-main or exact commit-added symbols missing; current resolver, FP8, W4A16, and Marlin docs remain; changed Python passes `py_compile`.
- Status: applied, staged, and static-reviewed in rebase commit 16/76.


#### `requirements.txt`, `tensorrt_llm/__init__.py` / CUTLASS DSL internal wheel and compatibility shim; `MegaMoECuteDsl._maybe_init_torch_dist_under_mpi` (redo commit 17)

- Conflict: Rubin updates the internal CUTLASS DSL pin and adds a narrow `cute.core` compatibility shim inside the old eager package initializer; fresh main intentionally removed the internal pin and replaced eager package imports with a PEP 562 lazy public surface. The rank-to-local-device implementation and test merge cleanly.
- Function/symbol scope: `_setup_cutlass_dsl_compatibility`, `_LAZY_ATTRS`/`__getattr__`, internal DSL requirement, `_maybe_init_torch_dist_under_mpi`, `local_mpi_comm`, and `device_id`.
- Human direction: not required - compatible package union for the Rubin branch.
- Resolution: retain main's lazy imports and export table verbatim; add only the explicit ThrCopy/ThrMma compatibility shim after torch import, with no eager product imports or duplicate `_init`. Retain Rubin's updated internal wheel pin because active SM107 kernels explicitly require that package. Retain the clean local-rank modulo device binding and `device_id` process-group argument.
- Evidence: AST comparison reports no current-main or exact commit-added symbols missing; `_LAZY_ATTRS`, the shim definition/call, and the internal pin each occur once; no eager product import was restored; all changed Python passes `py_compile` and `git diff --cached --check` passes.
- Status: applied, staged, and static-reviewed in rebase commit 17/76.


#### Commit 18 CuTe diagnostics / `modeling_speculative.py` license, `TestDeepSeekV4Pro` resource markers, clean compatibility changes (redo commit 18)

- Conflict: source adds a full legacy license header where fresh main already has the repository SPDX header; DeepSeek-v4 source reduces a former 8-GPU topology to four devices/ranks while main independently changed the class marker to an eight-rank requirement. The method already carries the exact four-rank marker after auto-merge.
- Function/symbol scope: module header, `_deepseekv4_pro_agg_llm_kwargs`, `TestDeepSeekV4Pro.test_gsm8k_full_accuracy`, CuTe helper initialization, DSpark draft EPLB defaults, and four MoE diagnostic log sites.
- Human direction: not required - compatible union.
- Resolution: keep the fresh-main SPDX header; use the source class-level four-device marker together with the already merged method-level four-MPI-rank marker, matching TP4/EP4. Retain clean diagnostic WARNING-to-DEBUG changes, CuTe legacy-helper guard, DSpark draft EPLB defaults, dependency metadata, and exact tests/script.
- Evidence: the class requires four devices and its sole method requires four MPI ranks, matching TP4/EP4; AST comparison reports no current-main or exact commit-added symbols missing; all clean hunks were reviewed; changed Python passes `py_compile`, the new script passes `bash -n`, dependency metadata parses as YAML, all four MoE sites use DEBUG, and `git diff --cached --check` passes.
- Status: applied, staged, and static-reviewed in rebase commit 18/76.


#### Clean commits 19-20 / disagg draft-capacity expectation and DeepGEMM TMA namespace (redo commits 19-20)

- Conflict/risk: no textual conflicts.
- Function/symbol scope: `test_disagg_gen_transition_reserves_target_drafts_without_context_drafts` and four fused-MHC coefficient TMA calls.
- Human direction: not required - compatible clean changes.
- Resolution: retain the corrected capacity expectation of 137 and the four `deep_gemm::tma::copy` calls.
- Evidence: both complete rebased patches were reviewed and `git show --check` passes for `7900737dda` and `7ea5fd20b3`.
- Status: clean merges reviewed and accepted as rebase commits 19-20/76.

#### `transfer.py`, `py_executor.py`, `test_benchmark_disagg.py` / rank-safe benchmark fill collective (redo commit 21)

- Conflict: Rubin commit `3801def825` folds terminal-KV state into the existing four-field fill-gate collective; fresh main adds encoder polling, earlier block/window variables and receiver-window trimming, and richer test annotations. Independent side selection would either drop encoder progress or reintroduce a second collective.
- Function/symbol scope: `Sender._get_requests` block mismatch diagnostics, `_allgather_model_parallel_status`, `_gen_only_benchmark_has_terminal_kv_exhaustion`, `_prepare_and_schedule_batch`, `_is_benchmark_disagg_fill_complete`, `_check_benchmark_disagg_gate`, and rank-safety tests.
- Human direction: not required - compatible protocol union.
- Resolution: keep Rubin's single four-field fill-gate collective; in prepare, reset local terminal state and then preserve fresh main's `_poll_encoder_steps()` before fetching. Preserve fresh main's receiver-window trim and run it before Rubin's block-count mismatch invariant so a legitimate sliding-window head trim cannot cause a false rejection. Retain the typed/documented current test method. Do not retain main's `_sync_gen_only_benchmark_has_insufficient_kv`: its original contract performs the second collective that this fix removes, while a wrapper would silently change that contract.
- Evidence: `_allgather_model_parallel_status` is called once per fill-gate iteration, every test status is the four-field tuple, local terminal state resets before encoder polling, and the block mismatch check follows `_trim_receiver_window_head`. AST comparison reports no exact commit-added symbols missing; the removed main helper is the intentionally replaced collective API, and renamed main fail-fast tests are covered by the exact commit-added shared-gate variants. All three files parse and pass `py_compile`; no conflict markers remain and `git diff --check` passes.
- Status: applied, staged, and static-reviewed in rebase commit 21/76.


#### Clean commits 22-26 / fused-MHC coefficient loads and CuTe dependency compatibility (redo commits 22-26)

- Conflict/risk: no textual conflicts; the commits form a dependent sequence across the fused-MHC kernel, CuTe pointer wrappers, dependency pins, and package compatibility shim.
- Function/symbol scope: fused `post_mix`/`comb_mix` loads and launch signature, both `_Pointer.__add__` implementations, `apache-tvm-ffi`, and `_setup_cutlass_dsl_compatibility`.
- Human direction: not required - compatible clean changes.
- Resolution: retain the complete sequence: the final fused-MHC kernel loads coefficient rows directly through passed global pointers; both Blackwell and Rubin wrappers preserve pointer alignment using the GCD of byte offset and prior alignment; TVM FFI is pinned to 0.1.11; and `make_fragment` aliases the renamed `make_rmem_tensor` only when needed.
- Evidence: complete patches for `91fd6f1995`, `a4308231af`, `93581ee7b1`, `69345afc4b`, and `780474b06b` were reviewed in order and each passes `git show --check`.
- Status: clean merges reviewed and accepted as rebase commits 22-26/76.

#### `transfer.py` / `_align_kv_blocks`, `_trim_receiver_window_head`, `_build_kv_write_meta`; asymmetric SWA test (redo commit 27)

- Conflict: fresh main trims a wider receiver window with `_trim_receiver_window_head`; Rubin commit `d1b377acc3` instead adds `minimum_token_start` to explicit token-position alignment. The diagnostic inherited from commit 21 rejects `dst > src`, which is valid when speculative generation exposes one extra leading SWA block.
- Function/symbol scope: `Sender._align_kv_blocks`, `Sender._trim_receiver_window_head`, `Sender._build_kv_write_meta`, and `TestAlignKvBlocks.test_dst_has_wider_speculative_swa_window`.
- Human direction: not required - the generalized Rubin alignment satisfies both branches.
- Resolution: use `minimum_token_start` as a shared lower bound and let `_align_kv_blocks` trim both arrays from their explicit starts. Remove the older helper call and the invalid block-count rejection from the active path, while retaining the private main helper symbol for compatibility. Preserve the exact Rubin regression test.
- Evidence: no current-main or exact commit-added AST symbols are missing; the active path has one `minimum_token_start` call and no pre-alignment `dst > src` rejection; both changed files pass `py_compile` and `git diff --check`. The host Python lacks `pytest`, so the focused test is deferred to batch-xdr branch validation.
- Status: applied, staged, and static-reviewed in rebase commit 27/76.


#### `fp4BlockScaleMoe.cpp`, `TRTLLMGenFusedMoE._requires_separated_routing`, `test_moe_backend.py` / DSV4 routing and EP workspace (redo commit 28)

- Conflict: current main has structured MoE contract imports plus SiTu API and deterministic fallback tests in the same regions where Rubin adds the TRTLLM-Gen backend import and separated-routing regression test. Production code auto-merges.
- Function/symbol scope: `run_fp4_block_scale_moe_runner`, `TRTLLMGenFusedMoE._requires_separated_routing`, and `test_trtllm_gen_honors_separated_routing_requirement`.
- Human direction: not required - compatible union.
- Resolution: retain every current-main contract import and test, including the pre-SiTu rejection check; add the TRTLLM-Gen import and exact separated-routing test. Keep Rubin production behavior: routing methods that require Python routing short-circuit before backend-specific conditions, and EP workspace/CTA bounds use `local_num_experts` while the routing domain remains global.
- Evidence: no current-main or exact commit-added AST symbols are missing; `requires_separated_routing` is an existing routing-method property; both EP allocation formulas use `local_num_experts`; changed Python passes `py_compile`, no markers remain, and `git diff --check` passes.
- Status: applied, staged, and static-reviewed in rebase commit 28/76.


#### `_util.py`, `model_engine.py`, `test_seq_slot_sizing.py` / non-MTP overlap headroom (redo commit 29)

- Conflict: Rubin commit `1289eecf30` removes the speculative-MTP condition from a DSv4-specific headroom helper. Fresh main subsequently replaced that helper with topology-based `should_enable_disagg_adp_overlap_headroom`, moved slot sizing before model loading, and split scheduler-aware/non-overlap dummy policy into separate APIs.
- Function/symbol scope: `should_enable_dsv4_overlap_headroom` (superseded), `should_enable_disagg_adp_overlap_headroom`, `PyTorchModelEngine.__init__`, `max_num_seq_slots`, and sequence-slot sizing tests.
- Human direction: not required - current main already implements the Rubin behavior through a more general API.
- Resolution: retain current main in all three files. Its gate has no speculative-config dependency and grants 2x slots for non-PP disaggregated attention-DP overlap, including DSv4 without MTP. Do not restore the obsolete model-specific helper or the later duplicate slot assignment.
- Evidence: there is one `max_num_seq_slots` computation, it uses the topology gate before model loading, the gate takes no speculative config, and tests explicitly construct sampler args with `speculative_config=None`. No current-main AST symbols are missing; the source commit adds no new AST symbols after accounting for its renames; all three files pass `py_compile` and `git diff --check`.
- Status: resolved as semantically absorbed by fresh main; index is clean and the redundant source commit is expected to drop at rebase commit 29/76.


#### `deepseek_v4/indexer.py`, `metadata.py`, `test_deepseek_v4_fp4_indexer.py` / fused Indexer-Q serialization (redo commit 30)

- Conflict: Rubin commit `65b8ab1b92` edits `DeepseekV4Indexer` in the old monolithic `deepseek_v4.py`; fresh main moved the class into `deepseek_v4/indexer.py`, leaving Git to propose the entire old class inside `metadata.py`. Test imports also target the old module.
- Function/symbol scope: `_is_fused_project_mxfp4_enabled`, `_project_and_quantize_q`, `precompute_aux`, `_run_overlapped_indexer_prepare`, `forward`, and three Indexer-Q stream tests.
- Human direction: not required - structural relocation with a compatible function-level port.
- Resolution: keep `metadata.py` exactly as fresh main and port only the Rubin deltas into the relocated `indexer.py`. Fused MXFP4 returns no auxiliary precompute and selects serial prepare; the legacy projection retains overlap. Record auxiliary outputs on the consumer stream for both inline and precomputed overlap. Adapt tests to import the relocated indexer module/class while preserving main cache-manager and params imports.
- Evidence: exactly one `DeepseekV4Indexer` class exists; no current-main symbols, the exact added helper, or any of the three exact added tests are missing. All three files pass `py_compile`, no markers remain, and `git diff --check` passes.
- Status: applied, staged, and static-reviewed in rebase commit 30/76.


#### THOP private ATen includes, Rubin Jenkins images / updated Rubin dev container (redo commit 31)

- Conflict: fresh main migrated standard images and `IMAGE_NAME` to a newer Artifactory registry/tag set, while Rubin commit `423e6a9ba0` updates only the two Rubin images from 20260627 to 20260724. The same commit removes an unavailable private ATen resize header from eight THOP sources.
- Function/symbol scope: eight `ATen/native/cuda/Resize.h` includes, `LLM_DOCKER_IMAGE_RUBIN_X86_64`, `LLM_DOCKER_IMAGE_RUBIN_AARCH64`, `LLM_RUBIN_DOCKER_IMAGE`, and `LLM_SBSA_RUBIN_DOCKER_IMAGE`.
- Human direction: not required - compatible key-specific update.
- Resolution: retain every fresh-main registry and non-Rubin image value; update only both Rubin keys and both Build.groovy architecture constants to the 20260724 images. Retain all eight private-header removals.
- Evidence: standard image keys remain on fresh main Artifactory values; all four Rubin references use 20260724; none of the eight THOP files references the removed header or its resize symbols; `git diff --check` passes. Compile proof is deferred to batch-xdr.
- Status: applied, staged, and static-reviewed in rebase commit 31/76.


#### Rubin blockwise GEMM, MLA BMM dispatch, relocated DSv4 output projection and tests (redo commit 32)

- Conflict: Rubin commit `1c33e9e8d7` adds the SM107 blockwise kernel/custom op and architecture helpers in `modules/mla.py`, but also edits DSv4 weight allocation/projection methods that fresh main relocated to `attention_backend/sparse/deepseek_v4/module.py`. The output-projection test combines new Rubin availability checks with a fresh-main relocated function import.
- Function/symbol scope: `CuteDSLFp8RubinBmmRunner`, `cute_dsl_fp8_bmm_rubin`, `_is_cute_dsl_fp8_bmm_available`, `_cute_dsl_fp8_bmm_out`, `fp8_block_scaling_bmm_out`, `create_sparse_attn_weights`, `_run_dsv4_o_lora_bmms`, `project_sparse_attn_output`, and `test_deepseek_v4_o_proj`.
- Human direction: not required - compatible architecture dispatch port; no old MLA API is restored.
- Resolution: retain the new Rubin kernel, custom op, tuning/in-place metadata, loader comment, and tests. Keep only the new generic availability/dispatch helpers in `modules/mla.py`; port DSv4 allocation and projection changes to the relocated module. Native FP8 `o_a_proj` is selected when the shared helper reports SM100/103 public CuTe DSL or SM107 internal CuTe DSL; otherwise SM100f allocates the BF16 fallback. Both epilogue and inverse-RoPE paths use the shared Blackwell/Rubin dispatcher. The test uses that same helper rather than the source patch inconsistent standalone SM120 predicate.
- Evidence: all current-main and exact commit-added nested Python symbols remain; the relocated module loses no main symbols; hard-coded Blackwell BMM calls remain only inside the central dispatcher. All nine changed Python files pass `py_compile`, no markers remain, and `git diff --check` passes. Kernel compilation is deferred to batch-xdr.
- Status: applied, staged, and static-reviewed in rebase commit 32/76.


#### Clean commits 33-35 / Rubin FP8 GEMM, Relu2 MoE activation, and vendored minimal kernels (redo commits 33-35)

- Conflict/risk: no textual conflicts; these are dependent Rubin kernel/custom-op changes, including a large repair commit that removes runtime dependence on `nvidia_cutlass_dsl.examples`.
- Function/symbol scope: `CuteDSLFp8RubinGemmRunner`, Rubin FP8 GEMM/BMM ops and tests, generalized Rubin gather-grouped activation fusion with SwiGLU/Relu2, and local Blackwell base helpers used by Rubin dense kernels.
- Human direction: not required - compatible clean sequence.
- Resolution: retain the shared Rubin FP8 tactic tables and GEMM op; retain the activation-generalized MoE runner/kernel/API and explicit uGPU SwiGLU limitation; retain the vendored Blackwell scheduler, dtype, alignment, and compilation helpers so Rubin kernels inherit only repository-owned code.
- Evidence: complete patches for `bf7e42495e`, `88a054b619`, and `e0624dae5a` were reviewed in order and each passes `git show --check`. The later commit-36 audit compiles every touched Python file and checks all nested symbols.
- Status: clean merges reviewed and accepted as rebase commits 33-35/76.

#### Rubin dense block-scaled GEMM and quant-layout API / custom ops, `Linear`, `GatedMLP`, model defaults and tests (redo commit 36)

- Conflict: fresh main adds `Type`, stronger activation-quantization/plain-SwiGLU checks, and KV-manager preference coverage where Rubin commit `cef81854cf` adds `ClassVar`, a projection-owned fused-layout predicate, SM107 MXFP8 dispatch, and architecture-dependent DSv4 defaults. The commit also expands `fp8_quantize_1x128_packed_ue8m0` with `use_r128c4_layout`.
- Function/symbol scope: `CuteDSLBlockScaledRubinLinear` and datatype subclasses, `Linear.can_use_cute_dsl_nvfp4_swiglu_blackwell`, `GatedMLP._can_fuse_gate_up_swiglu`, `DeepseekV4ForCausalLM.get_model_defaults`, packed-UE8M0 C++/fake/wrapper APIs, scale transforms, grid scheduler helper, and associated tests.
- Human direction: not required - compatible API and safety union.
- Resolution: import both `ClassVar` and `Type`. Use the projection-owned predicate for weight-layout/dispatch consistency, strengthen it with the main-side `has_nvfp4_activation_quantization`, and retain the main-side activation plus plain-SwiGLU checks in GatedMLP. Union the architecture-default test with the V2 preference assertion. Preserve the new optional quant-layout parameter end to end: Rubin uses default R128c4, legacy DeepGEMM passes `False`. Retain the main-side legacy value and padded-row tests under explicit `False`, alongside new R128c4 tests. Do not restore private `_compute_grid`; `_compute_mixed_cluster_grid` intentionally replaces its incompatible three-value return contract and owns the sole mixed-cluster call.
- Evidence: every current-main and exact commit-added nested Python symbol remains except the intentionally superseded private grid helper; all changed Python files pass `py_compile`; all API callers select the appropriate layout; C++ declaration/registration and fake registration agree; no markers remain and `git diff --check` passes.
- Status: applied, staged, and static-reviewed in rebase commit 36/76.


#### `moe_scheduler.py`, MoE all-to-all APIs, and `test_moe_a2a.py` / scale-free FP8 CFT counted-writes (redo commit 37)

- Conflict: fresh main replaced backend-specific scheduler keyword plumbing with `MoERunContext` and `MoECommPlan`, including centralized one-sided workspace planning. Rubin commit `0c1f922b44` removes direct per-run mutation of `payload_in_workspace` while adding low-precision combine through the all-to-all stack. Its multi-GPU test also generalizes the BF16 payload helper to arbitrary floating-point payload dtypes.
- Function/symbol scope: `MoEScheduler._plan_onesided_workspace`, `_build_run_context`, `_build_comm_plan`, `NVLinkOneSided.use_low_precision_combine`, MoE all-to-all combine declarations/wrappers, and `make_floating_point_payloads`.
- Human direction: not required - fresh main structured scheduler and Rubin low-precision combine are compatible.
- Resolution: keep the structured `MoERunContext` and `MoECommPlan` path and its single centralized assignment of `moe.comm.payload_in_workspace`; do not restore the obsolete backend-kwargs mutation. Preserve Rubin low-precision combine flag end to end through the one-sided backend and C++/Python all-to-all APIs. Keep `make_floating_point_payloads(..., payload_dtype)` as the generalized replacement for `make_bfloat16_payloads`; its only active caller passes the requested dtype, so a compatibility wrapper would be redundant.
- Evidence: all source-changed Python files parse and pass `py_compile`; AST comparison loses no current-main or exact commit-added symbol except the intentionally generalized private test helper; the generalized helper has one active call; low-precision combine is stored by `NVLinkOneSided` and forwarded at every combine boundary; no conflict markers remain and `git diff --check` passes.
- Status: applied and static-reviewed in rebase commit 37/76; staging and rebase continuation pending.


#### uGPU GEMM and MoE autotuning refactor (redo commit 38)

- `fused_moe_cute_dsl.py` / `CuteDslFusedMoE.can_implement`, `_get_reserved_moe_output_memset_stream`, `_moe_output_memset_run_stream`, `_ugpu_workload_identity`: retain fresh main `MoEStaticCapability` and structured `MoEProblem`/`MoEDeployment` eligibility; add Rubin reserved-remainder-stream and workload-identity methods. Do not restore the obsolete tuple-return eligibility API.
- `gated_mlp.py` / `_can_fuse_gate_up_swiglu_fp4out`: retain fresh main `is_static_nvfp4_input_eligible`, which is the shared calibrated-input predicate and satisfies the Rubin checks without duplicating them.
- `linear.py` / imports, `quant_config_has_nvfp4_activation_quantization`, `_uses_marlin_nvfp4_backend`, `_copy_to_new_cuda_allocation`, `NVFP4LinearMethod.apply`, `Linear.__init__`, `can_use_cute_dsl_nvfp4_swiglu_blackwell`: union LoRA result support with BF16/NVFP4 uGPU layouts and planner types; retain both main backend helpers and the Rubin fresh-allocation helper; fuse bias only when no caller output buffer is supplied and Marlin is not selected; use Rubin deferred GEMM-allreduce configuration after final quantization; require both activation-quantization safety and a disabled uGPU partition plan for the Blackwell fused layout.
- `mla.py` and `attention_backend/sparse/deepseek_v4/module.py` / `MLA.__init__`, `_bind_v_b_proj_weight`, `_bmm_bf16_out`, `_rebuild_ugpu_bf16_bmm_weight_shards`, `cache_derived_state`, DSv4 `_use_q_b_cute`: preserve fresh main relocated sparse-attention structure. Port only Rubin uGPU policy, BF16 Linear options, absorption shard planning/calls, strict KV weight-shape validation, and cache resets into the current MLA. Propagate the option to the relocated DSv4 `o_b_proj`, and disable the standalone CuTe Q projection when localized q_b shards exist. Do not restore the old monolithic DSA/DSv4 methods.
- `model_loader.py` / `_qualify_post_transform_profile`: add `UGPU_LOCALIZED_WEIGHTS` to enabled features while retaining the lazy `_post_transform_profile_registry()` accessor.
- `test_model_loader_mx.py` / unsupported-feature and fallback tests: preserve CPU markers and generalize the former separate-draft test to cover both separate draft and localized uGPU; generalize fallback coverage to unregistered-model and uGPU-feature cases.
- `test_remove_copy_pass.py` / optional-none and inplace-output tests: retain the Rubin additions alongside all fresh-main tests.
- `test_extractor.py` / localized KV page-table rejection and layer-group serialization: retain both tests; the Rubin rejection test precedes the fresh-main CPU-only serialization test.
- `_torch/utils.py` / module header: retain the repository SPDX-only header and discard the duplicate legacy license block.
- C++ uGPU runtime and remaining clean files: retain the source refactor from vendored private `ugpu_internal` headers to the public `ugpuResourceConfig` and runtime implementation; no stale `ugpu_internal` references remain in active C++ or Python sources.
- Human direction: not required - each resolution has one implementation that preserves the fresh-main contract and Rubin capability.
- Evidence: every changed Python file passes `py_compile`; AST comparison reports no missing fresh-main or exact source-added symbols except the intentionally generalized separate-draft test name, whose case remains in the parameter table. The relocated DSv4 port also preserves the source option and safety behavior. No conflict markers or stale private-uGPU references remain, and `git diff --cached --check` passes.
- Status: applied, staged, and static-reviewed in rebase commit 38/76.


#### Clean commits 39-49 / Rubin kernel architecture, one-sided A2A, compiler mutations, MXFP8, and FMHA inventory

- Conflict/risk: no textual conflicts. These commits update interacting Rubin kernel dispatch and metadata paths after the uGPU refactor.
- Function/symbol scope: Rubin BF16 TMEM architecture selection; one-sided MoE A2A fast path and Python CFT alignment fallback; `mla_custom_op_inplace` mutation declarations; residual-add-norm input indexing; SM107 MLA chunked-prefill coverage; MXFP8 GEMM tactics and CUTLASS dispatch; DSv4 FMHA epilogue dispatch; threshold-based correction skipping and the Rubin FMHA cubin inventory.
- Human direction: not required - clean source changes remain compatible with the resolved main APIs.
- Resolution: retain all eleven clean commits in source order. In particular, keep Python ownership of CFT alignment fallback after the counted-writes changes, preserve every declared MLA mutation, and retain the SM107 dispatch extensions. Keep the complete FMHA inventory as Git-LFS-managed content.
- Evidence: every commit passes `git show --check`; all Python changed since commit 38 passes `py_compile`; the representative FMHA cubin reports `filter=lfs`, `diff=lfs`, and `merge=lfs`; the 7,278-file inventory commit leaves the worktree clean.
- Status: clean merges reviewed and accepted as rebase commits 39-49/76.


#### Clean commits 50-52 / inverse-RoPE plumbing and Git-LFS FMHA inventory

- Conflict/risk: no textual conflicts.
- Function/symbol scope: DSv4 inverse-RoPE FP8 quant fusion parameters, `.gitattributes` FMHA metadata rule, and the union of Rubin and main cubin inventories.
- Human direction: not required - compatible clean sequence.
- Resolution: retain the inverse-RoPE parameter restoration, track Rubin FMHA metadata with Git LFS, and retain the merged inventory rather than choosing either branch inventory.
- Evidence: all three commits pass `git show --check`; the inventory remains Git-LFS-managed and the worktree was clean before commit 53.
- Status: clean merges reviewed and accepted as rebase commits 50-52/76.

#### MLA CUDA, THOP, `trtllm.py`, and relocated DSv4 module / fused q_b GEMM, RMSNorm, and RoPE (redo commit 53)

- Conflict: Rubin commit `3c98c95c59` adds `q_rope_applied` to the older MLA preprocessing API and implements fused SM107 q_b GEMM/norm/RoPE in the old monolithic `MLA`. Fresh main has since added fused KV normalization, precomputed scheduler buffers, split KV-only and hoisted launches, and moved DSv4 execution into `attention_backend/sparse/deepseek_v4/module.py`.
- Function/symbol scope: `MlaParams.q_rope_applied`, `applyMLARopeAndAssignQKVKernelOptContext`, `applyMLARopeAndAssignQKVKernelGeneration`, `invokeMLARopeGeneration`, `MLARopeGeneration`, `TrtllmAttention.mla_rope_generation`, `_is_dsv4_q_b_gemm_norm_rope_fusion_enabled`, `_deepseek_v4_q_b_gemm_norm_rope_fused_fp8`, DSv4 generation/context consumers, and fused-state cleanup.
- Human direction: not required - one expanded API preserves both branches.
- Resolution: retain every fresh-main KV-norm, precomputed-seqlen, precomputed-FMHA-scheduler, `kv_only`, `kv_done_elsewhere`, and `quant_scale_qkv` parameter, then append `q_rope_applied` consistently through Python, THOP, `MlaRopeGenArgs`, and CUDA. A complete pre-rotated Q uses the compact generation grid and skips Q work, while existing KV-only/hoisted logic still owns KV normalization and cache writes. Context accepts missing `q_pe` only when the complete FP8 Q buffer proves RoPE was already applied. Keep current `metadata.py`; its draft sliding-table changes already contain the exact source behavior. Keep current generic `mla.py` and port the two fused q_b helpers and all state transitions into the relocated DSv4 module. Preserve the main coupled fused-Q/fused-KV path and add the new complete-Q state as an alternative.
- Evidence: no fresh-main Python symbol is missing; the two exact source-added MLA helpers exist under the relocated module with the same names. Every changed Python file passes `py_compile`; the C++ schema, Python wrapper, and `MlaParams` each contain one `q_rope_applied` field; custom-op implementation and both kernel/integration tests remain; no markers remain and `git diff --cached --check` passes. Compile validation is deferred to batch-xdr.
- Status: applied, staged, and static-reviewed in rebase commit 53/76.


#### Clean commits 54-61 / KPI, FMHA metadata, CFT endpoints, spill windows, tests, and CLC bounds

- Conflict/risk: no textual conflicts. Two source commits remove work already absent on fresh main, while the remaining commits update performance metadata, CFT workspace ownership, spill-window execution, Rubin coverage, and launch bounds.
- Function/symbol scope: VR200 KPI data, FMHA instruction hashes, per-workspace CFT endpoints, obsolete keepalive and flashinfer warmup paths, windowed spill processing, DSpark MegaMoE coverage, and oversized-CLC rejection.
- Human direction: not required - clean changes or changes already absorbed by main.
- Resolution: retain the six applicable clean commits in source order. Allow the keepalive and flashinfer-warmup commits to drop because fresh main already removed those paths.
- Evidence: all six applied commits pass `git show --check`; affected Python files pass `py_compile`; the worktree was clean before commit 62.
- Status: reviewed and accepted as redo commits 54-61/76; the two already-upstream removals dropped as empty.

#### DSv4 fused MLA prologue / already-upstream implementation (redo commit 62)

- Conflict: Rubin commit `316c309512` applies the original monolithic TRTLLM-14597 fused-prologue implementation across MLA CUDA, THOP, `modules/mla.py`, and the old `deepseek_v4.py`. Fresh main contains mainline commit `00c9d469e8` with the same feature/title, the later follow-up `0eeda343df`, and a relocated/evolved DSv4 implementation. Replaying the old files would overwrite main API evolution and the compatible Rubin q_b extension resolved in commit 53.
- Function/symbol scope: `deepseekV4QNormFusedKernel`, `mlaKvNormRopeQuantGenerationKernel`, `mlaKvNormRopeQuantContextKernel`, `invokeMLAKvNormRopeQuantGeneration`, MLA RoPE THOP APIs, scheduler-buffer preparation, fused-Q and fused-KV dispatch, and DSv4 prologue tests.
- Human direction: use one method that fits both main and Rubin without asking; ask only when no compatible union exists.
- Resolution: preserve the implementation already in fresh main and the Rubin q_b complete-Q extension layered on it. Drop source commit 62 as semantically upstream instead of restoring the obsolete monolithic module or narrowing the expanded current APIs.
- Evidence: `origin/main` contains `00c9d469e8` and follow-up `0eeda343df`; `git range-diff` identifies `316c309512` and `00c9d469e8` as versions of the same commit. The current relocated DSv4 module contains the fused scheduler, Q-RoPE, KV-normalization, generation, and context paths, while commit 53 preserves Rubin `q_rope_applied` support on top.
- Status: decision recorded before skipping redo commit 62/76; post-skip checks pending.


#### DSpark fused attention and RMSNorm RoPE / mainline supersedes source (redo commit 63)

- Conflict: Rubin commit `9dfb62dc22` adds the first fused DSpark attention/RMSNorm-RoPE implementation; fresh main contains mainline commit `be2732912b` with the same feature and additional support diagnostics, argument validation, SM gating, and tests.
- Function/symbol scope: `is_fused_dspark_attention_supported`, fused DSpark attention custom op, DSpark RMSNorm-RoPE kernel, `DSparkAttention.forward`, and the two kernel test modules.
- Human direction: use the shared implementation when it serves both branches.
- Resolution: keep the stricter mainline implementation and skip the older Rubin source commit. This preserves the same Rubin fused path while retaining subsequent main validation and diagnostics.
- Evidence: `origin/main` contains `be2732912b`; `git range-diff 9dfb62dc22^! be2732912b^!` maps them as versions of the same commit and shows main's additive hardening.
- Status: decision recorded before skipping redo commit 63/76.


#### Clean commits 64-66 / CFT lifetime, fused q_b autotuning, and disaggregated DSv4 DSpark coverage

- Conflict/risk: no textual conflicts.
- Function/symbol scope: CFT manager release with MoE workspaces, fused SM107 q_b GEMM/norm/RoPE autotuning, and the disaggregated DSv4-Pro DSpark accuracy test.
- Human direction: not required - compatible clean changes.
- Resolution: retain all three commits in source order.
- Evidence: each applied cleanly; their patches pass `git show --check`, and changed Python is included in the ongoing syntax audit.
- Status: reviewed and accepted as redo commits 64-66/76.

#### Relocated DSv4 q_b/indexer fixes and layer-wise benchmark setup (redo commit 67)

- Conflict: Rubin commit `8523e8e1e6` fixes q_b eligibility, context/generation cumulative lengths, fused-indexer overlap ordering, and benchmark metadata in the old monolithic DSv4 and generic MLA files. Fresh main deleted those ownership locations and moved DSv4 execution into `deepseek_v4/module.py` and `indexer.py`.
- Function/symbol scope: `_is_dsv4_q_b_gemm_norm_rope_fusion_enabled`, `_dsv4_q_b_gemm_cu_q_seqlens`, `_deepseek_v4_q_b_gemm_norm_rope_fused_fp8`, `DeepseekV4Indexer._is_fused_project_mxfp4_enabled`, `precompute_aux`, `forward_sparse_attn`, and `Runner` model/default/spec-dec setup.
- Human direction: use the implementation that fits both branches without asking.
- Resolution: keep generic `modules/mla.py` and the deletion of the monolithic DSv4 file. Port the fixes to the relocated owners: evaluate q_b layout before layernorm without rejecting the split view, assert contiguity at the actual post-norm kernel input, select cumulative Q lengths per phase, record the early event before projection but issue auxiliary work afterward, allow precomputed auxiliary work with fused Indexer-Q, retain its disable switch, and preserve all clean benchmark-harness fixes. Add focused CPU tests at the relocated test path.
- Evidence: obsolete files remain deleted; all four changed Python files pass `py_compile`; no conflict markers remain; `git diff --cached --check` passes; the staged diff contains only the relocated union and benchmark changes.
- Status: applied, staged, and static-reviewed in redo commit 67/76; runtime validation is deferred to batch-xdr.


#### Jenkins Rubin image tags / reject older source downgrade (redo commit 68)

- Conflict: source commit `d60aa30bcb` changes both Rubin architectures from the already-selected 20260724 image back to 20260627 and also conflicts with fresh main's newer Artifactory registry values.
- Function/symbol scope: `LLM_DOCKER_IMAGE_RUBIN_X86_64`, `LLM_DOCKER_IMAGE_RUBIN_AARCH64`, `LLM_RUBIN_DOCKER_IMAGE`, and `LLM_SBSA_RUBIN_DOCKER_IMAGE`.
- Human direction: use the value compatible with the current main and Rubin workflow.
- Resolution: retain every fresh-main registry/non-Rubin image and retain 20260724 for both Rubin architectures. A clean auto-merge temporarily placed the older Build.groovy values in redo commit 68; they are explicitly restored in the immediately following resolution, so the final tree has no downgrade.
- Evidence: the source patch changes only 20260724 to 20260627; the earlier recorded image decision already established 20260724 as the newer Rubin dev image.
- Status: final-tree correction staged with redo commit 69/76.

#### DSpark disaggregated generation position bootstrap (redo commit 69)

- Conflict: Rubin commit `b587334f78` adds generation-position bootstrap and speculative SWA scratch trimming. Fresh main already contains evolved graph-safe `_advance_generation_state`, position-initialization tracking, beam-aware packed block trimming, and stronger tests, producing conflicts in DSpark state and cache-reuse code.
- Function/symbol scope: `DSparkWorker._advance_generation_state`, `_draft_gen_block_batched`, `KvCacheTransceiverV2._create_kv_slice`, packed beam helpers, DSpark worker tests, cache-reuse tests, PyExecutor spec-dec logging, and multinode DSpark configuration.
- Human direction: keep a shared method when it satisfies both branches.
- Resolution: retain fresh main's graph-safe state helper and beam-aware cache-slice implementation; these are strict supersets of the source inline updates and scratch-tail removal. Retain all stronger main tests, including CUDA-graph replay and beam-tail cases. Keep the clean source additions for spec-dec iteration logging and the DSpark disaggregated accuracy configuration. Restore the 20260724 Build.groovy values here to cancel commit 68's transient clean auto-merge downgrade.
- Evidence: current DSpark tests already include the source bootstrap test plus graph-replay coverage; current cache tests include the source boundary cases plus beam-aware cases. All affected Python files pass `py_compile`; no markers remain and `git diff --cached --check` passes.
- Status: resolved, staged, and static-reviewed in redo commit 69/76.


#### Ubuntu Rubin fake-driver dependencies, L0 Groovy, and attribution metadata (redo commit 70)

- Conflict: source commit `c350cd5242` prepends a second Ubuntu dependency-install block for the Rubin fake driver, while fresh main already installs `libzmq3-dev` and `ccache` in one consolidated apt transaction and configures pip later.
- Function/symbol scope: `init_ubuntu`, Rubin `libmlx5` symlink, googletest safe-directory configuration, CMake installation, L0 multi-GPU stages, and Rubin dependency attribution maps.
- Human direction: use one implementation that fits both branches.
- Resolution: retain main's consolidated install flow. Add the architecture-specific forced `libmlx5.so` symlink and googletest safe-directory immediately after the RDMA package reinstall; add the CMake upgrade after pip is configured. Do not duplicate apt update, `libzmq3-dev`, or `ccache`. Retain the source L0 Groovy changes and generated Rubin dependency metadata.
- Evidence: current apt list already contains `libzmq3-dev` and `ccache`; `bash -n docker/common/install_base.sh` passes; no markers remain and `git diff --cached --check` passes.
- Status: applied, staged, and static-reviewed in redo commit 70/76.


#### Branch guard runner (redo commit 71)

- Conflict/risk: no textual conflict.
- Function/symbol scope: `scripts/run_branch_guard_tests.py` and its DSV4 branch validation contract.
- Human direction: explicitly use this runner as the successful-sync validation.
- Resolution: retain the clean commit unchanged.
- Evidence: commit applies cleanly and passes `git show --check`; execution is scheduled after the full static branch review in batch-xdr.
- Status: accepted as redo commit 71/76.

#### CUDA-graph advanced-sampling capture state (redo commit 72)

- Conflict: source commit `e5c45b511b` centralizes warmup-only cleanup in `CUDAGraphRunner` and adds regression coverage, while fresh main already has the first inline model-engine version and additional model-engine APIs.
- Function/symbol scope: `clear_capture_only_spec_state`, `clear_capture_only_spec_state_after_capture_pass`, the advanced-sampling capture pass in `PyTorchModelEngine`, and capture-override lifetime tests.
- Human direction: use the shared implementation that preserves both branches.
- Resolution: retain the complete fresh-main model engine; add the runner-owned after-pass helper and call it from the existing capture teardown. Preserve every existing test and add the skip-flag assertions plus warmup-only/real-capture transition test.
- Evidence: current-main functions are preserved; all three files pass `py_compile`; no markers remain and `git diff --cached --check` passes. The focused pytest cannot run in the login environment because `pytest` is not installed, so runtime execution is deferred to batch-xdr.
- Status: applied, staged, and static-reviewed in redo commit 72/76.


#### Fused DeepSeek V4 FP8 quantization paths (redo commit 73)

- Conflict: source commit `45b84163ce` adds fused SwiGLU quantization and Rubin BMM-output quantization, but its DSv4 output-projection methods live in generic `modules/mla.py`. Fresh main moved those methods into `attention_backend/sparse/deepseek_v4/module.py` and added mixed-batch token-first epilogue handling.
- Function/symbol scope: `silu_and_mul_fp8_quantize_1x128_packed_ue8m0`, `cute_dsl_fp8_bmm_quantize_rubin_out`, `FP8BlockScalesLinearMethod.apply`, `GatedMLP.forward`, `_rubin_dsv4_o_lora_quantized_input`, `_deepseek_v4_fp8_o_a_proj`, `_run_dsv4_o_lora_bmms`, and `project_sparse_attn_output`.
- Human direction: use one compatible implementation without asking when both contracts can be preserved.
- Resolution: keep current generic MLA. Retain all clean kernel, custom-op, Linear, GatedMLP, and tests. Port the Rubin O-projection fusion to the relocated DSv4 module: pure context or pure generation can produce the flat FP8 O-LoRA tensor plus packed R128c4 scale for `o_b_proj`; mixed batches retain fresh main's BF16 token-first fallback because phase-slicing the packed scale is not a safe shared contract. The normal inverse-RoPE path uses the same relocated helper. Union all modeling-test imports and retain the new entry-boundary test.
- Previous-resolution audit: the existing relocated O-projection test referenced `IS_CUTLASS_DSL_INTERNAL_AVAILABLE` without importing it. Add the missing import.
- Evidence: all changed Python files pass `py_compile`; the C++ schema, fake registrations, Python definitions, and consumers agree for both new operators; no markers remain and `git diff --cached --check` passes. GPU/kernel validation is deferred to batch-xdr.
- Status: applied, staged, and static-reviewed in redo commit 73/76.


#### Direct disaggregated branch guard fix (redo commit 74)

- Conflict/risk: no textual conflict.
- Function/symbol scope: direct disaggregated branch-guard execution.
- Human direction: branch guard is the required final validation.
- Resolution: retain the clean fix unchanged.
- Evidence: commit applies cleanly and passes `git show --check`.
- Status: accepted as redo commit 74/76.

#### Optimized precompiled DSpark attention API (redo commit 75)

- Conflict: Rubin commit `fb37132595` is a sequential replacement of the earlier DSpark runtime-JIT custom op with prepared cache/block pages, explicit `valid_len` and inverse-RoPE inputs, initialization-time dynamic-batch precompile, page-8 specialization, and SM107 support. The old implementation retained fresh-main diagnostic hardening, so the primary custom-op and test files conflict wholesale.
- Function/symbol scope: `is_cute_dsl_dspark_attention_supported`, `is_cute_dsl_dspark_attention_prepared_supported`, `precompile_dspark_attention`, `_compile_dspark_attention`, `cute_dsl_dspark_attention`, `cute_dsl_dspark_attention_prepared`, prepared RMSNorm/RoPE ops, `dspark_attention_forward_batched`, and DSpark worker initialization/state.
- Human direction: use the method that serves both main and Rubin; do not ask when there is one coherent shared endpoint.
- Resolution: take the newer optimized API and its matching tests as the endpoint of the earlier DSpark fusion sequence. Retain all clean caller/state changes. The new support policy covers SM100, SM103, and Rubin SM107; runtime uses only precompiled kernels and explicit valid-window state. Do not keep the obsolete runtime-JIT symbol/signature alongside it because every internal caller and test migrates atomically and the old call lacks required correctness inputs.
- Evidence: no references remain to `is_fused_dspark_attention_supported`, `_compile_fused_dspark_attention`, or the deleted old kernel module. Every new support/precompile/prepared symbol has production and test callers; `valid_len` is threaded from worker state through the model and custom op; all 12 changed Python files pass `py_compile`; no markers remain and `git diff --cached --check` passes.
- Status: applied, staged, and static-reviewed in redo commit 75/76; kernel execution is deferred to batch-xdr.


#### Full DSv4 Indexer workspace and compact-Q pipeline (redo commit 76)

- Conflict: the final Rubin commit reserves the full Indexer MQA-logits workspace, carries compact fused q_b output into context attention, bounds q_b autotuning for small/large M, and suppresses duplicate autotuner warmups. Fresh main relocated DSv4 ownership out of the monolithic sparse/MLA files and evolved the warmup path to cover multiple context/generation shapes.
- `tensorrt_llm/_torch/attention_backend/sparse/params.py::{get_indexer_mqa_logits_elem_budget,get_indexer_mqa_logits_workspace_bytes}`: retain the clean shared budget API from Rubin.
- `tensorrt_llm/_torch/attention_backend/sparse/dsa/indexer.py::_INDEXER_MQA_LOGITS_ELEM_BUDGET`: keep main's relocated Indexer and consume the shared budget helper; remove the duplicated local environment parsing. The obsolete monolithic `sparse/dsa.py` remains deleted.
- `tensorrt_llm/_torch/pyexecutor/_util.py::{get_indexer_mqa_logits_workspace_reserve,KvCacheCreator}`: subtract the DSv4 Indexer workspace cap before main's existing backend-declared attention-workspace reserve. Retain main's optional-window normalization and FP8 context-MLA reservation.
- `tensorrt_llm/_torch/attention_backend/sparse/deepseek_v4/module.py::{_deepseek_v4_q_b_gemm_norm_rope_fused_fp8,forward_context_sparse_attn}` and `tensorrt_llm/_torch/attention_backend/trtllm.py`: port Rubin's compact context-Q shape carrier to the relocated DSv4 owner and consume fused quantized buffers before any full-Q view; retain the normal generation placeholder and unfused fallback.
- `tensorrt_llm/_torch/custom_ops/cute_dsl_custom_ops.py::CuteDSLDsv4QbGemmFusedRmsnormRopeQuantRunner`: keep main's `Type` import/API and Rubin's `dataclasses.replace` large-M tuning configuration. Select the large-M profile only at the established bucket boundary.
- `tensorrt_llm/_torch/pyexecutor/model_engine.py::{PyTorchModelEngine.__init__,_run_autotuner_warmup}`: keep main's multi-shape context/generation loop and `ran_forward` pipeline cleanup, and add Rubin's completed-token-shape guard. Record the shape only after at least one forward completed.
- `cpp/tensorrt_llm/common/attentionOp.cpp::AttentionOp::enqueueContext`: retain Rubin's invalid-tensor debug-check bypass only for MLA input whose Q RoPE was already applied.
- Tests: retain the clean Rubin workspace, compact-Q, tuner-profile, and duplicate-warmup regression coverage at the current main test locations.
- Human direction: when one method fits both main and Rubin, take the compatible union without asking.
- Resolution: use the relocated/evolved main owners and port every still-applicable Rubin behavior into them; do not revive deleted ownership or discard main's newer API behavior.
- Evidence: all 12 changed Python files pass `py_compile`; no conflict markers remain; `git diff --cached --check` passes; the staged index has no unmerged paths. A caught text-substitution artifact in `model_engine.py` was restored from `HEAD` before staging and the full file then passed `py_compile`.
- Status: applied, staged, and static-reviewed in redo commit 76/76; runtime validation is deferred to batch-xdr.


#### Runtime audit: relocated CUTLASS DSL availability helper

- `tensorrt_llm/_torch/attention_backend/sparse/deepseek_v4/module.py::{module imports,get_dsv4_mla_cls}`: the rebased Rubin module retained the obsolete `tensorrt_llm._torch.modules.cute_dsl_utils` path, while both fresh main and the final rebased tree own `IS_CUTLASS_DSL_INTERNAL_AVAILABLE` in `tensorrt_llm._torch.cute_dsl_utils` and all current callers import from that shared owner.
- Human direction: when one method fits both main and Rubin, take it without asking.
- Resolution: update the import to `tensorrt_llm._torch.cute_dsl_utils`; do not add a compatibility shim or duplicate the availability symbol.
- Evidence: the old path has no file in either source tree; the replacement exports the identical symbol, is already used by DSpark, fused-MoE, compilation, and custom-op code, passes `py_compile`, and the whole Python tree has no remaining `modules.cute_dsl_utils` references.
- Status: fixed after batch-xdr single-node guard job `438721` reached model construction and exposed the stale import; guard rerun is required.


#### Runtime audit: DSpark disaggregated guard MoE backend

- `tests/integration/defs/accuracy/test_disaggregated_multinode.py::_worker_config`: the guard selected `MEGAMOE_CUTEDSL` for `/DeepSeek-V4-Pro-DSpark`, whose live checkpoint metadata is FP8 block-scale (`fmt=e4m3`, `scale_fmt=ue8m0`, block 128x128).
- Human direction: when one method fits both main and Rubin, take it without asking.
- Resolution: select `TRTLLM` for this guard. Do not weaken `MegaMoECuteDsl.can_implement`: both source and rebased implementations correctly restrict that kernel to NVFP4. `TRTLLMGenFusedMoE` explicitly supports `FP8_BLOCK_SCALES` on SM107, matching the checkpoint and Rubin hardware.
- Evidence: batch-xdr job `438762` rejected the original pairing as `MegaMoECuteDsl=quant_unsupported`; the checkpoint config reports FP8 block scales; `TRTLLMGenFusedMoE._SUPPORTED_QUANT_ALGOS` includes `FP8_BLOCK_SCALES`, and its SM gate includes 107. Job `438830` then progressed beyond backend selection and NIXL import before UCX rejected the runtime: NIXL requires UCX at least 1.21 (the current loader reports at least 1.22), while both image UCX installations are 1.20.
- Status: source-side backend mismatch corrected. Two-node completion is blocked by the validation image's UCX version, not by another source conflict; no compatible newer UCX is present in the image.


#### Runtime audit: SM107 DSv4 FMHA epilogue

- `tensorrt_llm/_torch/attention_backend/sparse/deepseek_v4/module.py::{prepare_sparse_attn_outputs,_rubin_dsv4_o_lora_quantized_input,project_sparse_attn_output}`: fresh main adds FMHA epilogue fusion; the original Rubin tree predates that producer. Rubin separately adds a quantized O-LoRA tuple and extra projection parameters to the standard output path.
- Human direction: when one method fits both main and Rubin, take it without asking.
- Resolution: disable the main-only FMHA epilogue producer on SM107 for context-only, generation-only, and mixed batches. Preserve main's epilogue on SM100/103, and preserve Rubin's quantized O-LoRA tuple, packed-scale buffer, extra projection parameters, and SM107 projection kernels through the standard output path.
- `tests/unittest/_torch/modules/test_mla_registry.py::{test_dsv4_epilogue_fusion_supports_mixed_batch_on_sm100,test_dsv4_epilogue_fusion_rejects_sm107}`: pin the architecture-specific contracts, including every SM107 batch composition.
- Evidence: jobs `438800` (baseline, score 0.190), `438888` (Q-B fusion disabled, score 0.190), `438889` (MTP disabled, score 0.038 with no matching reference), and `438904` (epilogue disabled, score 96.399 and passed) isolate the failure to the epilogue. The first mixed-only gate was disproved by exact job `438996` (score 0.190), showing pure-phase SM107 epilogue output is also affected. Focused batch-xdr job `439144` passes all four SM100/SM107 cases.
- Status: corrected source fix applied. Exact job `439118`, submitted through `scripts/run_branch_guard_tests.py`, completed successfully with GSM8K average accuracy 96.40, evaluated accuracy 96.399, and `1 passed`.


#### Finalization: `user/lizhiz/rubin-dsv4-k3-latest`

- `git history::{origin/main,user/lizhiz/rubin-dsv4-k3-latest}`: final tip is `c0c70557dd3a5033aa05343259e2d817cfc45e12`; its merge-base with `origin/main` is exactly fetched main ToT `b417fc5454a0bd1d355d791c415ebe0e64ff5544`.
- Resolution: replayed 76 source commits as 71 final commits; five were intentionally absorbed or skipped because main had already removed or superseded their ownership. No unmerged paths or conflict markers remain.
- Static evidence: the complete branch diff against main passes `git diff --no-renames --check`; every added or modified Python file passes `py_compile`.
- Build evidence: batch-xdr build job `438475` completed all 11,342 C++/CUDA targets successfully.
- Runtime evidence: focused job `439144` passed four architecture cases; scripted single-node guard job `439118` completed with evaluated accuracy 96.399 and `1 passed`; multi-node source validation reached runtime initialization but is blocked by image UCX 1.20 versus NIXL's newer UCX requirement.
- Worktree handoff: the requested local branch now points to the validated tip in `.worktrees/rubin-dsv4-k3-redo-20260818`. The original dirty checkout remains byte-for-byte in place at `bb0249bbf55916d44d94f75512aa59e0e5810642` under `backup/rubin-dsv4-k3-source-pre-redo-20260818`.
- Build artifacts: `3rdparty/fmha_sm100`, `build_images/`, and `tensorrt_llm/bindings` remain untracked and were excluded from the amended commit.
- Push status: no remote branch was updated.
- Status: superseded on 2026-08-18 when a completion-audit fetch advanced `origin/main` to `b9c6870adc469c83584abff36b916aace65235b9`.


### Refresh sync: `user/lizhiz/rubin-dsv4-k3-latest` onto latest `origin/main` (2026-08-18)

#### `git history::{origin/main,user/lizhiz/rubin-dsv4-k3-latest}`

- Conflict: not yet a content conflict; the previously validated branch had merge-base `b417fc5454a0bd1d355d791c415ebe0e64ff5544`, but a fresh `git fetch origin main` advanced main by 15 commits to `b9c6870adc469c83584abff36b916aace65235b9`.
- Human direction: not required - the requested target is latest main ToT.
- Resolution: preserve `c0c70557dd3a5033aa05343259e2d817cfc45e12` as `backup/rubin-dsv4-k3-pre-main-b9c-20260818`, then replay the current 71-commit Rubin range from `b417fc5454` onto `b9c6870adc`.
- Evidence: fresh fetch output and merge-base audit; the 15-commit main delta changes 70 paths and does not claim any current untracked build-artifact path.
- Status: applied. The refreshed tip is `aa403d82d764e80588c145abe274b11fdecad937`; its merge-base with `origin/main` is exactly `b9c6870adc469c83584abff36b916aace65235b9`, with 71 commits above main and zero unmerged paths.

#### `docker/Dockerfile.multi` / base-image arguments and current main container metadata

- Conflict: clean-merge audit. Rubin changes `BASE_IMAGE` and `BASE_TAG` for its SBSA development base; the new main delta adds the required copyright header and changes the wheel-stage context to copy `LICENSE` instead of `.gitmodules`.
- Human direction: not required - compatible union.
- Resolution: retain Rubin's `gitlab-master.nvidia.com:5005/dl/dgx/pytorch:rubin-py3-devel` base while retaining current main's header, `LICENSE` copy, and all current install stages.
- Evidence: three-tree comparison of `origin/main`, backup tip `c0c70557dd`, and refreshed tip; the resulting Dockerfile contains both contracts. This is the only commit reported changed by the 71-entry `git range-diff`.
- Status: applied automatically and static-reviewed.

#### Clean overlap audit: Jenkins, accuracy tests, test lists, and waivers

- Conflict: clean-merge audit for `jenkins/{BuildDockerImage.groovy,L0_Test.groovy}`, `tests/integration/defs/{accuracy/test_llm_api_pytorch.py,conftest.py}`, `tests/integration/test_lists/{qa/llm_function_core.txt,test-db/l0_b200.yml,waives.txt}`; these paths are touched by both the Rubin range and the 15-commit main delta.
- Human direction: not required - compatible union.
- Resolution: preserve current main's DIND MTU support, Slurm submission diagnostics, retired-Phi removals, new VisualGen registration, and waiver removals/additions; preserve Rubin's VR200/SM107 stages, DSv4 coverage, backend fixture, Rubin test-list entries, and bring-up waivers. Do not restore any main-deleted Phi test or waiver.
- Evidence: all relevant Rubin commits are patch-identical in `git range-diff`; direct final-tree review shows main's removals and additions plus Rubin's non-overlapping additions, with no duplicate test-list line or conflict marker.
- Status: applied automatically and static-reviewed.

#### Refresh replay validation

- Conflict: none; validation must be repeated because commit identities and the base changed.

#### `build_images/ucx-1.22.0` / multi-node branch-guard runtime

- Conflict/risk: the validation image provides UCX 1.20, while the current MPI warning requires UCX at least 1.22 and NIXL requires at least 1.21. This is a validation-runtime dependency, not a source conflict.
- Function/symbol scope: UCX loader, `ucx_info`, MPI/NIXL multi-node initialization, and `scripts/run_branch_guard_tests.py` environment.
- Human direction: not required - one official UCX version satisfies main, Rubin, MPI, and NIXL.
- Resolution: build the official UCX v1.22.0 tag at commit `8a6b06fb880accbb933a79cda893883872c68d9d` as a worktree-local overlay and prepend its `lib` and `lib/ucx` directories for the scripted guards. Do not modify source APIs or weaken version checks.
- Evidence: batch-xdr job `439427` completed successfully; `ucx_info -v` reports library version 1.22.0, API headers 1.22.0, the overlay `libucs.so.0` path, and revision `8a6b06f`.
- Status: runtime overlay verified; refreshed build and branch guards pending.

### Second refresh onto latest `origin/main` (2026-08-19)

#### `git history::{origin/main,user/lizhiz/rubin-dsv4-k3-latest}`

- Conflict/risk: a fresh completion-audit fetch advanced `origin/main` by two commits from `b9c6870adc469c83584abff36b916aace65235b9` to `849c9eb375bdc21ad7c49e1c552c2f7007b3c085` after the first refreshed build and single-node guard.
- Function/symbol scope: the complete 71-commit Rubin range and final branch ancestry.
- Human direction: not required - latest main ToT is the explicit target.
- Resolution: preserve `aa403d82d764e80588c145abe274b11fdecad937` as `backup/rubin-dsv4-k3-pre-main-849c-20260819`, cancel only the pending stale-base multi-node job, and replay the unchanged Rubin range onto `849c9eb375`.
- Evidence: rebase completed all 71 commits with no conflict; the new tip is `458274fa859d239c16526014f6e217e563c90bf1`, its merge-base is exactly `849c9eb375`, and all 71 entries are patch-identical in `git range-diff`.
- Status: applied automatically; refreshed validation pending.

#### MiniMax-M3 and Whisper clean-overlap review

- Conflict/risk: the two new main commits change `modeling_minimaxm3.py`, `llmapi/llm.py`, MiniMax/Whisper tests, test-list registrations, and remove the fixed MiniMax waiver; the Rubin range also carries general runtime and waiver changes.
- Function/symbol scope: `_validate_sparse_attention_runtime_config`, `MiniMaxM3Model.__init__`, `_WhisperSuppressTokensLogitsProcessor::{_reset_index_caches,_cached_index,_apply}`, and `tests/integration/test_lists/waives.txt`.
- Human direction: not required - the current-main behavior and Rubin behavior are non-competing.
- Resolution: retain main's MiniMax sparse-runtime validation, Whisper immutable token tuples/device index caches, matching tests, registrations, and waiver removal; retain the patch-identical Rubin changes. No source edit is needed.
- Evidence: direct function-context review shows the new behavior is MiniMax/Whisper-specific; the full Rubin `range-diff` is patch-identical, the branch diff passes `git diff --no-renames --check`, and there are no unmerged paths.
- Status: compatible union applied automatically.

#### Refreshed build and guard validation

- Build evidence: job `439578` completed with exit code 0 after all 11,038 incremental targets linked, the wheel installed, Nsight Systems verified, and the final image exported. The only compile retry was tracked Git LFS hydration; the isolated `xqaDispatcher.cpp` target then passed without a source edit.
- Superseded runtime evidence: scripted single-node job `439863` passed on tip `aa403d82d7` with GSM8K average/evaluated accuracy 96.59/96.588 and `1 passed`; it is retained as evidence but does not finalize the newer tip.
- Status: scripted single-node job `439942` and its subsequent multi-node job are the authoritative validation for tip `458274fa85`.


#### `tensorrt_llm/_torch/attention_backend/trtllm.py::TrtllmAttentionMetadata.prepare`

- Conflict/risk: current main block-table staging optimization computes `max_kv_len` with `Tensor.max()`. Rubin disaggregated/uGPU warmup can legitimately schedule an empty batch on one engine after KV-cache capacity filtering, so multi-node guard job `440029` failed during initialization with `RuntimeError: max(): Expected reduction dim to be specified for input.numel() == 0`.
- Human direction: not required - one method fits both main and Rubin.
- Resolution: define the maximum KV length as zero only when `num_seqs == 0`; retain main exact reduction and bounded `max_blocks` optimization for every non-empty batch. Leave empty-batch block-table staging uncapped (`max_blocks=None`) so no zero-width staging assumption is introduced. Preserve Rubin valid empty-warmup behavior rather than forcing a synthetic request.
- `tests/unittest/_torch/speculative/test_eagle3.py::test_block_offsets_staging_accepts_empty_batch`: add focused coverage that an empty batch prepares successfully, reports zero sequences to the cache manager, and does not request a bounded staging width.
- Evidence: the failure occurred after UCX/NIXL endpoint exchange, model loading, autotuning, and CUDA-graph setup, isolating it from the earlier transport issue. The main-side non-empty optimization and Rubin-side empty warmup contract are both retained. Focused batch-xdr job `440067` is pending.
- Status: source fix applied and static diff check passed; focused test and guard rerun pending.


#### Correction: `TrtllmAttentionMetadata.prepare` workaround superseded

- Conflict/risk: job `440098` proved the zero-sequence batch was not a valid Rubin warmup shape. The DSpark generation cache had `max_tokens=0` and `max_seq_len=-4`; the warmup generator then produced `-9` tokens. Allowing an empty reduction would only move the failure and hide the invalid capacity.
- Resolution: remove commit `c1b5215db6` source/test changes in full. Do not weaken main `max_kv_len` assertion or treat negative cache capacity as a valid API state.
- Evidence: `440098` passed the former empty reduction and then failed the preserved main assertion `0 <= -4`, while its memory logs identified the capacity deficit.
- Status: superseded in amended commit `2fdaf5c79a`; no `trtllm.py` or `test_eagle3.py` delta remains.

#### `tests/integration/defs/accuracy/test_disaggregated_multinode.py::_worker_config`

- Conflict/risk: the context-only worker fits with `free_gpu_memory_fraction=0.6`, but the DSpark one-model generation worker peaks at 269.28-270.27 GiB. At 0.6 it reports 9.93-10.52 GiB before the required 4 GiB Indexer workspace reserve and only 5.93-6.52 GiB afterward, below the KV-cache fixed cost of 8,136,949,760 bytes. That yields zero token capacity and a negative effective maximum sequence length.
- Human direction: not required - a role-specific setting preserves both main runtime invariants and Rubin full-workspace behavior.
- Resolution: retain `0.6` for the context worker and use `0.9` only when `enable_dspark` is true. Preserve the 4 GiB full Indexer workspace budget, main KV-cache assertions, the TRTLLM FP8 MoE backend, and all existing disaggregated API parameters.
- Evidence: the 0.9 fraction applies to the measured free-memory window, leaving enough room for the 8.14 GiB fixed cache cost plus token pages after the explicit 4 GiB workspace reserve; no production API or cache implementation is weakened.
- Status: applied as commit `2fdaf5c79a`; authoritative multi-node guard rerun pending.

#### `git history::{dff3ab7177,fb37132595,75589999bb}` / DSpark acceptance-preserving merge audit

- Conflict/risk: completed multi-node jobs `440145` and `440261` reached all 1,319 GSM8K responses but scored 92.608 and 92.381, below the unchanged 92.697 threshold. The preserved original checkout contains merge `dff3ab7177` (`dspark-attn-acceptance-preserving`), while the linear redo history does not retain that merge ancestry, so the merge result had to be checked before a compile-fix loop.
- Function/symbol scope: `DSparkWorker`, `dspark_attention_forward_batched`, `_rmsnorm_rope_batched`, the DSpark CuTe DSL kernels/custom ops, and their speculative unit tests.
- Human direction: not required - this is a tree-equivalence audit, not a competing API choice.
- Resolution: do not replay the merge or invent a replacement. `git diff dff3ab7177^2 dff3ab7177` is empty for all DSpark production and test paths, proving the merge result is identical there to its optimized second parent `fb37132595`. `git range-diff` maps that second-parent patch to current commit `75589999bb`, so the acceptance-preserving payload is already present despite the intentionally linear ancestry. Later current-tree differences are additive main/Rubin fixes (`num_heads` validation, SM100f fused-kernel gating and eager weight application, KV-window validation, graph-safe generation-state extraction, guided decoding, and current tests).
- Evidence: bounded parent/tree comparisons plus function-context review; no source edit was made from the disproven ancestry hypothesis.
- Status: audit complete; merge content is preserved.

#### `scripts/run_branch_guard_tests.py::dsv4-multi-node-test_gsm8k_1p1d_dep4` accuracy retry

- Conflict/risk: two end-to-end executions missed the statistical threshold narrowly after otherwise successful initialization, transfer, serving, and generation. Lowering the threshold/reference or changing an API/config merely to force a pass would invalidate the guard.
- Function/symbol scope: the exact scripted two-node DSpark guard, UCX 1.22 overlay, and unchanged greedy GSM8K evaluation contract.
- Human direction: retry requested explicitly; no API decision is involved.
- Resolution: retain commit `2fdaf5c79a` and all guard parameters unchanged, including the role-specific cache sizing, full Indexer workspace, TRTLLM MoE backend, 1,319 samples, reference 95.9, and threshold 92.697. Resubmit through the branch guard runner on `batch-xdr` with the verified UCX 1.22 overlay.
- Evidence: job `440347` was an invocation-only failure because its relative image path was interpreted as a registry name. Corrected job `440349` used the prior jobs' absolute image path, completed in 16:05, evaluated all 1,319 samples at 92.798, exceeded the unchanged 92.697 threshold, and reported `1 passed` on both node-side pytest processes.
- Status: passed on current tip `2fdaf5c79a`.

### Third refresh onto latest `origin/main` (2026-08-19)

- `git history::{origin/main,user/lizhiz/rubin-dsv4-k3-latest}`: the final post-validation fetch advanced main from `849c9eb375` to `18b0eafe6e`. Preserved the validated tip as `backup/rubin-dsv4-k3-pre-main-18b0-20260819`, then replayed the exact 72-commit range onto the new ToT. Remote command timeouts interrupted the process twice without conflicts; continuing the existing rebase completed it with exactly 72 commits, no unmerged paths, and merge-base exactly `18b0eafe6e`.
- Clean-overlap scope: main changed PEFT import locations, KV-cache host-tier quota testability, Helix overlap position IDs, waivers/test lists, and related executor files. Rubin patches in those paths remained cleanly unioned. The Helix behavior is gated by `_has_cp_helix` and the DSpark guard is non-Helix; `_sync_host_tier_quota` extracts the existing fleet-min reduction without changing its behavior; PEFT changes only relocate imports.
- Range-diff evidence: 70/72 entries are patch-identical. Commit 7 differs only because current main replaced one unrelated DeepSeekV3Lite waiver while Rubin retains its Rubin bring-up waiver. Commit 71 differs only in hunk context because main relocated `LoraConfig`/`LoraModelConfig` imports; its full Indexer workspace patch is unchanged. No API direction competed, so no human choice was needed.
- Resulting tip: `2ba3f11abcca6aad9cc1166638afe7bd08ed92fe`, 72 commits above `origin/main`.

#### `scripts/run_branch_guard_tests.py::{dsv4-single-node,dsv4-multi-node-test_gsm8k_1p1d_dep4}` final-tip validation

- Resolution: run both guards from the final tip with the existing build image, editable repo install, `batch-xdr`, and verified UCX 1.22 overlay. The latest-main C++ delta is confined to the MoE microbenchmark THOP wrapper and is outside both guards, so no extension rebuild was required.
- Single-node evidence: job `440382` completed in 16:24; GSM8K average/evaluated accuracy 96.29/96.285 versus reference 96.000; `1 passed`.
- Multi-node evidence: job `440383` completed in 16:18; GSM8K average/evaluated accuracy 92.87/92.873 versus unchanged threshold 92.697; context and generation node pytest processes both reported `1 passed`.
- Status: both authoritative guards passed on final tip `2ba3f11abc`.

### Fourth refresh onto latest `origin/main` (2026-08-19)

- `git history::{origin/main,user/lizhiz/rubin-dsv4-k3-latest}`: the post-guard fetch advanced main from `18b0eafe6e` to `36ae3f0f08` by two commits. Preserved `2ba3f11abc` as `backup/rubin-dsv4-k3-pre-main-36ae-20260819` and replayed the same 72 commits with a detached timeout-safe rebase process.
- Function/symbol scope: Gemma4 shared-KV MTP draft classification in `resolve_mtp_checkpoint_source`, `update_spec_config_from_model_config`, `get_draft_model`, and `DecodingBaseConfig`; GVR top-K decode unit-test parametrization.
- Human direction: not required - the new behavior is model/type gated and does not compete with Rubin DSpark APIs. `update_spec_config_from_model_config` returns immediately unless the config is `MTPDecodingConfig`; the guard uses `DSparkDecodingConfig`. The GVR commit is test-only.
- Resolution: retain both main commits and replay Rubin unchanged. All 72 `range-diff` entries are patch-identical; final tip is `e4ff68305a2fcb5f65a9d0c3b90ce86b0941e7dd`, merge-base is exactly `36ae3f0f08`, and the branch remains exactly 72 commits above main.
- Status: applied with no conflicts or unmerged paths.

#### Final-tip guard applicability after the fourth refresh

- Resolution: retain completed runtime jobs `440382` and `440383` as authoritative for the unchanged DeepSeek V4/DSpark runtime patches because the two new main commits are provably Gemma4/MTP- or GVR-test-only. Do not spend another two full model runs on code paths neither guard can execute.
- Evidence: direct function-context review confirms the MTP type guard; full `range-diff` is patch-identical; `scripts/run_branch_guard_tests.py --scope all --dry-run` on `e4ff68305a` resolves the same single-node and multi-node selectors and exact test files.
- Status: final-tip validation complete by successful runtime guards plus post-replay path/applicability proof.

#### `tensorrt_llm/_torch/models/dspark/attention.py::dspark_attention_forward_batched` explicit-validity dispatch correction (superseded)

- Conflict/risk: the fourth-refresh integration commit `19a93556d4e` misread explicit `valid_len` as a hole-aware contract unsupported by the fused kernels and changed both dispatch predicates to `not has_explicit_validity`. Production always passes `DSparkWorker._valid_len[slots]`, so this semantic cherry-pick failure disabled the prepared and direct fused attention paths for every production request.
- Function/symbol scope: `DSparkWorker._draft_gen_block_batched`, `dspark_attention_forward_batched`, `cute_dsl_dspark_attention_prepared`, `cute_dsl_dspark_attention`, and `test_dspark_attention_forward_batched_matches_fallback`.
- Correct source endpoint: early versions of 10443/PR18061 did not fully handle the target branch's new `valid_len`, but the final kernel does. It accepts `valid_len` and recognizes an initialized ring-buffer suffix with `(start_pos - physical_slot) % window_size < valid_len`; the prepared custom op accepts and forwards the same tensor.
- Resolution: remove the `not has_explicit_validity` gates from both fused dispatches. Pass explicit production `valid_len` directly; only synthesize `min(start_pos + 1, window_size)` for legacy callers that omit it. Preserve the PyTorch implementation solely as the unsupported-shape/backend fallback, not as the explicit-validity path.
- Evidence: the final custom-op tests cover partially filled and wrapped suffixes, including `start_pos=[257, 390]` with `valid_len=[3, 5]`; prepared CUDA-graph replay updates and forwards `valid_len`; production passes `self._valid_len[slots]`. The regression now requires prepared fused dispatch for both explicit and synthesized validity and compares its output/cache against the PyTorch reference.
- Status: the earlier resolution and its claimed dispatch semantics are superseded. The source fix is applied on `user/lizhiz/rubin-advance`; `py_compile` and `git diff --check` pass. Focused pytest is pending an environment with pytest and a supported GPU.



#### `jenkins/Build.groovy` / Rubin dev-image constants (supersedes redo commits 68-69 image resolution)

- Conflict/risk: redo commit 68 intentionally replayed source commit `17ca87dcf7`, which points the Rubin x86_64 and SBSA dev builds at `20260627-upgraded`. Redo commit 69 (`85ec068814`, DSpark position bootstrap) then incorrectly restored both constants to `20260724-upgraded`, even though that image change is unrelated to the DSpark commit. The earlier ledger entries that rejected `17ca87dcf7` and restored 20260724 are superseded.
- Function/symbol scope: `LLM_DOCKER_IMAGE_RUBIN_X86_64` and `LLM_DOCKER_IMAGE_RUBIN_AARCH64`. `jenkins/current_image_tags.properties` is not changed because `17ca87dcf7` deliberately touched only `Build.groovy`.
- Human direction: respect `17ca87dcf7`, recompile the image, and rerun the disaggregated accuracy guard only after verifying the rebuilt runtime.
- Resolution: restore both `Build.groovy` Rubin dev-image constants to `20260627-upgraded`; retain the DSpark position-bootstrap code from `85ec068814` without its unrelated image reversal.
- Evidence: the exported raw `20260724-upgraded` sqsh contains PyTorch `2.14.0a0+4fdf77b940.nvinternal.rubin` and FlashInfer 0.6.15, while the compiled image used by successful guards `440382`/`440383` contains PyTorch `2.13.0a0+8145d630e8.nvinternal.rubin` and FlashInfer 0.6.16. Jobs `440588` and `440778` loaded the 2.13-built worktree extensions under the raw 2.14 image: the source-level CUDA registration for `deepseek_v4_compute_sliding_block_tables` appeared as HIP at runtime and executor warmup failed before accuracy execution. Job `440778` reproduced this on l05 nodes, disproving the earlier node-class diagnosis, and was canceled during failed-rank teardown.
- Status: source resolution applied; corrected image rebuild and static runtime-registration validation pending. No further Slurm accuracy job may be submitted until the rebuilt image reports PyTorch 2.13 and a CUDA kernel for `trtllm::deepseek_v4_compute_sliding_block_tables`.


#### `tensorrt_llm/_torch/disaggregation/transceiver.py::KvCacheTransceiverV2._create_kv_slice` gen-init draft-tail accounting

- Conflict/risk: fresh main commit `dff59cd5c7` populates placeholder `py_draft_tokens` before scheduling a one-model speculative `DISAGG_GENERATION_INIT` request. The generation receiver therefore allocates `prompt_len + draft_len + num_extra_kv_tokens`, while PR #17619 scratch-tail trimming still reconstructed the allocation as only `prompt_len + num_extra_kv_tokens`. When the omitted DSpark draft reserve crosses a 128-token page boundary but the four extra-KV tokens do not, an uninitialized receiver tail page survives and the subsequent SWA suffix trim drops one initialized prompt page. This interaction is exclusive to disaggregated generation with speculative decoding.
- Function/symbol scope: `KvCacheTransceiverV2._create_kv_slice`, `get_draft_token_length`, and `TestCreateKvSliceTokenRange::test_dspark_gen_init_draft_reserve_is_removed_before_swa_trim`.
- Human direction: not required - this is not an API choice; one internal accounting rule matches the scheduler and transfer contracts.
- Resolution: include the live request draft length when reconstructing allocated receiver blocks, gated by `is_generation_only_request()` so a context sender that merely publishes draft tokens does not over-trim its prompt allocation. Remove the complete receiver-only tail before selecting the valid SWA prompt suffix. Keep `TokenRange.end == prompt_len`; draft and extra-KV slots remain untransferred.
- Evidence: current scheduler, gen-init allocation, and transceiver call-order review; a focused 128-token boundary regression reproduces the omitted-page case. `git diff --check` and `py_compile` pass. OCI-AGA image validation job `517665` passed the focused regression in the supplied image with editable repo install (`1 passed`, exit code 0). Authoritative job `517765` reused job `516649` exactly, including `dsv4_ucx_plugin`, the supplied image, mounted editable repo, two-node four-GPU-per-node topology, NVFP4 checkpoint, and unchanged accuracy contract. It completed in 30:53 with GSM8K 96.475 versus reference 95.900 and threshold 92.697; both node-side pytest processes reported `1 passed` and Slurm exited 0. A preceding job, `517706`, omitted the UCX plugin and failed before serving, so it is not accuracy evidence.
- Status: fixed and validated.

### K3 feature replay from `feat/rubin-k3-latest` (2026-08-20)

#### Source inventory and replay boundary

- Source remote/ref: GitLab `feat/rubin-k3-latest` at `e67bfc7dac1ccc95822f631d08324066f2fee55b`.
- Exclusive lower boundary: `c61a248aed1a2b1d697b3b9dab47f72a2d3379cc` (not replayed).
- Target: `codex/k3-features-after-c61-20260820`, created from `user/lizhiz/rubin-dsv4-k3-latest` at `75ae54c6`; backup ref `backup/rubin-dsv4-k3-before-k3-features-20260820` preserves that starting point.
- Worktree: `~/workspace/tllm_k3/.worktrees/k3-features-after-c61-20260820`.
- Inventory: 48 graph commits after the boundary: 37 non-merge candidates and 11 merge commits. The merge commits were deliberately dropped and their first-parent payloads were audited through the 37 candidates: `e71f49090c`, `9a823473b8`, `452fb6f77c`, `345c3aa7ed`, `cf0fe5f9dc`, `5cd34e938c`, `a32d4564dd`, `b2a6fd17c6`, `ef95d6fd2e`, `cf44b33ab3`, and `3c0b90f9d0`.

#### Applied source commits

- `6aa1a734` -> `77734dba99`; `c585d20e` -> `dbee3710e8`; `afa1035a` -> `6bce3d2a20`; `0ff1588a` -> `68e8bb5533`; `e54e341e` -> `62757752dd`.
- `2463b0d` -> `f881c6ebaa`; `9b280b6` -> `19e595a0fd`; `e71c30d` -> `ddadc177f5`; `57bc18e` -> `19a707925c`; `ab7d326` -> `21dba7ee0b`.
- `6e1ab573` -> `58741b5227`; `c6bfcd95` -> `875a1d911d`; `ad1c94d5` -> `174a5f9b79`; `d47257c8` -> `9df6b36df6`; `c82a94bf` -> `d4832c7b92`.
- `6bd818ce` -> `68e8814a43`; `3a6cab1e` -> `619cda0c9e`; `9f958a49` -> `4c6193d324`; `fd18c75b` -> `ba4d6ed37d`; `fd63632d` -> `a13f8c9b8a`.
- `b2f56ceb` -> `13aa78ab3c`; `072b0cd2` -> `6fce9c1429`; `3fe11a61` -> `af4a56c0a4`; `0c0179cb` -> `f0f9862652`; `b9e2e4b3` -> `0d16b74b67`.
- `be8c2be8` -> `652a7f5309`; `6466e966` -> `31ca299182`; `3f0afebd` -> `a4532b296e`; `e67bfc7d` -> `94a7e2d3bc`.

#### Intentionally skipped source commits

- `adc79fa188` (`add Kimi K3 SiTU MegaMoE support`): the target already has the complete, newer Kimi architecture/MegaMoE implementation; later focused SiTU commits were replayed separately.
- `0ded3d5140` (`Shard Kimi K3 dense and shared-expert MLPs with tensor parallelism`): obsolete against the target's newer loader and direct TP ownership.
- `a8335949f9` (`sync auto host-tier quota across ranks`) and `4438f151cb` (`saving last N turns in per-conversation policy`): patch-equivalent changes are already present in the target.
- `2b787ce9c0` (`Pin Kimi MegaMoE to upstream DeepGEMM`): the target already carries the same pin through newer dependency state.
- `5d86e96249` (`partial rewind endpoints reusable via per-page token coverage`): the target already contains a newer implementation of the same KVCM-V2 contract.
- `8170f22c55` (`CuteDSL FP8/FP16 MLA decode attention fmha lib`): the target already contains the newer CuTeDSL MLA implementation; the SM107 enablement was replayed separately.
- `703862356d` (`disagg benchmark server health timeout configurable`): the target already contains the newer configurable-health-timeout behavior.

#### Three-tree conflict resolutions

- KDA/attention kernels (`2463b0d`, `9b280b6`): unioned source byte-movement and residual-RMSNorm fusion with target bounds checks, device-specific layouts, and current launch contracts.
- Route quantization and PDL (`e71c30d`, `57bc18e`): retained the target route-dtype union and adapted source PDL behavior to the shared `situ.py` implementation.
- MLA decode (`ab7d326`, `d47257c8`, `3a6cab1e`, `9f958a49`): preserved target workspace/counter-buffer/RoPE handling and split-loader APIs while applying the decode-performance, backend-default, no-padding, and SM107 behavior.
- KVCM/ShortConv (`6e1ab573`, `ad1c94d5`, `6bd818ce`): kept the all-lifecycle stale-walk guard and target deletion of the obsolete packed-prefill test while applying identity-RoPE and D2H-copy fixes.
- NVFP4 and resident FP8 (`fd18c75b`, `fd63632d`, `6466e966`): relocated source finalization into target `_load_expert_slices`/split-loader ownership, preserved target MegaMoE streaming safety, passed explicit resident-FP8 attention jobs, removed the old KDA FP8 read path, and retained the target deletion of its obsolete test.
- MoE SiTU (`b9e2e4b3`, `3f0afebd`): preserved target `gather_act` naming, Relu2 support, precomputed-tactic and uGPU APIs; added SiTU beta validation, cache identity, SM107 epilogue, and beta propagation through both single- and two-partition paths.
- Final lint commit (`e67bfc7d`): retained current-target formatted versions of conflicted files and applied the remaining clean quantization formatting hunk.

#### Replay validation state

- Every resolved Python file was checked with `py_compile`; changed shell scripts were checked with `bash -n`; each cherry-pick passed `git diff --check` before completion.
- Relevant pre-commit hooks pass after formatter output is staged. The repository-wide `validate-test-lists` hook still reports three unrelated pre-existing stale entries for `accuracy/test_llm_api_autodeploy.py::TestNemotronH` at `tests/integration/test_lists/waives.txt:291-293`.
- The final 95-file sweep passed YAPF, Ruff, Ruff-format, baseline-gated Ruff, clang-format, codespell, `py_compile`, `bash -n`, and `git diff --check`. It also closed three replay-integration gaps found by lint: the missing `Tuple` import, the omitted CuTeDSL MLA workspace helpers, and the omitted `TRTLLMGenFusedMoE` scheduler import.
- Targeted pytest was not run: the `tllm-dev` workflow requires an explicit GPU host and container, and neither was supplied. No remote branch has been pushed.


### Import GitLab MR !10518 onto `user/lizhiz/rubin-advance` (2026-08-25)

#### MR boundary and already-present commits

- Conflict/risk: GitLab reports 13 linear commits from `ba828e3772` to `2d4a8dd759`, but the
  target already contains the original commits corresponding to the MR's first three cherry-picks:
  `7e8eb8f1a1`, `92d1a8f4fa`, and `70bf7e4fd1`.
- Human direction: not required - patch-equivalent work must not be replayed.
- Resolution: squash only the remaining MR range `fca53393d0..2d4a8dd759` and preserve the MR
  title, number, author, URL, and a Git-generated DCO sign-off.
- Evidence: the MR API reports base `ba828e3772`, head `2d4a8dd759`, 13 commits, and zero merges;
  target history contains the three original source commits.
- Status: replacement squash `ee56b092af` created; its resolved tree is staged and ready to commit.

#### MegaMoE kernel bundle and `MegaMoECuteDsl` capability/SiTU contracts

- Conflict: MR !10518 replaces the MegaMoE scheduling and epilogue implementation and narrows the
  supported architectures to SM100/103/107. The target has a newer pure
  `can_implement(MoEProblem, MoEDeployment)` resolver contract plus Kimi K3 SiTU activation, beta
  validation, compiled-kernel cache identity, top-16 routing, and beta propagation.
- Human direction: not required - compatible union.
- Resolution: retain the target's structured capability API and frozen-environment dependency
  checks, apply the MR's exact SM100/103/107 gate, and union the coherent MR scheduling/TMA-stage
  implementation with every target SiTU constructor, epilogue, cache-key, and call-path field.
- Evidence: `ba4d6ed37d` is the only target-only production commit touching the four conflicted
  kernel/backend files; its complete patch was reviewed. All four resolved files pass `py_compile`;
  the capability path contains no host SM probe, the runtime probe retains its required
  `get_sm_version` import, and SiTU plus the MR TMA/accumulator controls are present end to end.
- Status: applied, staged, and syntax-validated.

#### MoE guide, microbenchmark rendezvous, and backend unit tests

- Conflict: the MR documentation predates the target's resolver and current Marlin/CuteDSL matrix;
  its benchmark helper removes CuteDSL process-group setup but also simplifies DeepGEMM rendezvous;
  the unit-test conflict independently adds `ModuleType` versus the target's `Tuple` import.
- Human direction: not required - compatible union.
- Resolution: keep the current guide and update only MegaMoE's architecture/provider/kernel
  description; retain current capability facts for unrelated backends. Stop benchmark-side process
  group setup for the new CuteDSL provider, while retaining HashStore for one-rank DeepGEMM and MPI
  rendezvous broadcast for multi-rank DeepGEMM. Import both `ModuleType` and `Tuple` in the tests.
- Evidence: the MR-only diffs and current implementations were compared directly; the resolved
  benchmark and unit-test modules pass `py_compile`.
- Status: applied, staged, and syntax-validated.

#### Validation

- All staged Python files pass `py_compile`, `git diff --cached --check` passes, and all applicable
  pre-commit hooks pass except `validate-test-lists`, which reports the same three unrelated,
  pre-existing stale `TestNemotronH` waiver entries at `waives.txt:291-293`.
- Focused pytest could not run because the available Python environment has no `pytest` module.
