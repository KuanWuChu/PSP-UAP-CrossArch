# Project instructions — PSP-UAP Cross-Architecture Study

Context: 115_1作業. Read RESEARCH_README.md, docs/UPSTREAM_AUDIT.md, docs/EXPERIMENT_PROTOCOL.md, and docs/STATUS.md before editing.

## Authorized scope

Implement E0 reproduction support, E1 target-only evaluation, E2 matched synthetic-source ablations, E3 crop-area ablations. Do not implement a new attack loss, transformer-native objective, classifier fine-tuning, or ensemble as this phase's method. No parallel agents are required by this file.

## Evidence and source preservation

- Upstream commit is dbbc4ee8ee66b6cd5286b939d7254fd7edcbfb8c. Keep MIT attribution.
- loss.py MUST remain byte-identical to docs/UPSTREAM_SNAPSHOT.json in this phase. Verify with scripts/check_upstream_integrity.py.
- Preserve all upstream Python files for the faithful route; put research extensions in a separate package. Do not silently fix original bugs. Record compatibility fixes or corrections separately.
- Documentation, planned commands, tests, smoke outputs, completed training, and reproduced paper numbers are distinct statuses.
- No fabricated FR, timing, GPU usage, confidence intervals, or checkpoint availability.
- Original supplement already tested DeiT3. No first-Transformer-test claim.
- The published DeiT3 run's exact layer mapping is NOT established by the public main.py. Conv2d-only is verified for public loss.py, not for unpublished experiment code.

## Experiment discipline

- Keep activations objective unchanged; E2 RW/T toggles must be explicit and matched within each contrast.
- Synthetic crop sources are detached; gradients go through the added learnable delta. Reuse existing loss functions.
- Standardize image manifests, raw RGB perturbation units, transforms and model weight identities.
- Never use held-out target outputs to select a checkpoint or tune crop ranges; never select final result by max test FR.
- Separate upstream-faithful, corrected-reproduction, and fixed-budget research results.
- Seed Python, NumPy, torch CPU and CUDA; preserve RNG state in resumable runs. A replay claim needs evidence.
- Preserve current weights and upstream hashes; do not silently upgrade old pretrained=True weights.

## Repository workflow

Use one feature branch per PR; origin is user's fork, upstream is author's repository. Push PRs to user's fork only, never the author's repo. Do not force push or merge unrequested PRs. Leave teammate invitations to the user.

Public repo excludes ImageNet images, pretrained model binaries, personal paths, credentials and raw private logs. UAP .npy files may be shared via versioned release artifacts after checking size, metadata, provenance and hashes. Logs/CSV must use relative image IDs.

## Validation

CPU: analytic FR fixture, epsilon/channel/clamp ordering, manifest disjointness, detached crop-gradient path, deterministic sampling, zero-delta identity, resume equivalence, numerical failure handling, original loss hash. GPU: clearly labeled smoke checks before long runs. A modern Blackwell GPU does not imply torch 1.11 compatibility; inspect compute capability and software support first.

For final handoff, report branch/commit, changed paths, tests actually run, runnable commands, remaining blockers, and experiments not run.
