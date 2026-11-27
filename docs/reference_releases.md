# Reference model release lifecycle

## Retention policy

- Maintain one active reference directory in the working tree. The DVC pointer is tracked in Git.
- Treat `models/reference_history/` as temporary rollback storage, not the release registry.
- Publish every approved reference and its evidence to the existing S3 DVC remote. Keep the Git release tags and the corresponding historical DVC objects. Do not run remote garbage collection or configure S3 expiration rules that delete retained releases.
- Do not publish experimental CV fold models or MLflow/Optuna databases to this release directory.
- The legacy HistGradientBoosting release has only the original model and metadata. Its Phase 4 comparison is preserved in the XGBoost release evidence; do not invent a retrospective evidence bundle for the historical release.

## Reference layout

```text
models/
  reference.dvc
  reference/
    best_model_pipeline.joblib
    best_model_metadata.json
    evaluation/
      validation.json
      frozen_plan.json
      partition_identity.json
      heldout.json
      fairness.json
      robustness.json
      drift.json
      previous_reference.json
      manifest.json
```

The eight evidence JSON files are semantic copies of the approved embedded evidence, not byte-identical copies of the original report files. The manifest binds the files to the exact model SHA-256 and approval digest, and records each file SHA-256. Existing model and metadata bytes are preserved during migration. This bundle contains the frozen shortlist, configuration parameters and evaluation results; it does not contain every optimization trial or non-promoted model binary. Preserve the relevant experimental evidence separately before experimental cleanup if exact replay or broader audit is needed.

## One-time migration for Phase 4

Run in Git Bash from the project root, with the project's virtual environment activated. Stop any concurrent promotion or release work. Do not run the cleanup utility yet.

```bash
python -m pytest tests/test_promote.py tests/test_manage_reference.py -q
python -m scripts.manage_reference bundle
sha256sum models/reference/best_model_pipeline.joblib
```

Expected XGBoost model SHA-256: `e3d91ddbef536965481f3b614ee44c55f1f7960290849688e9d178a7f35a2711`.

Publish the expanded reference directory. Use your existing authenticated DVC write configuration for `dvc push` if needed; never commit credentials or `.dvc/config.local`.

```bash
dvc add models/reference
dvc push models/reference.dvc
dvc status -c models/reference.dvc

git add src/models/promote.py scripts/manage_reference.py tests/test_promote.py tests/test_manage_reference.py docs/reference_releases.md models/reference.dvc
git commit -m "feat: version reference models with evaluation evidence"
git push
```

The original Phase 3 source archive identifies commit `31614b0e171314548178bfffd352cfb1d3f16fd3` with the old reference DVC directory hash `bcf803e2c824ac6b16e4228aaf47beca.dir`. Check the saved rollback pointer and the Git version before tagging. Do not force or overwrite existing tags; if they already exist, verify they identify the intended releases.

```bash
git show 31614b0e171314548178bfffd352cfb1d3f16fd3:models/reference.dvc
cat models/reference_history/cd14584e198e46d5b105ac29d61c85ab/reference.dvc

git tag -a model-v1.0.0 31614b0e171314548178bfffd352cfb1d3f16fd3 -m "HistGradientBoosting reference"
git tag -a model-v2.0.0 -m "XGBoost reference with approved evaluation evidence"
git push origin model-v1.0.0 model-v2.0.0
```

First verify both releases without removal:

```bash
python -m scripts.manage_reference verify \
  --current-tag model-v2.0.0 \
  --backup-tag cd14584e198e46d5b105ac29d61c85ab=model-v1.0.0
```

Only after successful verification, repeat with explicit deletion:

```bash
python -m scripts.manage_reference verify \
  --current-tag model-v2.0.0 \
  --backup-tag cd14584e198e46d5b105ac29d61c85ab=model-v1.0.0 \
  --remove-backups

git status --short
dvc status
```

The verifier clones each tag from `origin` in a temporary directory, overrides the DVC cache with a new empty cache, pulls `models/reference.dvc` from the existing `adult-income-ai` remote, and compares the pointer and every recovered file with the expected local release. The current bundle is mandatory; historical bundles are verified when present. The original checkout is not switched. Recovery failure, missing tags, incomplete transactions, or integrity mismatches stop removal. Verification and removal share the promotion lock; the removal command repeats recovery before deletion. No remote object is deleted. The DVC cache may still contain older objects; this is distinct from keeping historical model directories in the working tree.

## Future releases

1. Complete the candidate evaluation and approval procedure separately from orchestration. Confirm promotion using the approved evidence digest.
2. Promotion stages the approved model, metadata and evidence bundle; it preserves the previous reference and DVC pointer in a transaction directory. A pending rollback transaction prevents another promotion until it is finalized. Do not manually remove its lock after an interrupted or failed rollback.
3. Run `dvc add models/reference`, `dvc push models/reference.dvc`, and `dvc status -c models/reference.dvc`. Commit and push the new pointer with the relevant code/configuration.
4. Create a new annotated `model-vMAJOR.MINOR.PATCH` tag at that commit and push the tag. Do not change previous release tags.
5. Run `python -m scripts.manage_reference verify --current-tag NEW_TAG --backup-tag TRANSACTION_ID=PREVIOUS_TAG`. Inspect the result; then repeat with `--remove-backups` to finalize. The previous tag must match the saved pointer and every backup file.

The existing candidate-evaluation and fairness scripts are still Phase 4-specific; this update does not generalize their evaluation protocols or authorize new test evaluations.

## Fresh clone and historical recovery

A fresh clone of the default branch followed by `dvc pull` retrieves the data and active reference pointed to by that branch. It does not pull all historical releases. To inspect an old release without changing a working project, use a separate clone:

```bash
git clone --branch model-v1.0.0 --single-branch <repository-url> adult-income-ai-v1
cd adult-income-ai-v1
# Install the requirements from this release in its own environment.
dvc pull
```

Historical recovery depends on Git tags, DVC pointers, compatible dependencies and retained S3 objects. Reproducing all original tuning trials also depends on preserved experiment records and the original execution environment.
