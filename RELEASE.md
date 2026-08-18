# Release Guide

`coreai-torch` is published to [PyPI](https://pypi.org/p/coreai-torch) by
`.github/workflows/release.yml`, which uses PyPI
[Trusted Publishing](https://docs.pypi.org/trusted-publishers/): the workflow
authenticates with a short-lived OIDC token minted by GitHub Actions for that
one job, so no long-lived API token is stored in the repo or in an org secret.

## Cutting a release

1. On a release branch, pin dependencies, bump `__version__` in
   `coreai_torch/__version__.py`, update the docs version metadata, and get the
   PR reviewed and merged. `/release stage1` walks these checks.
2. Tag the merge commit `vMAJOR.MINOR.PATCH` — matching `__version__` exactly —
   and push the tag:

   ```bash
   git tag v0.5.0 <merge-sha>
   git push origin v0.5.0
   ```

3. Pushing the tag starts the `Release` workflow. `build` verifies the tag
   matches `__version__` and builds the wheel and sdist; `smoke-test` installs
   that wheel into a clean venv on each supported Python version and imports the
   public API.
4. `publish` then waits on the `pypi` GitHub environment. A release manager other
   than the tag pusher approves it, and the vetted artifact is uploaded to PyPI.
5. Deploy the docs: `./docs/deploy.sh` and `./docs/deploy.sh --remote pie`.

To exercise the build and smoke test without publishing, run the workflow
manually — `gh workflow run release.yml --ref <branch>`. The `publish` job is
gated to tag pushes, so a dry run can never upload.

Nothing about a release is done from a laptop: the artifact PyPI receives is
built and tested by the workflow, from the tagged commit, and no developer
holds a credential that can upload it.

## Trusted Publishing setup

One-time configuration, already in place for `apple/coreai-torch`. Recorded here
so it can be audited or rebuilt.

### On PyPI

As an owner of the `coreai-torch` project, under **Manage project → Publishing →
Add a new publisher → GitHub**:

| Field             | Value                |
| ----------------- | -------------------- |
| Owner             | `apple`              |
| Repository name   | `coreai-torch`       |
| Workflow name     | `release.yml`        |
| Environment name  | `pypi`               |

The environment name is not optional here. Without it, *any* workflow in the
repo named `release.yml` can publish; with it, PyPI additionally requires the
OIDC token to assert the `pypi` environment, which is where the approval gate
lives.

For a project that does not exist on PyPI yet, the same form is available as a
**pending publisher** from your account's Publishing page; the project is
created on first successful upload.

### On GitHub

Under **Settings → Environments → New environment → `pypi`**:

- **Required reviewers**: the release managers team, with **prevent
  self-review** enabled — so the person who pushed the tag cannot approve their
  own release.
- **Wait timer**: 15 minutes, leaving time to cancel a run started by mistake.
- **Deployment branches and tags**: restrict to the tag pattern `v*.*.*`, so a
  run from a branch cannot reach the environment even if the workflow gate is
  ever loosened.

No secret is added to the environment. The `id-token: write` permission in the
`publish` job is what lets it mint the OIDC token, and it is scoped to that job
alone.

## Internal publishing

Trusted Publishing covers PyPI only. Publishing to Apple's internal Artifactory
index still uses `scripts/release.sh` with `UV_PUBLISH_USERNAME` /
`UV_PUBLISH_PASSWORD`; pass `--skip-publish` when the PyPI upload is handled by
the workflow.
