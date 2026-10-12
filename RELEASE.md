# Release Guide

`coreai-torch` is published to [PyPI](https://pypi.org/p/coreai-torch) by
`.github/workflows/release.yml`, which uses PyPI
[Trusted Publishing](https://docs.pypi.org/trusted-publishers/): the workflow
authenticates with a short-lived OIDC token minted by GitHub Actions for that
one job, so no long-lived API token is stored in the repo or in an org secret.

## Cutting a release

1. On a release branch, pin dependencies, bump `__version__` in
   `coreai_torch/__version__.py`, update the docs version metadata, and get the
   PR reviewed and merged.
2. Tag the merge commit `vMAJOR.MINOR.PATCH` — matching `__version__` exactly —
   with a signed tag, and push it:

   ```bash
   git tag -s v0.5.0 <merge-sha> -m 'coreai-torch 0.5.0'
   git push origin v0.5.0
   ```

   The tag is the trust anchor for the release: the workflow builds, tests, and
   publishes from it, so signing it is what ties the published artifact to a
   known signer. Check it shows as `Verified` on GitHub after pushing.

3. Pushing the tag starts the `Release` workflow. `build` verifies the tag
   matches `__version__`, builds the wheel and sdist, and rebuilds a wheel from
   the sdist to confirm the sdist is complete; `smoke-test` installs the built
   wheel into a clean venv on each supported Python version and imports the
   public API.
4. `publish` then waits on the `pypi` GitHub environment. A release manager other
   than the tag pusher approves it, and the vetted artifact is uploaded to PyPI.
5. Deploy the docs: `./docs/deploy.sh`.

To exercise the build and smoke test without publishing, run the workflow
manually — `gh workflow run release.yml --ref <branch>`. The `publish` job is
gated to tag pushes, so a dry run can never upload.

Nothing about a release is done from a laptop: the artifact PyPI receives is
built and tested by the workflow, from the tagged commit, and no developer
holds a credential that can upload it.

## Trusted Publishing setup

One-time configuration, per PyPI's
[Trusted Publishing docs](https://docs.pypi.org/trusted-publishers/). The
publisher must be bound to workflow `release.yml` **and** environment `pypi` —
without the environment, any workflow named `release.yml` in the repo can
publish.

The `pypi` GitHub environment carries the release policy: required reviewers with
**prevent self-review** (so the tag pusher cannot approve their own release), a
15-minute wait timer, and deployments restricted to the tag pattern `v*.*.*`. It
holds no secrets — `id-token: write` on the `publish` job is what mints the OIDC
token, scoped to that job alone.

## Other indexes

Trusted Publishing covers PyPI only. Publishing to another index still goes
through `scripts/release.sh` with `UV_PUBLISH_USERNAME` / `UV_PUBLISH_PASSWORD`;
pass `--skip-publish` when the PyPI upload is handled by the workflow.
