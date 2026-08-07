# Contributing to canto

Thanks for your interest! canto is an early-stage project, so the most useful
contributions right now are bug reports, test coverage, and documentation.

## Getting started

```bash
git clone https://github.com/naivestud/canto
cd canto
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Running the checks

The test-suite runs on CPU and takes a few seconds.

```bash
pytest                          # tests
ruff check .                    # lint
ruff format --check .           # formatting
mypy                            # types
```

Or install the pre-commit hooks to run lint and formatting automatically:

```bash
pre-commit install
```

## Making a change

1. Open an issue for anything substantial so we can agree on the approach.
2. Keep changes focused — one concern per pull request.
3. Add or update tests for behaviour changes.
4. Follow the existing style; the linters enforce most of it.
5. Update `docs/` and the `CHANGELOG.md` (Unreleased section) where relevant.

## Code layout

- `src/canto/` — library code (one module per concern)
- `tests/` — mirrors `src/canto/`, one `test_*.py` per module
- `examples/` — small runnable scripts
- `docs/` — prose documentation

## A note on model weights

This repo contains architecture and tooling, not pretrained weights. Please do
not open PRs that add large binary checkpoints; instead, link them from an
issue or discussion.

## Cutting a release

Maintainers release by tagging; the `Release` workflow builds and publishes the
tagged version.

1. Update `CHANGELOG.md` — move anything under `Unreleased` into a new version
   heading and date it.
2. Bump the version and tag in one step:

   ```bash
   scripts/release.sh 0.4.0
   ```

3. Push the branch and the tag:

   ```bash
   git push origin main v0.4.0
   ```

The tag triggers `.github/workflows/release.yml`, which builds the sdist and
wheel and attaches them to a GitHub release.
