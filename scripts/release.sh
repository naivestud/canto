#!/usr/bin/env bash
# Tag a release. Usage: scripts/release.sh <new-version>
#
#   scripts/release.sh 0.4.0
#
# Updates the version in pyproject.toml and canto/__init__.py, commits, and
# creates a signed-off annotated tag that the Release workflow picks up.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <new-version>" >&2
  exit 1
fi

VERSION="$1"
if [[ ! "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "version must look like 1.2.3, got: $VERSION" >&2
  exit 1
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Refuse to cut a release on a dirty tree.
if [[ -n "$(git status --porcelain)" ]]; then
  echo "working tree is dirty; commit or stash first" >&2
  exit 1
fi

sed -i "s/^version = \".*\"/version = \"$VERSION\"/" pyproject.toml
sed -i "s/^__version__ = \".*\"/__version__ = \"$VERSION\"/" src/canto/__init__.py

git add pyproject.toml src/canto/__init__.py
git commit -m "Bump version to $VERSION"
git tag -a "v$VERSION" -m "v$VERSION"

echo "tagged v$VERSION — push with: git push origin main v$VERSION"
