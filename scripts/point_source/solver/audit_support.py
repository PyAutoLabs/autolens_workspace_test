"""Provenance and pinned historical source for point-solver audits."""

import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess


HISTORICAL_SHA256 = "7a5ea76af2b856370820e0d5c1f8090ba0e270e4d4dd5677f8230f0134a4c834"
HISTORICAL_FIXTURE = Path(__file__).with_name("historical_hessian.json")


def library_provenance(module):
    """Identify the imported code, without treating an enclosing repo as its own."""
    package = Path(module.__file__).resolve().parent
    directory = package.parent
    if (directory / ".git").exists():

        def git(*args):
            return subprocess.check_output(
                ["git", "-C", str(directory), *args], text=True
            ).strip()

        return {
            "kind": "checkout",
            "sha": git("rev-parse", "HEAD"),
            "dirty": git("status", "--porcelain"),
        }

    distribution = importlib.metadata.distribution(module.__name__)
    # Require the metadata to describe the imported package, not another install
    # visible on sys.path. Editable/source trees without Git must not masquerade
    # as the installed wheel whose version happens to be visible.
    files = distribution.files
    init = package / "__init__.py"
    if not files or not any(
        Path(distribution.locate_file(f)).resolve() == init for f in files
    ):
        raise RuntimeError(f"No matching installed distribution for {module.__name__}")
    content = hashlib.sha256()
    for path in sorted(package.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        content.update(path.relative_to(package).as_posix().encode() + b"\0")
        content.update(hashlib.sha256(path.read_bytes()).digest())
    return {
        "kind": "installed",
        "version": distribution.version,
        "content_sha256": content.hexdigest(),
        "sha": None,
        "dirty": None,
    }


def historical_fixture(path=HISTORICAL_FIXTURE):
    """Load the reviewed Git extract; reject any unreviewed change to its pins/source."""
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != HISTORICAL_SHA256:
        raise ValueError(
            "Historical fixture digest mismatch; verify against pinned Git source"
        )
    return json.loads(data)
