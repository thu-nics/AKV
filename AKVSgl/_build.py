"""Prepare a verified SGLang source tree and delegate to its native backends."""

from contextlib import contextmanager
import fcntl
import hashlib
import importlib
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import tomllib
import urllib.request

ASSETS = Path(__file__).resolve().parent
ROOT = ASSETS.parent
SOURCE = ASSETS / ".sglang"
CONFIG = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["akv"]
VERSIONS = {"sglang": "0.5.20+akv", "gateway": "0.3.2+akv"}


def _git(directory, *args):
    return subprocess.check_output(
        ["git", "-C", str(directory), *args], text=True
    ).strip()


def _overlays():
    return sorted((ASSETS / "sglang").rglob("*.py"))


def _signature():
    digest = hashlib.sha256()
    for path in (ROOT / "pyproject.toml", ASSETS / "_build.py", ASSETS / "sglang.patch"):
        digest.update(path.read_bytes())
    return digest.hexdigest()


@contextmanager
def _lock(name):
    with (ASSETS / f".sglang.{name}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def _packaging(source):
    # Keep the runtime tree unchanged; these are build metadata adjustments only.
    project = source / "python/pyproject.toml"
    text = project.read_text().replace(
        'dynamic = ["version"]', 'version = "0.5.20+akv"'
    )
    # Static version avoids consulting unrelated AKV Git tags in source archives.
    text = text.replace('  "setuptools-scm>=8.0",\n', '')
    start = text.index("[tool.setuptools_scm]")
    end = text.index("\n# Rust extension", start)
    text = text[:start] + text[end:]
    project.write_text(text)
    # version.py otherwise reads a stale generated _version.py, if one exists.
    (source / "python/sglang/_version.py").write_text(
        '__version__ = "0.5.20+akv"\n__version_tuple__ = (0, 5, 20, "akv")\n'
    )
    project = source / "sgl-model-gateway/bindings/python/pyproject.toml"
    text = project.read_text().replace('version = "0.3.2"', 'version = "0.3.2+akv"')
    text = text.replace('["maturin>=1.0,<2.0"]', '["maturin>=1.0,<2.0", "protoc-wheel-0==30.2"]')
    text = text.replace('[tool.maturin]', '[tool.maturin]\nfeatures = ["vendored-openssl"]')
    project.write_text(text)


def _validate_existing(*, changed_inputs=False):
    if not (SOURCE / ".akv-input").is_file():
        raise RuntimeError(f"Incomplete or unrecognized source tree: {SOURCE}; preserve it before rebuilding.")
    if not changed_inputs and (SOURCE / ".akv-input").read_text() != _signature():
        raise RuntimeError(f"Patch/build inputs changed. Preserve or move {SOURCE}, then reinstall.")
    if _git(SOURCE, "rev-parse", "HEAD^{tree}") != CONFIG["base-tree"]:
        raise RuntimeError(f"Unexpected upstream revision in {SOURCE}.")
    expected = (SOURCE / ".akv-packaged-tree").read_text()
    if _git(SOURCE, "write-tree") != expected:
        raise RuntimeError(f"The generated source index was modified: {SOURCE}.")
    overlays = {"python/" + str(p.relative_to(ASSETS)) for p in _overlays()}
    changed = set(_git(SOURCE, "diff", "--name-only").splitlines()) - overlays
    if changed_inputs:
        # Removed overlay files may still have their previous, now dangling link.
        changed = {
            rel for rel in changed
            if not (rel.startswith("python/sglang/")
                    and (SOURCE / rel).is_symlink()
                    and (SOURCE / rel).resolve() == (ASSETS / rel[7:]).resolve())
        }
    if changed:
        raise RuntimeError(f"Generated upstream files were edited; preserve them before rebuilding: {sorted(changed)}")
    for path in _overlays():
        link = SOURCE / "python" / path.relative_to(ASSETS)
        if changed_inputs and not link.exists() and not _git(
            SOURCE, "ls-files", "--", str(link.relative_to(SOURCE))
        ):
            continue  # A newly added overlay will be installed in the fresh tree.
        if not link.is_symlink() or link.resolve() != path.resolve():
            raise RuntimeError(f"Broken editable source mapping: {link}")


def prepare():
    """Safe under simultaneous SGLang and gateway metadata/build requests."""
    with _lock("prepare"):
        if SOURCE.exists():
            current = (SOURCE / ".akv-input")
            unchanged = current.is_file() and current.read_text() == _signature()
            _validate_existing(changed_inputs=not unchanged)
            if unchanged:
                return SOURCE
        patch = ASSETS / "sglang.patch"
        if hashlib.sha256(patch.read_bytes()).hexdigest() != CONFIG["patch-sha256"]:
            raise RuntimeError("sglang.patch does not match its pinned SHA-256.")
        staging = Path(tempfile.mkdtemp(prefix=".sglang.prepare-", dir=ASSETS))
        try:
            _git(staging, "init", "-q")
            _git(staging, "remote", "add", "origin", CONFIG["upstream"])
            # Codeload avoids cloning history and works with GitHub source ZIPs.
            url = f"https://codeload.github.com/sgl-project/sglang/tar.gz/{CONFIG['base']}"
            prefix = f"sglang-{CONFIG['base']}/"
            with urllib.request.urlopen(url, timeout=60) as response:
                with tarfile.open(fileobj=response, mode="r|gz") as archive:
                    for member in archive:
                        if member.name.rstrip("/") == prefix.rstrip("/"):
                            continue
                        if not member.name.startswith(prefix):
                            raise RuntimeError("Unexpected path in upstream archive")
                        member.name = member.name[len(prefix):]
                        archive.extract(member, staging, filter="data")
            _git(staging, "add", "-f", ".")
            if _git(staging, "write-tree") != CONFIG["base-tree"]:
                raise RuntimeError("Upstream archive does not match the pinned official source tree")
            commit = _git(
                staging, "-c", "user.name=AKVSgl", "-c", "user.email=build@localhost",
                "-c", "commit.gpgsign=false", "commit-tree", CONFIG["base-tree"],
                "-m", f"Official SGLang source archive: {CONFIG['base']}",
            )
            _git(staging, "update-ref", "HEAD", commit)
            _git(staging, "apply", "--index", str(patch))
            for path in _overlays():
                target = staging / "python" / path.relative_to(ASSETS)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
                _git(staging, "add", "--", str(target.relative_to(staging)))
            tree = _git(staging, "write-tree")
            if tree != CONFIG["tree"]:
                raise RuntimeError(f"Patched source tree mismatch: {tree} != {CONFIG['tree']}")
            _packaging(staging)
            _git(staging, "add", "python/pyproject.toml", "sgl-model-gateway/bindings/python/pyproject.toml")
            # _version.py may be ignored by the upstream checkout.
            _git(staging, "add", "-f", "python/sglang/_version.py")
            (staging / ".akv-packaged-tree").write_text(_git(staging, "write-tree"))
            for path in _overlays():
                target = staging / "python" / path.relative_to(ASSETS)
                target.unlink()
                # Relative links survive moving the whole clone to a new path.
                target.symlink_to(os.path.relpath(path, target.parent))
            (staging / ".akv-input").write_text(_signature())
            previous = None
            if SOURCE.exists():
                previous = Path(tempfile.mkdtemp(prefix=".sglang.previous-", dir=ASSETS))
                SOURCE.rename(previous)
            try:
                staging.rename(SOURCE)
            except BaseException:
                if previous is not None:
                    previous.rename(SOURCE)
                raise
        finally:
            if staging.exists():
                shutil.rmtree(staging)
        return SOURCE


@contextmanager
def _native(kind):
    source = prepare()
    path = source / ("python" if kind == "sglang" else "sgl-model-gateway/bindings/python")
    # Two builds of one package must not share mutable native build outputs.
    with _lock(kind):
        cwd = Path.cwd()
        old_protoc = os.environ.get("PROTOC")
        try:
            os.chdir(path)
            protoc = shutil.which("protoc")
            if protoc:
                os.environ["PROTOC"] = protoc
            yield importlib.import_module("setuptools.build_meta" if kind == "sglang" else "maturin")
        finally:
            os.chdir(cwd)
            if old_protoc is None:
                os.environ.pop("PROTOC", None)
            else:
                os.environ["PROTOC"] = old_protoc


def _delegate(kind, hook, *args, **kwargs):
    with _native(kind) as backend:
        method = getattr(backend, hook, None)
        if method is None and hook.startswith("get_requires_for_build_"):
            return []
        if method is None:
            raise AttributeError(hook)
        return method(*args, **kwargs)


def _sdist(kind, directory, config_settings=None):
    """Ship the small patch project, not a copy of the upstream repository."""
    name = "sglang" if kind == "sglang" else "sglang_router"
    prefix = f"{name}-{VERSIONS[kind]}"
    output = Path(directory).resolve() / f"{prefix}.tar.gz"
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp) / prefix
        stage.mkdir()
        entry = ASSETS / "build" / kind
        shutil.copy2(entry / "backend.py", stage / "backend.py")
        project = (entry / "pyproject.toml").read_text()
        # Standalone archives carry the same immutable preparation configuration.
        project += "\n[tool.akv]\n" + "".join(f'{key} = "{value}"\n' for key, value in CONFIG.items())
        (stage / "pyproject.toml").write_text(project)
        assets = stage / "AKVSgl"
        assets.mkdir()
        for name in ("_build.py", "__init__.py", "sglang.patch"):
            shutil.copy2(ASSETS / name, assets / name)
        shutil.copytree(ASSETS / "sglang", assets / "sglang", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        shutil.copytree(ASSETS / "build", assets / "build", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        shutil.copy2(ROOT / "LICENSE", stage / "LICENSE")
        metadata = _delegate(kind, "prepare_metadata_for_build_wheel", tmp, config_settings)
        shutil.copy2(Path(tmp) / metadata / "METADATA", stage / "PKG-INFO")
        with tarfile.open(output, "w:gz", format=tarfile.PAX_FORMAT) as archive:
            archive.add(stage, arcname=prefix)
    return output.name


def hooks(kind):
    """Return only the standard PEP 517/660 hooks for one distribution."""
    from functools import partial

    result = {name: partial(_delegate, kind, name) for name in (
        "get_requires_for_build_wheel", "get_requires_for_build_editable",
        "prepare_metadata_for_build_wheel", "prepare_metadata_for_build_editable",
        "build_wheel", "build_editable",
    )}
    result["get_requires_for_build_sdist"] = lambda config_settings=None: []
    result["build_sdist"] = partial(_sdist, kind)
    return result
