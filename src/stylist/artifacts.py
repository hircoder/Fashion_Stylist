"""Fetch a prebuilt index tarball at startup (for container deployments).

Deploying the service somewhere like Railway means the image cannot carry a 100K-row
index and the container cannot spend 10 minutes embedding on boot. So: build the index
once locally (`make index`), publish `index.tar.gz` somewhere reachable, and set
INDEX_URL + INDEX_SHA256. On boot, if data/index is missing, this module installs it.

Safety rules, since the archive comes from the network:
  * lock first, then re-check (another worker may have installed it already)
  * stream to a temp file next to the destination, enforce a size cap, verify sha256
  * extract only regular files / dirs with safe relative paths (tarfile "data" filter
    plus our own checks), never links or devices
  * verify the index checksums, then one atomic rename into place
"""

from __future__ import annotations

import contextlib
import fcntl
import logging
import os
import shutil
import tarfile
import urllib.request
from pathlib import Path

from stylist.config import Settings
from stylist.index import CHECKSUMMED, IndexMeta, sha256_file

log = logging.getLogger(__name__)


class ArtifactError(RuntimeError):
    pass


@contextlib.contextmanager
def _file_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def _download(url: str, out: Path, max_bytes: int) -> None:
    done = 0
    with urllib.request.urlopen(url, timeout=120) as resp, open(out, "wb") as f:  # noqa: S310
        total = int(resp.headers.get("Content-Length") or 0)
        if total > max_bytes:
            raise ArtifactError(f"index archive is {total} bytes, above INDEX_MAX_BYTES")
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            done += len(chunk)
            if done > max_bytes:
                raise ArtifactError(f"index archive exceeds INDEX_MAX_BYTES ({max_bytes} bytes)")
            f.write(chunk)


def _check_member(member: tarfile.TarInfo) -> None:
    name = member.name
    if name.startswith("/") or name.startswith("\\") or os.path.isabs(name):
        raise ArtifactError(f"archive member with absolute path: {name}")
    parts = Path(name).parts
    if ".." in parts:
        raise ArtifactError(f"archive member escapes the target directory: {name}")
    if not (member.isfile() or member.isdir()):
        raise ArtifactError(f"archive member is not a regular file or directory: {name}")


def safe_extract(tar_path: Path, dest: Path, max_bytes: int) -> None:
    """Extract into `dest` (created fresh). Rejects links, devices, abs paths, `..`, size."""
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    total = 0
    try:
        with tarfile.open(tar_path, "r:*") as tar:
            for member in tar:
                _check_member(member)
                total += max(member.size, 0)
                if total > max_bytes:
                    raise ArtifactError(f"extracted size exceeds INDEX_MAX_BYTES ({max_bytes})")
                tar.extract(member, path=dest, filter="data")
    except (tarfile.TarError, OSError) as exc:
        shutil.rmtree(dest, ignore_errors=True)
        raise ArtifactError(f"cannot extract {tar_path.name}: {exc}") from exc
    except ArtifactError:
        shutil.rmtree(dest, ignore_errors=True)
        raise


def _find_index_root(extracted: Path) -> Path:
    for candidate in [extracted, *sorted(p for p in extracted.iterdir() if p.is_dir())]:
        if (candidate / "meta.json").exists():
            return candidate
    raise ArtifactError("archive does not contain an index directory (no meta.json found)")


def verify_index_files(index_dir: Path) -> None:
    meta = IndexMeta.from_json((index_dir / "meta.json").read_text())
    for name in CHECKSUMMED:
        path = index_dir / name
        if not path.exists():
            raise ArtifactError(f"downloaded index is missing {name}")
        if sha256_file(path) != meta.checksums.get(name):
            raise ArtifactError(f"downloaded index has a bad checksum for {name}")
    if not (index_dir / "bm25").is_dir():
        raise ArtifactError("downloaded index is missing the bm25 directory")


def install_index(url: str, sha256: str, index_dir: Path, max_bytes: int) -> None:
    parent = index_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    lock = parent / f".{index_dir.name}.lock"
    with _file_lock(lock):
        if (index_dir / "meta.json").exists():
            log.info("index appeared while waiting for the lock, nothing to do")
            return
        pid = os.getpid()
        tmp_tar = parent / f".{index_dir.name}.{pid}.download"
        tmp_dir = parent / f".{index_dir.name}.{pid}.extract"
        try:
            log.info("downloading index from %s", url)
            _download(url, tmp_tar, max_bytes)
            got = sha256_file(tmp_tar)
            if got.lower() != sha256.lower():
                raise ArtifactError(f"index archive sha256 mismatch: got {got[:12]}...")
            safe_extract(tmp_tar, tmp_dir, max_bytes)
            root = _find_index_root(tmp_dir)
            verify_index_files(root)
            os.replace(root, index_dir)
            log.info("index installed at %s", index_dir)
        finally:
            tmp_tar.unlink(missing_ok=True)
            shutil.rmtree(tmp_dir, ignore_errors=True)
    lock.unlink(missing_ok=True)


def ensure_index(settings: Settings) -> None:
    """Install the index from INDEX_URL when it is missing locally. No-op otherwise."""
    index_dir = Path(settings.index_dir)
    if (index_dir / "meta.json").exists():
        return
    if not settings.index_url:
        return
    if not settings.index_sha256:
        raise ArtifactError("INDEX_SHA256 must be set together with INDEX_URL")
    install_index(settings.index_url, settings.index_sha256, index_dir, settings.index_max_bytes)
