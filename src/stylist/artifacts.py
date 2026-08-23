"""Fetch a prebuilt index tarball at startup (for container deployments).

Deploying the service somewhere like Railway means the image cannot carry a 100K-row
index and the container cannot spend 10 minutes embedding on boot. So: build the index
once locally (`make index`), publish `index.tar.gz` somewhere reachable, and set
INDEX_URL + INDEX_SHA256. On boot, if data/index is missing, this module installs it.

Safety rules, since the archive comes from the network:
  * only http(s) urls (file:// needs INDEX_ALLOW_FILE_URL, for tests and local dev);
    redirects are checked with the same rule
  * lock first, then re-check (another worker may have installed it already); the lock
    file is never removed, so two workers can never hold locks on different inodes
  * stream to a temp file next to the destination, enforce a size cap, verify sha256
  * extract only regular files / dirs with safe relative paths, a member count cap, a
    depth cap, no duplicates, only the file types an index contains, fixed permissions
  * verify the index checksums, then one atomic rename into place
"""

from __future__ import annotations

import contextlib
import ipaddress
import logging
import os
import shutil
import socket
import tarfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

try:
    import fcntl
except ImportError:  # pragma: no cover - windows
    fcntl = None  # type: ignore[assignment]

from stylist.catalog import PIPELINE_VERSION
from stylist.config import Settings
from stylist.index import (
    IndexMeta,
    IndexValidationError,
    SearchIndex,
    sha256_file,
    verify_checksums,
)

log = logging.getLogger(__name__)


ALLOWED_SUFFIXES = {".json", ".npy", ".parquet"}  # everything an index directory holds
MAX_MEMBERS = 2000
MAX_DEPTH = 6


class ArtifactError(RuntimeError):
    pass


@contextlib.contextmanager
def _file_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fh:  # append mode: never truncates another holder's file
        if fcntl is not None:
            fcntl.flock(fh, fcntl.LOCK_EX)
        else:
            log.warning("no fcntl on this platform, index install is not locked")
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(fh, fcntl.LOCK_UN)


def _host_is_local(host: str) -> bool:
    """Loopback, private, link-local (cloud metadata lives there) or reserved addresses,
    by literal or by resolving the name."""
    if not host or host.lower() in ("localhost", "localhost.localdomain"):
        return True
    try:
        addrs = {str(info[4][0]) for info in socket.getaddrinfo(host, None)}
    except socket.gaierror:
        return False  # unresolvable: the download fails on its own
    for addr in addrs:
        try:
            ip = ipaddress.ip_address(addr.split("%")[0])
        except ValueError:
            continue
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            return True
        if ip.is_unspecified or ip.is_multicast:
            return True
    return False


def check_url(url: str, allow_file: bool, allow_private: bool = False) -> None:
    """Only http(s) may fetch an index, and not from a private / loopback / link-local
    host (INDEX_ALLOW_PRIVATE_URL=1 for an internal mirror); file:// needs INDEX_ALLOW_FILE_URL."""
    parts = urllib.parse.urlsplit(url)
    scheme = parts.scheme.lower()
    if scheme in ("http", "https"):
        if not allow_private and _host_is_local(parts.hostname or ""):
            raise ArtifactError(
                f"INDEX_URL host {parts.hostname!r} is a loopback / private / link-local "
                f"address; set INDEX_ALLOW_PRIVATE_URL=1 for an internal mirror"
            )
        return
    if scheme == "file":
        if allow_file:
            return
        raise ArtifactError("INDEX_URL is a file:// url, set INDEX_ALLOW_FILE_URL=1 to allow it")
    raise ArtifactError(f"INDEX_URL has an unsupported scheme {scheme!r} (use https)")


class _RedirectGuard(urllib.request.HTTPRedirectHandler):
    """Redirect targets get the same scheme and host checks as the original url."""

    def __init__(self, allow_file: bool, allow_private: bool = False):
        super().__init__()
        self.allow_file = allow_file
        self.allow_private = allow_private

    def check(self, newurl: str) -> None:
        scheme = urllib.parse.urlsplit(newurl).scheme.lower()
        if scheme not in ("http", "https"):  # a redirect may never land on a local file
            raise ArtifactError(f"redirect to an unsupported scheme {scheme!r} refused")
        check_url(newurl, allow_file=False, allow_private=self.allow_private)

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        self.check(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _download(
    url: str, out: Path, max_bytes: int, allow_file: bool = False, allow_private: bool = False
) -> None:
    check_url(url, allow_file, allow_private)
    opener = urllib.request.build_opener(_RedirectGuard(allow_file, allow_private))
    done = 0
    try:
        resp = opener.open(url, timeout=120)  # noqa: S310 - scheme checked above
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise ArtifactError(f"cannot download the index archive: {exc}") from exc
    with resp, open(out, "wb") as f:
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
    if ".." in parts or any(p.startswith("~") for p in parts):
        raise ArtifactError(f"archive member escapes the target directory: {name}")
    if len(parts) > MAX_DEPTH:
        raise ArtifactError(f"archive member nested too deep ({len(parts)} levels): {name}")
    if not (member.isfile() or member.isdir()):
        raise ArtifactError(f"archive member is not a regular file or directory: {name}")
    if member.isfile() and Path(name).suffix.lower() not in ALLOWED_SUFFIXES:
        raise ArtifactError(f"archive member has an unexpected file type: {name}")


def safe_extract(
    tar_path: Path, dest: Path, max_bytes: int, max_members: int = MAX_MEMBERS
) -> None:
    """Extract into `dest` (created fresh). Only regular files and directories with safe
    relative paths; the size cap counts bytes actually written, not declared sizes;
    permissions come from us (0644 / 0755), never from the archive."""
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    written = 0
    seen: set[str] = set()
    try:
        with tarfile.open(tar_path, "r:*") as tar:
            for count, member in enumerate(tar, 1):
                if count > max_members:
                    raise ArtifactError(f"archive has more than {max_members} members")
                _check_member(member)
                norm = os.path.normpath(member.name)
                if norm in seen:
                    raise ArtifactError(f"archive lists {member.name} twice")
                seen.add(norm)
                target = (dest / member.name).resolve()
                if dest.resolve() not in target.parents and target != dest.resolve():
                    raise ArtifactError(
                        f"archive member escapes the target directory: {member.name}"
                    )
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    os.chmod(target, 0o755)  # nosec B103 - a directory, ours, no world write
                    continue
                src = tar.extractfile(member)
                if src is None:
                    raise ArtifactError(f"cannot read archive member {member.name}")
                target.parent.mkdir(parents=True, exist_ok=True)
                with src, open(target, "wb") as out:
                    while True:
                        chunk = src.read(1 << 20)
                        if not chunk:
                            break
                        written += len(chunk)
                        if written > max_bytes:
                            raise ArtifactError(
                                f"extracted size exceeds INDEX_MAX_BYTES ({max_bytes} bytes)"
                            )
                        out.write(chunk)
                os.chmod(target, 0o644)
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


def verify_index_files(index_dir: Path, expected_model: str | None = None) -> None:
    """Cheap checks: meta parses, checksums match, and the loader would accept the
    pipeline version and model (so a stale local index counts as missing)."""
    try:
        meta = IndexMeta.from_json((index_dir / "meta.json").read_text())
        if meta.pipeline_version != PIPELINE_VERSION:
            raise IndexValidationError(
                f"pipeline version {meta.pipeline_version} != code {PIPELINE_VERSION}"
            )
        if expected_model and meta.embedding_model != expected_model:
            raise IndexValidationError(
                f"embedding model {meta.embedding_model!r} != configured {expected_model!r}"
            )
        verify_checksums(index_dir, meta)
    except (OSError, ValueError, TypeError, KeyError, IndexValidationError) as exc:
        raise ArtifactError(f"index at {index_dir} is not valid: {exc}") from exc


def verify_index_loads(index_dir: Path, expected_model: str | None = None) -> None:
    """The full loader, run once on a freshly extracted archive: whatever the service
    would refuse at startup is refused at install time instead."""
    try:
        SearchIndex.load(index_dir, expected_model=expected_model)
    except IndexValidationError as exc:
        raise ArtifactError(f"downloaded index fails validation: {exc}") from exc


def index_is_valid(index_dir: Path, expected_model: str | None = None) -> bool:
    if not (index_dir / "meta.json").exists():
        return False
    try:
        verify_index_files(index_dir, expected_model)
    except ArtifactError as exc:
        log.warning("%s", exc)
        return False
    return True


def install_index(
    url: str,
    sha256: str,
    index_dir: Path,
    max_bytes: int,
    allow_file: bool = False,
    allow_private: bool = False,
    expected_model: str | None = None,
) -> None:
    check_url(url, allow_file, allow_private)
    parent = index_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    lock = parent / f".{index_dir.name}.lock"
    with _file_lock(lock):
        if index_is_valid(index_dir, expected_model):
            log.info("a valid index appeared while waiting for the lock, nothing to do")
            return
        if index_dir.exists():
            broken = parent / f".{index_dir.name}.broken"
            shutil.rmtree(broken, ignore_errors=True)
            os.replace(index_dir, broken)
            log.warning("moved an incomplete index aside to %s", broken)
            shutil.rmtree(broken, ignore_errors=True)
        pid = os.getpid()
        tmp_tar = parent / f".{index_dir.name}.{pid}.download"
        tmp_dir = parent / f".{index_dir.name}.{pid}.extract"
        try:
            log.info("downloading index from %s", url)
            _download(url, tmp_tar, max_bytes, allow_file, allow_private)
            got = sha256_file(tmp_tar)
            if got.lower() != sha256.lower():
                raise ArtifactError(f"index archive sha256 mismatch: got {got[:12]}...")
            safe_extract(tmp_tar, tmp_dir, max_bytes)
            root = _find_index_root(tmp_dir)
            verify_index_files(root, expected_model)
            verify_index_loads(root, expected_model)
            os.replace(root, index_dir)
            log.info("index installed at %s", index_dir)
        finally:
            tmp_tar.unlink(missing_ok=True)
            shutil.rmtree(tmp_dir, ignore_errors=True)
    # the lock file stays: removing it would let a late worker lock a fresh inode


def ensure_index(settings: Settings) -> None:
    """Install the index from INDEX_URL when it is missing locally. No-op otherwise."""
    index_dir = Path(settings.index_dir)
    if not settings.index_url:
        return  # nothing to fetch from; the loader reports a missing/broken index itself
    if index_is_valid(index_dir, settings.embedding_name):
        return
    if not settings.index_sha256:
        raise ArtifactError("INDEX_SHA256 must be set together with INDEX_URL")
    install_index(
        settings.index_url,
        settings.index_sha256,
        index_dir,
        settings.index_max_bytes,
        allow_file=settings.index_allow_file_url,
        allow_private=settings.index_allow_private_url,
        expected_model=settings.embedding_name,
    )
