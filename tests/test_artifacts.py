import hashlib
import io
import os
import tarfile

import pytest

from stylist.artifacts import ArtifactError, ensure_index, safe_extract
from stylist.config import Settings
from stylist.index import SearchIndex, build_index


@pytest.fixture(scope="module")
def index_tar(tmp_path_factory, fixture_catalog, hash_embedder):
    root = tmp_path_factory.mktemp("src")
    build_index(fixture_catalog, root / "index", hash_embedder, limit=40, sampling="popular")
    tar_path = root / "index.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(root / "index", arcname="index")
    return tar_path, hashlib.sha256(tar_path.read_bytes()).hexdigest()


def _settings(tmp_path, **env):
    return Settings.from_env({"EMBEDDER": "hash", "INDEX_DIR": str(tmp_path / "index"), **env})


def test_ensure_index_is_noop_without_url(tmp_path):
    ensure_index(_settings(tmp_path))
    assert not (tmp_path / "index").exists()


def test_ensure_index_is_noop_when_index_present(tmp_path, index_tar):
    (tmp_path / "index").mkdir()
    (tmp_path / "index" / "meta.json").write_text("{}")
    ensure_index(_settings(tmp_path, INDEX_URL="file:///does/not/exist", INDEX_SHA256="x"))


def test_ensure_index_requires_sha256(tmp_path, index_tar):
    with pytest.raises(ArtifactError, match="INDEX_SHA256"):
        ensure_index(_settings(tmp_path, INDEX_URL=index_tar[0].as_uri()))


def test_ensure_index_downloads_verifies_and_installs(tmp_path, index_tar):
    tar_path, sha = index_tar
    ensure_index(_settings(tmp_path, INDEX_URL=tar_path.as_uri(), INDEX_SHA256=sha))
    idx = SearchIndex.load(tmp_path / "index")
    assert idx.n_rows == 40
    leftovers = [p for p in tmp_path.iterdir() if p.name != "index"]
    assert leftovers == []  # temp files cleaned up


def test_bad_checksum_installs_nothing(tmp_path, index_tar):
    tar_path, _ = index_tar
    with pytest.raises(ArtifactError, match="sha256"):
        ensure_index(_settings(tmp_path, INDEX_URL=tar_path.as_uri(), INDEX_SHA256="0" * 64))
    assert not (tmp_path / "index").exists()


def test_size_limit_is_enforced(tmp_path, index_tar):
    tar_path, sha = index_tar
    with pytest.raises(ArtifactError, match="bytes"):
        ensure_index(
            _settings(
                tmp_path, INDEX_URL=tar_path.as_uri(), INDEX_SHA256=sha, INDEX_MAX_BYTES="1000"
            )
        )


def _tar_with(member_name, *, symlink=False, data=b"x"):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        info = tarfile.TarInfo(member_name)
        if symlink:
            info.type = tarfile.SYMTYPE
            info.linkname = "/etc/passwd"
            tar.addfile(info)
        else:
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


@pytest.mark.parametrize(
    "member,symlink",
    [("../escape.txt", False), ("/abs/path.txt", False), ("index/link", True)],
)
def test_unsafe_members_are_rejected(tmp_path, member, symlink):
    tar_path = tmp_path / "bad.tar.gz"
    tar_path.write_bytes(_tar_with(member, symlink=symlink))
    with pytest.raises(ArtifactError):
        safe_extract(tar_path, tmp_path / "out", max_bytes=10_000)
    assert not (tmp_path / "out").exists() or not any((tmp_path / "out").iterdir())


def test_second_process_finds_installed_index_under_lock(tmp_path, index_tar, monkeypatch):
    tar_path, sha = index_tar
    s = _settings(tmp_path, INDEX_URL=tar_path.as_uri(), INDEX_SHA256=sha)
    ensure_index(s)
    mtime = os.stat(tmp_path / "index" / "meta.json").st_mtime
    ensure_index(s)  # no second download / reinstall
    assert os.stat(tmp_path / "index" / "meta.json").st_mtime == mtime
