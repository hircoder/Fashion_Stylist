import io
import json
import os
import tarfile

import pytest

from stylist.artifacts import ArtifactError, ensure_index, safe_extract
from stylist.config import Settings
from stylist.index import SearchIndex


def _settings(tmp_path, **env):
    env.setdefault("INDEX_ALLOW_FILE_URL", "1")  # the tests serve the tarball from disk
    return Settings.from_env({"EMBEDDER": "hash", "INDEX_DIR": str(tmp_path / "index"), **env})


def test_ensure_index_is_noop_without_url(tmp_path):
    ensure_index(_settings(tmp_path))
    assert not (tmp_path / "index").exists()


def test_ensure_index_is_noop_when_a_valid_index_is_present(tmp_path, index_tar):
    tar_path, sha = index_tar
    s = _settings(tmp_path, INDEX_URL=tar_path.as_uri(), INDEX_SHA256=sha)
    ensure_index(s)
    s2 = _settings(tmp_path, INDEX_URL="file:///does/not/exist", INDEX_SHA256="x")
    ensure_index(s2)  # valid index already there: the bad url is never touched


def test_ensure_index_reinstalls_over_a_broken_index(tmp_path, index_tar):
    tar_path, sha = index_tar
    (tmp_path / "index").mkdir()
    (tmp_path / "index" / "meta.json").write_text("{}")  # interrupted install
    ensure_index(_settings(tmp_path, INDEX_URL=tar_path.as_uri(), INDEX_SHA256=sha))
    assert SearchIndex.load(tmp_path / "index").n_rows == 40


def test_extraction_cap_counts_bytes_actually_written(tmp_path):
    # member declares size 1 byte but the cap must be enforced on real bytes: build an
    # archive whose members total more than max_bytes
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for i in range(3):
            info = tarfile.TarInfo(f"index/f{i}.npy")
            data = b"x" * 5000
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    tar_path = tmp_path / "big.tar.gz"
    tar_path.write_bytes(buf.getvalue())
    with pytest.raises(ArtifactError, match="bytes"):
        safe_extract(tar_path, tmp_path / "out", max_bytes=12_000)
    assert not (tmp_path / "out").exists()


def test_ensure_index_requires_sha256(tmp_path, index_tar):
    with pytest.raises(ArtifactError, match="INDEX_SHA256"):
        ensure_index(_settings(tmp_path, INDEX_URL=index_tar[0].as_uri()))


def test_ensure_index_downloads_verifies_and_installs(tmp_path, index_tar):
    tar_path, sha = index_tar
    ensure_index(_settings(tmp_path, INDEX_URL=tar_path.as_uri(), INDEX_SHA256=sha))
    idx = SearchIndex.load(tmp_path / "index")
    assert idx.n_rows == 40
    leftovers = [p for p in tmp_path.iterdir() if p.name not in ("index", ".index.lock")]
    assert leftovers == []  # temp files cleaned up (the lock file stays on purpose)


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


def test_index_without_bm25_files_is_not_considered_installed(tmp_path, index_tar):
    tar_path, sha = index_tar
    s = _settings(tmp_path, INDEX_URL=tar_path.as_uri(), INDEX_SHA256=sha)
    ensure_index(s)
    import shutil

    shutil.rmtree(tmp_path / "index" / "bm25")
    meta = json.loads((tmp_path / "index" / "meta.json").read_text())
    meta["checksums"] = {k: v for k, v in meta["checksums"].items() if not k.startswith("bm25/")}
    (tmp_path / "index" / "meta.json").write_text(json.dumps(meta))
    ensure_index(s)  # must notice and reinstall
    assert (tmp_path / "index" / "bm25").is_dir()
