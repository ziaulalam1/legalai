from pathlib import Path
from ldc.purge import purge_path

def test_purge_file(tmp_path: Path):
    f = tmp_path / "x.txt"
    f.write_text("hello")
    purge_path(f)
    assert not f.exists()

def test_purge_dir(tmp_path: Path):
    d = tmp_path / "d"
    d.mkdir()
    (d / "a.txt").write_text("a")
    purge_path(d)
    assert not d.exists()
