from pathlib import Path

from ldc.eml_intake import extract_docs
from ldc.purge import purge_path

SAMPLE_EML = Path(__file__).parent.parent / "samples" / "sample.eml"


def test_infer_eml_purge(tmp_path: Path) -> None:
    work_dir = tmp_path / "work" / "sample-1_at_example.com"

    docs = extract_docs(SAMPLE_EML, work_dir)

    assert len(docs) > 0, "Expected at least one document extracted from sample.eml"
    assert work_dir.exists(), "work_dir should exist before purge"

    purge_path(work_dir)

    assert not work_dir.exists(), "work_dir must not exist after purge"
