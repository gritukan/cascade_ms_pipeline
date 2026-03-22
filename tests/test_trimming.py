from __future__ import annotations

from pathlib import Path

from cascade_ms_pipeline.trimming import trim_mzml_remove_ms2


MZML_TEXT = """<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<mzML xmlns=\"http://psi.hupo.org/ms/mzml\">
  <cvList count=\"1\"></cvList>
  <run id=\"run1\">
    <spectrumList count=\"3\">
      <spectrum id=\"scan=1\" index=\"0\"><cvParam accession=\"MS:1000511\" name=\"ms level\" value=\"1\"/></spectrum>
      <spectrum id=\"scan=2\" index=\"1\"><cvParam accession=\"MS:1000511\" name=\"ms level\" value=\"2\"/></spectrum>
      <spectrum id=\"scan=3\" index=\"2\"><cvParam accession=\"MS:1000511\" name=\"ms level\" value=\"2\"/></spectrum>
    </spectrumList>
  </run>
</mzML>
"""


def test_trim_mzml_removes_requested_ms2(tmp_path: Path) -> None:
    inp = tmp_path / "input.mzML"
    out = tmp_path / "trimmed.mzML"
    inp.write_text(MZML_TEXT, encoding="utf-8")

    stats = trim_mzml_remove_ms2(inp, out, scans_to_remove={2}, native_ids_to_remove=set())
    text = out.read_text(encoding="utf-8")

    assert stats.total_spectra == 3
    assert stats.total_ms2 == 2
    assert stats.removed_ms2 == 1
    assert 'id="scan=2"' not in text
    assert 'id="scan=3"' in text
    assert 'count="2"' in text
