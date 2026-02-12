from pathlib import Path

from inspect_ai._eval.loader import parse_spec_str


def test_parse_spec_str_does_not_treat_directories_as_task_files(
    tmp_path: Path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "comet").mkdir()

    file, name = parse_spec_str("comet")

    assert file is None
    assert name == "comet"
