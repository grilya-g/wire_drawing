import os
import pytest
from correct_odb_listing import get_odb_files_not_in_exclude

@pytest.fixture
def setup_files(tmp_path):
    # Create exclude list
    exclude_txt = tmp_path / "exclude.txt"
    exclude_content = [
        "fileA1234\n",
        "fileB5678\n",
        "fileC9999\n"
    ]
    exclude_txt.write_text("".join(exclude_content), encoding="utf-8")

    # Create odb folder and files
    odb_folder = tmp_path / "odb_folder"
    odb_folder.mkdir()
    odb_files = [
        "fileA1234.odb",  # should be excluded
        "fileB5678.odb",  # should be excluded
        "fileD0000.odb",  # should be included
        "fileEabcd.odb",  # should be included
        "fileC9999.odb",  # should be excluded
        "notanodb.txt",   # not an odb file
        "fileF0000.ODB",  # should be included (case-insensitive)
    ]
    for fname in odb_files:
        (odb_folder / fname).write_text("dummy", encoding="utf-8")

    return str(exclude_txt), str(odb_folder)

def test_get_odb_files_not_in_exclude_basic(setup_files):
    exclude_txt, odb_folder = setup_files
    result = get_odb_files_not_in_exclude(exclude_txt, odb_folder)
    # Only files not in exclude list (minus last 4 chars) should be included
    expected = {"fileD0000.odb", "fileEabcd.odb", "fileF0000.ODB"}
    assert set(result) == expected

def test_empty_exclude_list(tmp_path):
    exclude_txt = tmp_path / "exclude.txt"
    exclude_txt.write_text("", encoding="utf-8")
    odb_folder = tmp_path / "odb_folder"
    odb_folder.mkdir()
    odb_files = ["abc1234.odb", "def5678.odb"]
    for fname in odb_files:
        (odb_folder / fname).write_text("dummy", encoding="utf-8")
    result = get_odb_files_not_in_exclude(str(exclude_txt), str(odb_folder))
    assert set(result) == set(odb_files)

def test_no_odb_files(tmp_path):
    exclude_txt = tmp_path / "exclude.txt"
    exclude_txt.write_text("somefile1234\n", encoding="utf-8")
    odb_folder = tmp_path / "odb_folder"
    odb_folder.mkdir()
    # No .odb files
    (odb_folder / "notanodb.txt").write_text("dummy", encoding="utf-8")
    result = get_odb_files_not_in_exclude(str(exclude_txt), str(odb_folder))
    assert result == []

def test_exclude_list_with_extra_whitespace(tmp_path):
    exclude_txt = tmp_path / "exclude.txt"
    exclude_txt.write_text(" fileA1234  \nfileB5678\n", encoding="utf-8")
    odb_folder = tmp_path / "odb_folder"
    odb_folder.mkdir()
    odb_files = ["fileA1234.odb", "fileB5678.odb", "fileC0000.odb"]
    for fname in odb_files:
        (odb_folder / fname).write_text("dummy", encoding="utf-8")
    result = get_odb_files_not_in_exclude(str(exclude_txt), str(odb_folder))
    assert set(result) == {"fileC0000.odb"}
