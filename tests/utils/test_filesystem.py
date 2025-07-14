import pathlib

import pyine.utils.filesystem as filesystem


def test_get_relative_path_to_root(monkeypatch) -> None:
    """Test that get_relative_path_to_root returns the correct relative path."""
    mock_project_root = pathlib.Path("/fake/project/root")
    mock_module_file = mock_project_root / "some" / "module" / "file.py"
    result = filesystem.get_relative_path_to_root(module_file=mock_module_file, project_root=mock_project_root)
    assert result == "some/module/file.py"


def test_get_path_size_file(tmp_path) -> None:
    """Test that get_path_size correctly calculates file size."""
    test_file = tmp_path / "test_file.txt"
    test_content = b"test content with some data" * 10
    test_file.write_bytes(test_content)
    size = filesystem.get_path_size(test_file)
    assert size == len(test_content)


def test_get_path_size_directory(tmp_path) -> None:
    """Test that get_path_size correctly calculates directory size."""
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    file1 = tmp_path / "file1.txt"
    file2 = subdir / "file2.txt"
    file3 = subdir / "file3.txt"
    content1 = b"content1" * 10
    content2 = b"content2" * 20
    content3 = b"content3" * 30
    file1.write_bytes(content1)
    file2.write_bytes(content2)
    file3.write_bytes(content3)
    expected_size = len(content1) + len(content2) + len(content3)
    size = filesystem.get_path_size(tmp_path)
    assert size == expected_size


def test_get_human_readable_size() -> None:
    """Test that get_human_readable_size formats byte sizes correctly."""
    assert filesystem.get_human_readable_size(0) == "0.0B"
    assert filesystem.get_human_readable_size(1023) == "1023.0B"
    assert filesystem.get_human_readable_size(1024) == "1.0KiB"
    assert filesystem.get_human_readable_size(1536) == "1.5KiB"
    assert filesystem.get_human_readable_size(1048576) == "1.0MiB"  # 1024^2
    assert filesystem.get_human_readable_size(1073741824) == "1.0GiB"  # 1024^3
    assert filesystem.get_human_readable_size(1099511627776) == "1.0TiB"  # 1024^4
    assert filesystem.get_human_readable_size(1024, suffix="bytes") == "1.0Kibytes"


def test_check_output_path_overwrite_exists_confirmed(monkeypatch, tmp_path) -> None:
    """Test that check_output_path_overwrite removes existing path when confirmed."""
    test_dir = tmp_path / "test_dir"
    test_dir.mkdir()
    test_file = test_dir / "test_file.txt"
    test_file.write_text("test content")
    monkeypatch.setattr("builtins.input", lambda _: "y")

    exit_called = False

    def mock_exit(code):
        nonlocal exit_called
        exit_called = True
        assert code == 0

    monkeypatch.setattr("builtins.exit", mock_exit)
    filesystem.check_output_path_overwrite(test_dir)
    assert not exit_called
    assert not test_file.exists()
    assert not test_dir.exists()


def test_check_output_path_overwrite_exists_declined(monkeypatch, tmp_path) -> None:
    """Test that check_output_path_overwrite exits when overwrite is declined."""
    test_dir = tmp_path / "test_dir"
    test_dir.mkdir()
    test_file = test_dir / "test_file.txt"
    test_file.write_text("test content")
    monkeypatch.setattr("builtins.input", lambda _: "n")

    exit_called = False

    def mock_exit(code):
        nonlocal exit_called
        exit_called = True
        assert code == 0

    monkeypatch.setattr("builtins.exit", mock_exit)
    filesystem.check_output_path_overwrite(test_dir)
    assert exit_called
    assert test_dir.exists()
    assert test_file.exists()


def test_check_output_path_overwrite_not_exists(tmp_path) -> None:
    """Test that check_output_path_overwrite does nothing when path doesn't exist."""
    test_dir = tmp_path / "nonexistent_dir"
    assert not test_dir.exists()
    filesystem.check_output_path_overwrite(test_dir)
    assert not test_dir.exists()


def test_slugify() -> None:
    """Test that slugify creates filesystem-safe strings."""
    assert filesystem.slugify("Hello World") == "hello-world"
    assert filesystem.slugify("Hello_World") == "hello-world"
    assert filesystem.slugify("Hello  World") == "hello-world"
    assert filesystem.slugify("Hello--World") == "hello-world"
    assert filesystem.slugify("Hello World!") == "hello-world"
    assert filesystem.slugify("  Hello World  ") == "hello-world"
    assert filesystem.slugify("UPPER_CASE") == "upper-case"
    assert filesystem.slugify("special@#characters") == "specialcharacters"
