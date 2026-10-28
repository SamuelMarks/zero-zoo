"""Unit tests for all verification and badge scripts in `scripts/`."""

from __future__ import annotations

import tempfile
from pathlib import Path

from scripts import (
    check_compliance,
    check_deps,
    check_docs,
    lint_typing,
    update_badges,
)

NL = chr(10)


def test_update_badges() -> None:
    """Test `get_color`, `update_readme_badges`, and `update_badges.main`."""
    assert update_badges.get_color(100.0) == "brightgreen"
    assert update_badges.get_color(95.0) == "green"
    assert update_badges.get_color(85.0) == "yellowgreen"
    assert update_badges.get_color(75.0) == "yellow"
    assert update_badges.get_color(65.0) == "orange"
    assert update_badges.get_color(50.0) == "red"

    with tempfile.TemporaryDirectory() as tmpdir:
        missing_readme = Path(tmpdir) / "MISSING.md"
        assert update_badges.update_readme_badges(missing_readme) is False

        readme = Path(tmpdir) / "README.md"
        readme_lines = [
            "![cov](https://img.shields.io/badge/coverage-80%25-yellowgreen)",
            "![docs](https://img.shields.io/badge/docs-90%25-green)",
        ]
        readme.write_text(NL.join(readme_lines), encoding="utf-8")
        assert (
            update_badges.update_readme_badges(readme, test_cov=100.0, doc_cov=100.0)
            is True
        )
        text = readme.read_text(encoding="utf-8")
        assert "coverage-100%25-brightgreen" in text
        assert "docs-100%25-brightgreen" in text

        # Second call when already up-to-date exercises the unchanged branch
        assert (
            update_badges.update_readme_badges(readme, test_cov=100.0, doc_cov=100.0)
            is True
        )
        assert update_badges.main([str(readme)]) == 0
        assert update_badges.main() == 0


def test_check_compliance() -> None:
    """Test `check_compliance` with missing, passing, and failing plan files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        missing_plan = Path(tmpdir) / "NONEXISTENT.md"
        todo_file = Path(tmpdir) / "TODO.md"
        assert check_compliance.main([str(missing_plan)]) == 0

        valid_plan = Path(tmpdir) / "SEMANTIC_PLAN.md"
        valid_plan.write_text("Requires `get_zero_pytorch_mlp`.", encoding="utf-8")
        todo_file.write_text("- [x] get_zero_pytorch_mlp", encoding="utf-8")
        assert check_compliance.main([str(valid_plan), str(todo_file)]) == 0

        invalid_plan = Path(tmpdir) / "INVALID_PLAN.md"
        invalid_plan.write_text("Requires `nonexistent_api_symbol`.", encoding="utf-8")
        assert check_compliance.main([str(invalid_plan)]) == 1
        assert check_compliance.main() == 0


def test_lint_typing() -> None:
    """Test `lint_typing` on both clean and violation-containing files."""
    assert lint_typing.main(["src/zero_zoo", "scripts", "tests"]) == 0
    assert lint_typing.main() == 0

    with tempfile.TemporaryDirectory() as tmpdir:
        bad_file = Path(tmpdir) / "bad.py"
        bad_lines = [
            '"""Doc."""',
            "#" + " type: ignore",
            "#" + " pragma: no cover",
            "#" + " noqa",
            "def f(a, *args, **kwargs):",
            "    x: " + "Any = " + "object()",
            "    y = a." + "__dict__",
            "    return " + "getattr(x, 'b')",
        ]
        bad_file.write_text(NL.join(bad_lines), encoding="utf-8")
        assert lint_typing.main([str(bad_file), str(Path(tmpdir) / "nonexistent")]) == 1


def test_check_docs() -> None:
    """Test `check_docs` on clean files and files with missing docstrings/sections."""
    assert check_docs.main(["src/zero_zoo", "scripts", "tests"]) == 0
    assert check_docs.main() == 0

    with tempfile.TemporaryDirectory() as tmpdir:
        zoo_dir = Path(tmpdir) / "zero_zoo"
        zoo_dir.mkdir()
        bad_doc_file = zoo_dir / "bad_docs.py"
        bad_doc_lines = [
            "class UndocClass:",
            "    def __init__(self, x: int) -> None:",
            '        """Init without args section."""',
            "        self.x = x",
            "",
            "def undoc_fn(a: int) -> int:",
            "    return a",
            "",
            "def bad_sections(a: int, *args: int, **kwargs: int) -> int:",
            '    """Missing Args, Returns, and Raises."""',
            "    def inner() -> None:",
            '        """Inner."""',
            "        raise RuntimeError('inner')",
            "    raise ValueError('bad')",
        ]
        bad_doc_file.write_text(NL.join(bad_doc_lines), encoding="utf-8")
        assert (
            check_docs.main([str(bad_doc_file), str(Path(tmpdir) / "nonexistent")]) == 1
        )


def test_check_deps() -> None:
    """Test `check_deps` on clean files and disallowed imports/requirements."""
    assert check_deps.main(["src/zero_zoo", "scripts", "tests"]) == 0
    assert check_deps.main() == 0

    with tempfile.TemporaryDirectory() as tmpdir:
        bad_py = Path(tmpdir) / "bad_import.py"
        bad_py_lines = [
            '"""Doc."""',
            "import disallowed_pkg",
            "from another_bad import item",
            "from . import rel",
        ]
        bad_py.write_text(NL.join(bad_py_lines), encoding="utf-8")
        assert check_deps.main([str(bad_py), str(Path(tmpdir) / "nonexistent")]) == 1

        bad_req = Path(tmpdir) / "requirements-bad.txt"
        bad_req_lines = [
            "# comment",
            "",
            "disallowed-package>=1.0",
            "pydantic>=2.0.0",
        ]
        bad_req.write_text(NL.join(bad_req_lines), encoding="utf-8")
        req_errs = check_deps.check_requirements_file(bad_req)
        assert len(req_errs) == 1
