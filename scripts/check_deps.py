"""Dependency isolation verifier for Zero Zoo source, tests, and requirements files."""

from __future__ import annotations

import ast
import re
import sys
from collections.abc import Sequence
from pathlib import Path


def collect_python_files(targets: Sequence[str]) -> list[Path]:
    """Collect all Python files under the given file or directory paths.

    Args:
        targets: Sequence of file or directory path strings.

    Returns:
        Sorted list of `.py` `Path` objects.
    """
    files: list[Path] = []
    for target in targets:
        path = Path(target)
        if path.is_file() and path.suffix == ".py":
            files.append(path)
        elif path.is_dir():
            files.extend(sorted(path.rglob("*.py")))
    return files


ALLOWED_IMPORTS: frozenset[str] = frozenset(
    {
        "__future__",
        "ast",
        "collections",
        "importlib",
        "inspect",
        "math",
        "os",
        "pathlib",
        "re",
        "sys",
        "tempfile",
        "typing",
        "unittest",
        "pydantic",
        "cdd",
        "ml_switcheroo_ir",
        "ml_switcheroo_compiler",
        "ml_ecosystem_snapshots",
        "zero_jax",
        "zero_flax",
        "zero_torch",
        "zero_pytorch",
        "zero_keras",
        "zero_tensorflow",
        "zero_mlx",
        "zero_pax",
        "zero_grain",
        "zero_optax",
        "zero_orbax",
        "zero_chex",
        "zero_zoo",
        "scripts",
        "tests",
        "pytest",
    }
)

ALLOWED_REQUIREMENTS: frozenset[str] = frozenset(
    {
        "pydantic",
        "cdd",
        "ml-switcheroo-ir",
        "ml-switcheroo-compiler",
        "ml-ecosystem-snapshots",
        "zero-jax",
        "zero-flax",
        "zero-pytorch",
        "zero-keras",
        "zero-tensorflow",
        "zero-mlx",
        "zero-pax",
        "zero-grain",
        "zero-optax",
        "zero-orbax",
        "zero-chex",
        "pytest",
        "pytest-cov",
        "mypy",
        "ruff",
        "interrogate",
        "build",
        "tox",
        "tox-uv",
    }
)


def check_python_imports(file_path: Path) -> list[str]:
    """Verify that a Python file only imports approved stdlib and ecosystem modules.

    Args:
        file_path: Path to the `.py` file to inspect.

    Returns:
        List of disallowed import error messages.
    """
    source = file_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(file_path))
    errors: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top_pkg = alias.name.split(".")[0]
                if top_pkg not in ALLOWED_IMPORTS:
                    errors.append(
                        f"{file_path}:{node.lineno}: Disallowed import '{top_pkg}'."
                    )
        elif (
            isinstance(node, ast.ImportFrom)
            and node.level == 0
            and node.module is not None
        ):
            top_pkg = node.module.split(".")[0]
            if top_pkg not in ALLOWED_IMPORTS:
                errors.append(
                    f"{file_path}:{node.lineno}: Disallowed import from '{top_pkg}'."
                )

    return errors


def check_requirements_file(req_path: Path) -> list[str]:
    """Verify that a `requirements*.txt` file only lists approved packages.

    Args:
        req_path: Path to the requirements file.

    Returns:
        List of disallowed requirement error messages.
    """
    errors: list[str] = []
    for lineno, raw_line in enumerate(
        req_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        pkg_name = re.split(r"[^A-Za-z0-9_.-]+", line)[0]
        if pkg_name not in ALLOWED_REQUIREMENTS:
            errors.append(f"{req_path}:{lineno}: Disallowed requirement '{pkg_name}'.")
    return errors


def main(argv: Sequence[str] | None = None) -> int:
    """Run the dependency isolation check across Python files and requirements files.

    Args:
        argv: Optional target directory/file paths.

    Returns:
        Exit code `0` if all imports and requirements are compliant, else `1`.
    """
    targets = list(argv) if argv is not None else ["src/zero_zoo", "scripts", "tests"]
    errors: list[str] = []
    for py_file in collect_python_files(targets):
        errors.extend(check_python_imports(py_file))

    if argv is None:
        for req_file in sorted(Path(".").glob("requirements*.txt")):
            errors.extend(check_requirements_file(req_file))

    for err in errors:
        print(err)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or None))
