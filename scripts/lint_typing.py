"""AST-based static typing and reflection linter for Zero Zoo."""

from __future__ import annotations

import ast
import sys
from collections.abc import Sequence
from pathlib import Path

BANNED_CALLS: frozenset[str] = frozenset(
    {
        "getattr",
        "setattr",
        "hasattr",
        "delattr",
        "vars",
        "globals",
        "locals",
        "eval",
        "exec",
        "cast",
    }
)

BANNED_NAMES: frozenset[str] = frozenset({"Any", "object"})

BANNED_DUNDERS: frozenset[str] = frozenset(
    {
        "__dict__",
        "__setattr__",
        "__getattribute__",
        "__new__",
        "__class__",
    }
)

TYPE_IGNORE_MARKER: str = "#" + " type: ignore"
PRAGMA_NO_COVER_MARKER: str = "#" + " pragma: no cover"
NOQA_MARKER: str = "#" + " noqa"


def _check_comments(source: str, file_path: Path) -> list[str]:
    """Check source lines for banned suppression comments.

    Args:
        source: Raw Python file text.
        file_path: Path to the Python file being inspected.

    Returns:
        List of human-readable violation strings.
    """
    errors: list[str] = []
    for lineno, line in enumerate(source.splitlines(), start=1):
        if TYPE_IGNORE_MARKER in line:
            errors.append(
                f"{file_path}:{lineno}: Banned '{TYPE_IGNORE_MARKER}' comment."
            )
        if PRAGMA_NO_COVER_MARKER in line:
            errors.append(
                f"{file_path}:{lineno}: Banned '{PRAGMA_NO_COVER_MARKER}' comment."
            )
        if NOQA_MARKER in line:
            errors.append(f"{file_path}:{lineno}: Banned '{NOQA_MARKER}' comment.")
    return errors


def _check_function_annotations(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    file_path: Path,
) -> list[str]:
    """Verify that a function or method has complete parameter and return annotations.

    Args:
        node: AST function definition node.
        file_path: Path to the Python file being inspected.

    Returns:
        List of human-readable violation strings.
    """
    errors: list[str] = []
    if node.returns is None:
        errors.append(
            f"{file_path}:{node.lineno}: '{node.name}' missing return annotation."
        )

    all_args = (
        list(node.args.posonlyargs) + list(node.args.args) + list(node.args.kwonlyargs)
    )
    for arg in all_args:
        if arg.arg not in ("self", "cls") and arg.annotation is None:
            errors.append(
                f"{file_path}:{node.lineno}: Arg '{arg.arg}' in '{node.name}' untyped."
            )
    if node.args.vararg is not None and node.args.vararg.annotation is None:
        errors.append(
            f"{file_path}:{node.lineno}: '*args' in '{node.name}' missing annotation."
        )
    if node.args.kwarg is not None and node.args.kwarg.annotation is None:
        errors.append(
            f"{file_path}:{node.lineno}: '**kwargs' in '{node.name}' untyped."
        )
    return errors


def check_file(file_path: Path) -> list[str]:
    """Scan a single Python file for typing, reflection, and comment violations.

    Args:
        file_path: Path to the `.py` file to inspect.

    Returns:
        List of violation messages found in `file_path`.
    """
    source = file_path.read_text(encoding="utf-8")
    errors = _check_comments(source, file_path)
    tree = ast.parse(source, filename=str(file_path))

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            errors.extend(_check_function_annotations(node, file_path))
        elif isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            errors.append(
                f"{file_path}:{node.lineno}: Banned type/name '{node.id}' used."
            )
        elif isinstance(node, ast.Attribute):
            if node.attr in BANNED_NAMES or node.attr in BANNED_DUNDERS:
                errors.append(
                    f"{file_path}:{node.lineno}: Banned attribute '{node.attr}' used."
                )
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in BANNED_CALLS
        ):
            errors.append(
                f"{file_path}:{node.lineno}: Banned call '{node.func.id}' used."
            )

    return errors


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


def main(argv: Sequence[str] | None = None) -> int:
    """Run the static typing and reflection linter across target paths.

    Args:
        argv: Optional CLI target paths.

    Returns:
        Exit code `0` if no violations are found, or `1` if violations exist.
    """
    args = list(argv) if argv is not None else ["src/zero_zoo", "scripts", "tests"]
    all_errors: list[str] = []
    for py_file in collect_python_files(args):
        all_errors.extend(check_file(py_file))

    for err in all_errors:
        print(err)
    return 1 if all_errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or None))
