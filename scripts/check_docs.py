"""AST-based 100% docstring and parameter/return/raises verifier for Zero Zoo."""

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


def _has_non_none_return(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Check whether a function returns a non-`None` value by its annotation.

    Args:
        node: AST function definition node.

    Returns:
        `True` if not `__init__` and return annotation is not `None`.
    """
    if node.name == "__init__" or node.returns is None:
        return False
    return not (isinstance(node.returns, ast.Constant) and node.returns.value is None)


def _has_direct_raise(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Check whether a function body directly contains a `raise` statement.

    Args:
        node: AST function definition node.

    Returns:
        `True` if the function directly raises an exception.
    """
    for child in ast.walk(node):
        if child is not node and isinstance(
            child, (ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            continue
        if isinstance(child, ast.Raise):
            return True
    return False


def _check_function_docstring(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    file_path: Path,
    require_raises: bool,
) -> list[str]:
    """Validate Google-style docstring completeness for a function or method.

    Args:
        node: AST function definition node.
        file_path: Path to the Python file being inspected.
        require_raises: Whether direct `raise` statements require a `Raises:` section.

    Returns:
        List of docstring violation strings.
    """
    errors: list[str] = []
    doc = ast.get_docstring(node)
    if not doc:
        errors.append(
            f"{file_path}:{node.lineno}: Function '{node.name}' missing docstring."
        )
        return errors

    param_names: list[str] = []
    for arg in (
        list(node.args.posonlyargs) + list(node.args.args) + list(node.args.kwonlyargs)
    ):
        if arg.arg not in ("self", "cls"):
            param_names.append(arg.arg)
    if node.args.vararg is not None:
        param_names.append(f"*{node.args.vararg.arg}")
    if node.args.kwarg is not None:
        param_names.append(f"**{node.args.kwarg.arg}")

    if param_names and "Args:" not in doc:
        errors.append(
            f"{file_path}:{node.lineno}: '{node.name}' missing 'Args:' section."
        )
    for param in param_names:
        escaped = re.escape(param)
        if not re.search(rf"(?m)^\s*{escaped}\s*:", doc):
            errors.append(
                f"{file_path}:{node.lineno}: Param '{param}' missing in '{node.name}'."
            )

    if _has_non_none_return(node) and "Returns:" not in doc and "Yields:" not in doc:
        errors.append(
            f"{file_path}:{node.lineno}: '{node.name}' missing 'Returns:' section."
        )

    if require_raises and _has_direct_raise(node) and "Raises:" not in doc:
        errors.append(
            f"{file_path}:{node.lineno}: '{node.name}' missing 'Raises:' section."
        )

    return errors


def check_file_docs(file_path: Path) -> list[str]:
    """Inspect a single Python file for 100% module, class, and function docstrings.

    Args:
        file_path: Path to the `.py` file to inspect.

    Returns:
        List of docstring violation strings.
    """
    source = file_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(file_path))
    errors: list[str] = []

    if not ast.get_docstring(tree):
        errors.append(f"{file_path}:1: Module missing docstring.")

    require_raises = "zero_zoo" in file_path.parts
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            if not ast.get_docstring(node):
                errors.append(
                    f"{file_path}:{node.lineno}: Class '{node.name}' missing docstring."
                )
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            errors.extend(_check_function_docstring(node, file_path, require_raises))

    return errors


def main(argv: Sequence[str] | None = None) -> int:
    """Run the 100% docstring verifier across target paths.

    Args:
        argv: Optional CLI target paths.

    Returns:
        Exit code `0` if 100% compliant, or `1` if violations exist.
    """
    args = list(argv) if argv is not None else ["src/zero_zoo", "scripts", "tests"]
    all_errors: list[str] = []
    for py_file in collect_python_files(args):
        all_errors.extend(check_file_docs(py_file))

    for err in all_errors:
        print(err)
    return 1 if all_errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or None))
