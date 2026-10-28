"""Updates README badges based on test and documentation coverage."""

from __future__ import annotations

import re
import sys
from collections.abc import Sequence
from pathlib import Path


def get_color(pct: float) -> str:
    """Return shields.io color name for a given coverage percentage.

    Args:
        pct: Coverage percentage in `[0.0, 100.0]`.

    Returns:
        Shields.io badge color string.
    """
    if pct >= 100.0:
        return "brightgreen"
    if pct >= 90.0:
        return "green"
    if pct >= 80.0:
        return "yellowgreen"
    if pct >= 70.0:
        return "yellow"
    if pct >= 60.0:
        return "orange"
    return "red"


def update_readme_badges(
    readme_path: Path = Path("README.md"),
    test_cov: float = 100.0,
    doc_cov: float = 100.0,
) -> bool:
    """Update test and doc coverage badges in `README.md` if present.

    Args:
        readme_path: Path to the `README.md` file.
        test_cov: Test coverage percentage to display.
        doc_cov: Documentation coverage percentage to display.

    Returns:
        `True` if `readme_path` exists and was processed; `False` otherwise.
    """
    if not readme_path.exists():
        return False

    content = readme_path.read_text(encoding="utf-8")
    test_color = get_color(test_cov)
    doc_color = get_color(doc_cov)
    updated = re.sub(
        r"coverage-\d+(?:\.\d+)?%25-[a-z]+",
        f"coverage-{int(test_cov)}%25-{test_color}",
        content,
    )
    updated = re.sub(
        r"docs-\d+(?:\.\d+)?%25-[a-z]+",
        f"docs-{int(doc_cov)}%25-{doc_color}",
        updated,
    )
    if updated != content:
        readme_path.write_text(updated, encoding="utf-8")
    return True


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for updating README coverage badges.

    Args:
        argv: Optional CLI arguments (`[readme_path]`).

    Returns:
        Exit code `0`.
    """
    args = list(argv) if argv is not None else []
    readme_path = Path(args[0]) if len(args) >= 1 else Path("README.md")
    update_readme_badges(readme_path=readme_path)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or None))
