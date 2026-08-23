"""Safe filesystem operations without external-service dependencies."""

from __future__ import annotations

from pathlib import Path


def write_text(path: Path, content: str) -> None:
    """Input: a path and UTF-8 content. Output: a text file with an existing parent."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def validate_slug(slug: str) -> str:
    """Input: a user slug. Output: a lowercase slug with letters, numbers, and dashes."""

    if not slug or slug.strip() != slug:
        raise ValueError("Slug must not be empty or have leading or trailing spaces.")
    allowed_characters = set("abcdefghijklmnopqrstuvwxyz0123456789-")
    if (
        set(slug) - allowed_characters
        or slug.startswith("-")
        or slug.endswith("-")
        or "--" in slug
    ):
        raise ValueError("Slug must use lowercase letters, numbers, and single dashes.")
    return slug
