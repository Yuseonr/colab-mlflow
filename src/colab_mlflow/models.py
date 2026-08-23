"""Small data models shared by services."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Project:
    """Result of initializing a user's source project."""

    slug: str
    root: Path
    configuration: Path


@dataclass(frozen=True)
class Experiment:
    """Result of creating an experiment in a source project."""

    slug: str
    root: Path
    manifest: Path
    pipeline: Path
