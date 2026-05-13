"""Utilities for reading raw text data from disk."""


def read_text(path: str) -> str:
    """Read the full contents of a text file from disk.

    Args:
        path: Path to the text file, relative to the working directory.

    Returns:
        The file contents as a single string.
    """
    with open(path, "r", encoding="utf-8") as f:
        return f.read()