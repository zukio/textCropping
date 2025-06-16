import os


def is_subpath(child: str, parent: str) -> bool:
    """Return True if 'child' is the same as or inside 'parent'."""
    child_abs = os.path.abspath(child)
    parent_abs = os.path.abspath(parent)
    return os.path.commonpath([child_abs, parent_abs]) == parent_abs
