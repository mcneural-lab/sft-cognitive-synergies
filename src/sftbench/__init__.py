import os
import pathlib


def find_project_root(marker: str = ".git") -> pathlib.Path | None:
    """
    Finds the project root directory by searching upwards for a marker file/directory.
    Falls back to pyproject.toml so public release bundles work without .git.

    Args:
        marker (str): The name of the file or directory to look for
                      to identify the project root. Defaults to '.git'.

    Returns:
        pathlib.Path or None: The path to the project root, or None if the
                              marker is not found in any parent directory.
    """
    try:
        # Try getting the path from __file__ if it's available
        start_dir = pathlib.Path(__file__).resolve().parent
    except NameError:
        # If __file__ is not defined (e.g., in REPL), use current working directory
        start_dir = pathlib.Path(os.getcwd()).resolve()

    # Search upwards from the starting directory.
    for parent in [start_dir] + list(start_dir.parents):
        if (parent / marker).exists() or (parent / "pyproject.toml").exists():
            return parent

    return pathlib.Path.cwd()
