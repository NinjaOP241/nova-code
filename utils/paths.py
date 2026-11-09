from pathlib import Path


def resolve_path(base: str | Path, path: str | Path) -> Path:
    # Convert the supplied path into a Path object so we can
    # use pathlib methods like is_absolute() and resolve().
    path = Path(path)

    # If the path already contains its complete location,
    # we don't need to combine it with the base directory.
    if path.is_absolute():
        return path.resolve()

    # The path is relative, so it needs a starting directory.
    # Combine it with the base directory to get its full location.
    #
    # Example:
    # base = "/home/sayan/project"
    # path = "src/main.py"
    # Combined = "/home/sayan/project/src/main.py"
    #
    # resolve() cleans up the path (for example, by removing "..")
    # and returns the complete absolute path.
    return (Path(base).resolve() / path).resolve()


def display_path_rel_to_cwd(path: str, cwd: Path | None) -> str:
    try:
        p = Path(path)
    except Exception:
        return path

    if cwd:
        try:
            return str(p.relative_to(cwd))
        except ValueError:
            pass

    return str(p)


def ensure_parent_directory(path: str | Path) -> Path:
    """
    Ensure that the parent directory of the given path exists.

    If the parent directory does not exist, it will be created.
    This is useful when writing files to ensure that the necessary
    directories are in place before attempting to write the file.
    """

    path = Path(path)

    # parent gives us the directory containing the requested file.
    #
    # For example:
    # path = /project/src/main.py
    # path.parent = /project/src
    #
    # parents=True creates any missing parent directories as well.
    # exist_ok=True prevents an error if the directory already exists.
    path.parent.mkdir(parents=True, exist_ok=True)

    return path


def is_binary_file(path: str | Path) -> bool:
    """
    Check whether a file is likely to be binary by looking for null bytes
    in its first 8192 bytes.

    This is a heuristic, not a guaranteed way to identify binary files.
    """

    try:
        # Open the file in binary mode ("rb").
        # This reads raw bytes without trying to decode them as text.
        with open(path, "rb") as f:

            # Read only the first 8192 bytes (8 KB).
            # Checking a small portion is much cheaper than reading
            # an entire potentially large file.
            chunk = f.read(8192)

            # Binary files commonly contain null bytes (\x00).
            # If we find one in the inspected portion, treat the file
            # as binary.
            return b"\x00" in chunk

    except (OSError, IOError):
        # If we cannot open or read the file, we cannot inspect its contents.
        # This implementation treats that situation as "not binary".
        return False
