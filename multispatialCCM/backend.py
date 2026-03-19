_BACKEND = "auto"
_VALID = {"auto", "python", "rust"}


def set_backend(backend: str) -> None:
    if backend not in _VALID:
        raise ValueError("backend must be one of: auto, python, rust")
    global _BACKEND
    _BACKEND = backend


def get_backend() -> str:
    return _BACKEND
