from importlib import import_module
from types import ModuleType


def get_rust_module() -> ModuleType | None:
    """Return native Rust module when available, otherwise None."""
    try:
        return import_module("_multispatialccm_rust")
    except ImportError:
        return None
