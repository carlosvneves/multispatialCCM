from importlib import import_module, util
from types import ModuleType
from pathlib import Path


def get_rust_module() -> ModuleType | None:
    """Return native Rust module when available, otherwise None."""
    try:
        return import_module("_multispatialccm_rust")
    except ImportError:
        pass

    roots = [Path.cwd(), Path(__file__).resolve().parent.parent]
    patterns = [
        "target/maturin/lib_multispatialccm_rust*.dylib",
        "target/maturin/lib_multispatialccm_rust*.so",
        "target/release/lib_multispatialccm_rust*.dylib",
        "target/release/lib_multispatialccm_rust*.so",
        "target/release/_multispatialccm_rust*.so",
        "target/release/_multispatialccm_rust*.dylib",
        "target/release/_multispatialccm_rust*.pyd",
        "target/debug/_multispatialccm_rust*.so",
        "target/debug/_multispatialccm_rust*.dylib",
        "target/debug/_multispatialccm_rust*.pyd",
    ]

    for root in roots:
        for pattern in patterns:
            for candidate in root.glob(pattern):
                spec = util.spec_from_file_location("_multispatialccm_rust", candidate)
                if spec is None or spec.loader is None:
                    continue
                module = util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return module

    return None
