# Installation and Packaging

This project is packaged as a mixed Python + Rust distribution using `maturin`.

## Build Artifacts Locally

From repository root:

```bash
uv run --with maturin maturin build --release
```

Artifacts are written to `dist/`:

- wheel (`.whl`) for your platform/Python ABI
- source tarball (`.tar.gz`)

## Install Into Another Project

In your main project environment:

```bash
pip install /absolute/path/to/multispatialCCM/dist/<wheel-file>.whl
```

or with `uv`:

```bash
uv pip install /absolute/path/to/multispatialCCM/dist/<wheel-file>.whl
```

## Editable Install (for development)

```bash
uv run --with maturin maturin develop --release
```

## Verify Installation

```python
from multispatialCCM import set_backend, get_backend

set_backend("rust")
print(get_backend())
```

If the native module is available, `backend="rust"` uses Rust kernels for `SSR_pred_boot` and `CCM_boot`.

## Compatibility Notes

- Runtime requires Python >= 3.10.
- Wheel is platform-specific; build on each target platform or use CI wheel builds.
- For reproducibility in scientific workflows, pin package version and keep input data fixed.
