use pyo3::prelude::*;

#[pymodule]
fn _multispatialccm_rust(_py: Python<'_>, _m: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}
