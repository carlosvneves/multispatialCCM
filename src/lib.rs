use pyo3::prelude::*;
use pyo3::types::PyDict;

#[pyfunction]
#[pyo3(signature = (a, b=None, e=2, tau=1, predstep=1, match_sugi=0))]
fn ssr_pred_boot(
    py: Python<'_>,
    a: PyObject,
    b: Option<PyObject>,
    e: usize,
    tau: usize,
    predstep: usize,
    match_sugi: usize,
) -> PyResult<PyObject> {
    let simplex_mod = py.import_bound("multispatialCCM.simplex")?;
    let python_impl = simplex_mod.getattr("_ssr_pred_boot_python")?;

    let kwargs = PyDict::new_bound(py);
    kwargs.set_item("A", a)?;
    kwargs.set_item("B", b)?;
    kwargs.set_item("E", e)?;
    kwargs.set_item("tau", tau)?;
    kwargs.set_item("predstep", predstep)?;
    kwargs.set_item("matchSugi", match_sugi)?;

    let out = python_impl.call((), Some(&kwargs))?;
    Ok(out.into_py(py))
}

#[pymodule]
fn _multispatialccm_rust(_py: Python<'_>, _m: &Bound<'_, PyModule>) -> PyResult<()> {
    _m.add_function(wrap_pyfunction!(ssr_pred_boot, _m)?)?;
    Ok(())
}
