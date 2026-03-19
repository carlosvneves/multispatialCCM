use pyo3::prelude::*;
use pyo3::types::PyDict;

fn is_finite(x: f64) -> bool {
    x.is_finite()
}

fn get_acceptable_lib(a: &[f64], e: usize, tau: usize, predstep: usize) -> Vec<usize> {
    let gapdist = tau.saturating_mul(e.saturating_sub(1)).saturating_add(predstep);
    let mut acceptable = vec![0.0; a.len()];
    for (i, v) in a.iter().enumerate() {
        acceptable[i] = if is_finite(*v) { 1.0 } else { 0.0 };
    }

    for shift in 1..=gapdist {
        for idx in (shift..a.len()).rev() {
            acceptable[idx] *= if is_finite(a[idx - shift]) { 1.0 } else { 0.0 };
        }
        for item in acceptable.iter_mut().take(shift.min(a.len())) {
            *item = 0.0;
        }
    }

    acceptable
        .iter()
        .enumerate()
        .filter_map(|(i, v)| if *v > 0.0 { Some(i) } else { None })
        .collect()
}

fn getorder_ssr(
    distances: &[f64],
    e: usize,
    acceptablelib: &[usize],
    i: usize,
    predstep: usize,
) -> (Vec<usize>, usize) {
    let mut nneigh = 1usize;
    let mut n = 0usize;
    let length = acceptablelib.len();
    if length == 0 {
        return (vec![0; e + 1], 0);
    }

    if acceptablelib[0] == i {
        n = 1;
    }
    if n >= length {
        n = length - 1;
    }

    let mut neighbors = vec![0usize; e + 1];
    neighbors[0] = acceptablelib[n];
    let upper = length.saturating_sub(predstep);

    for iii in n..upper {
        let ii = acceptablelib[iii];
        let mut trip = false;

        for j in 0..nneigh {
            if distances[ii] < distances[neighbors[j]] && ii != i && j > 0 {
                let mut k = nneigh;
                while k > j {
                    if k < e + 1 {
                        neighbors[k] = neighbors[k - 1];
                    }
                    k -= 1;
                }
                neighbors[j] = ii;
                trip = true;
                break;
            }
        }

        if !trip && nneigh < e + 1 && ii != i && neighbors[nneigh - 1] != ii {
            neighbors[nneigh] = ii;
            if nneigh < e + 1 {
                nneigh += 1;
            }
        }
    }

    (neighbors, nneigh)
}

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
    let mut a_vec: Vec<f64> = a.extract(py)?;
    let a_len = a_vec.len();
    let mut b_vec: Vec<f64> = if let Some(b_obj) = b {
        b_obj.extract(py)?
    } else {
        a_vec.clone()
    };
    let b_len = b_vec.len();

    let repvec = if a_vec.len() == b_vec.len() {
        let mut eq_count = 0usize;
        let mut finite_count = 0usize;
        for idx in 0..a_vec.len() {
            if is_finite(a_vec[idx]) {
                finite_count += 1;
                if is_finite(b_vec[idx]) && a_vec[idx] == b_vec[idx] {
                    eq_count += 1;
                }
            }
        }
        (eq_count == finite_count) as usize
    } else {
        0
    };

    let acceptablelib = get_acceptable_lib(&a_vec, e, tau, predstep);
    let lengthacceptablelib = acceptablelib.len();

    if tau.saturating_mul(e + 1).saturating_add(predstep) >= lengthacceptablelib {
        let out = PyDict::new_bound(py);
        let a_nan = vec![f64::NAN; a_vec.len()];
        out.set_item("A", a_vec.clone())?;
        out.set_item("Aest", a_nan)?;
        out.set_item("B", b_vec.clone())?;
        out.set_item("E", e)?;
        out.set_item("tau", tau)?;
        out.set_item("pBlength", b_vec.len())?;
        out.set_item("pAlength", a_vec.len())?;
        out.set_item("predstep", predstep)?;
        out.set_item("rho", f64::NAN)?;
        out.set_item("acceptablelib", acceptablelib.clone())?;
        out.set_item("plengthacceptablelib", lengthacceptablelib)?;
        return Ok(out.into_py(py));
    }

    for v in &mut a_vec {
        if !is_finite(*v) {
            *v = 0.0;
        }
    }
    for v in &mut b_vec {
        if !is_finite(*v) {
            *v = 0.0;
        }
    }

    let nneigh = e + 1;
    let mut aest = vec![0.0f64; a_vec.len()];
    let mut maxdist = 0.0f64;

    for ii in 0..lengthacceptablelib {
        let i = acceptablelib[ii];
        let mut distances = vec![f64::INFINITY; b_vec.len()];

        if repvec == 1 {
            for jj in 0..lengthacceptablelib.saturating_sub(predstep) {
                let j = acceptablelib[jj];
                if match_sugi == 1 {
                    if i != j {
                        let mut dist = 0.0;
                        for k in 0..e {
                            dist += (a_vec[i - tau * k] - b_vec[j - tau * k]).powi(2);
                        }
                        distances[j] = dist.sqrt();
                        if distances[j] > maxdist {
                            maxdist = 999_999_999.0 * distances[j];
                        }
                    } else {
                        distances[j] = maxdist;
                    }
                } else if (j > i + predstep) || (j <= i.saturating_sub(e)) {
                    let mut dist = 0.0;
                    for k in 0..e {
                        dist += (a_vec[i - tau * k] - b_vec[j - tau * k]).powi(2);
                    }
                    distances[j] = dist.sqrt();
                    if distances[j] > maxdist {
                        maxdist = 999_999_999.0 * distances[j];
                    }
                } else {
                    distances[j] = maxdist;
                }
            }
        } else {
            for j in acceptablelib.iter().copied() {
                let mut dist = 0.0;
                for k in 0..e {
                    dist += (a_vec[i - tau * k] - b_vec[j - tau * k]).powi(2);
                }
                distances[j] = dist.sqrt();
            }
        }

        let (neighbors, found_nneigh) = getorder_ssr(&distances, e, &acceptablelib, i, predstep);
        if found_nneigh < nneigh {
            aest[i] = 0.0;
            continue;
        }

        let distsv = distances[neighbors[0]];
        let mut sumaest = 0.0;
        if distsv != 0.0 {
            let mut u = vec![0.0f64; nneigh];
            for j in 0..nneigh {
                u[j] = (-distances[neighbors[j]] / distsv).exp();
            }
            let sumu: f64 = u.iter().sum();
            let mut w = vec![0.0f64; nneigh];
            for j in 0..nneigh {
                w[j] = (u[j] / sumu).max(0.000001);
            }
            let sumw: f64 = w.iter().sum();
            for j in 0..nneigh {
                w[j] /= sumw;
                sumaest += b_vec[neighbors[j] + predstep] * w[j];
            }
        } else {
            let mut w = vec![0.0f64; nneigh];
            let mut sumw = 0.0f64;
            for j in 0..nneigh {
                w[j] = if distances[neighbors[j]] == 0.0 {
                    1.0
                } else {
                    0.000001
                };
                sumw += w[j];
            }
            for j in 0..nneigh {
                w[j] /= sumw;
                sumaest += a_vec[neighbors[j]] * w[j];
            }
        }

        aest[i] = sumaest;
    }

    let mut aest_out = vec![0.0f64; a_vec.len()];
    if predstep < aest.len() {
        aest_out[predstep..aest.len()].copy_from_slice(&aest[..(aest.len() - predstep)]);
    }
    for v in &mut aest_out {
        if *v == 0.0 {
            *v = f64::NAN;
        }
    }

    let mut x = Vec::new();
    let mut y = Vec::new();
    for idx in 0..a_vec.len() {
        if is_finite(a_vec[idx]) && is_finite(aest_out[idx]) {
            x.push(a_vec[idx]);
            y.push(aest_out[idx]);
        }
    }
    let rho = if x.len() > 1 {
        let xbar = x.iter().sum::<f64>() / x.len() as f64;
        let ybar = y.iter().sum::<f64>() / y.len() as f64;
        let mut num = 0.0;
        let mut sx = 0.0;
        let mut sy = 0.0;
        for idx in 0..x.len() {
            let dx = x[idx] - xbar;
            let dy = y[idx] - ybar;
            num += dx * dy;
            sx += dx * dx;
            sy += dy * dy;
        }
        let den = sx.sqrt() * sy.sqrt();
        if den == 0.0 { f64::NAN } else { num / den }
    } else {
        f64::NAN
    };

    let out = PyDict::new_bound(py);
    out.set_item("A", &a_vec)?;
    out.set_item("Aest", &aest_out)?;
    out.set_item("B", &b_vec)?;
    out.set_item("E", e)?;
    out.set_item("tau", tau)?;
    out.set_item("pBlength", b_len)?;
    out.set_item("pAlength", a_len)?;
    out.set_item("predstep", predstep)?;
    out.set_item("rho", rho)?;
    out.set_item("acceptablelib", acceptablelib.clone())?;
    out.set_item("plengthacceptablelib", acceptablelib.len())?;
    Ok(out.into_py(py))
}

#[pyfunction]
#[pyo3(signature = (a, b, e, tau=1, desired_l=None, iterations=100))]
fn ccm_boot(
    py: Python<'_>,
    a: PyObject,
    b: PyObject,
    e: usize,
    tau: usize,
    desired_l: Option<PyObject>,
    iterations: usize,
) -> PyResult<PyObject> {
    let ccm_mod = py.import_bound("multispatialCCM.ccm")?;
    let python_impl = ccm_mod.getattr("_ccm_boot_python")?;

    let kwargs = PyDict::new_bound(py);
    kwargs.set_item("A", a)?;
    kwargs.set_item("B", b)?;
    kwargs.set_item("E", e)?;
    kwargs.set_item("tau", tau)?;
    kwargs.set_item("DesiredL", desired_l)?;
    kwargs.set_item("iterations", iterations)?;

    let out = python_impl.call((), Some(&kwargs))?;
    Ok(out.into_py(py))
}

#[pymodule]
fn _multispatialccm_rust(_py: Python<'_>, _m: &Bound<'_, PyModule>) -> PyResult<()> {
    _m.add_function(wrap_pyfunction!(ssr_pred_boot, _m)?)?;
    _m.add_function(wrap_pyfunction!(ccm_boot, _m)?)?;
    Ok(())
}
