use pyo3::prelude::*;
use pyo3::types::PyDict;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

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

fn get_acceptable_lib_ccm(a: &[f64], e: usize, tau: usize, plengtht: usize) -> (Vec<usize>, Vec<usize>) {
    let gapdist = tau.saturating_mul(e.saturating_sub(1));
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
    let mut acceptablelib = Vec::new();
    for (i, v) in acceptable.iter().enumerate() {
        if *v > 0.0 && i <= plengtht.saturating_sub(1) {
            acceptablelib.push(i);
        }
    }
    let mut acceptablelib2 = Vec::new();
    let lim2 = plengtht.saturating_sub(1).saturating_sub(tau);
    for &i in &acceptablelib {
        if i < lim2 {
            acceptablelib2.push(i);
        }
    }
    (acceptablelib, acceptablelib2)
}

fn get_rho_ccm(a: &[f64], aest: &[f64], acceptablelib: &[usize]) -> f64 {
    if acceptablelib.is_empty() {
        return 0.0;
    }
    let n = acceptablelib.len() as f64;
    let xbar = acceptablelib.iter().map(|&i| a[i]).sum::<f64>() / n;
    let ybar = acceptablelib.iter().map(|&i| aest[i]).sum::<f64>() / n;

    let mut num = 0.0;
    let mut xxbarsq = 0.0;
    let mut yybarsq = 0.0;
    for &i in acceptablelib {
        let dx = a[i] - xbar;
        let dy = aest[i] - ybar;
        num += dx * dy;
        xxbarsq += dx * dx;
        yybarsq += dy * dy;
    }
    let denom = xxbarsq.sqrt() * yybarsq.sqrt();
    if denom == 0.0 {
        return 0.0;
    }
    let r = num / denom;
    if (-1.0..=1.0).contains(&r) { r } else { 0.0 }
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
    let mut a_vec: Vec<f64> = a.extract(py)?;
    let mut b_vec: Vec<f64> = b.extract(py)?;
    let length_a = a_vec.len();

    let plengtht = a_vec.iter().filter(|v| is_finite(**v)).count().min(a_vec.len());
    let (acceptablelib, acceptablelib2) = get_acceptable_lib_ccm(&a_vec, e, tau, plengtht);
    let lengthacceptablelib = acceptablelib.len();
    let from_idx = tau.saturating_mul(e.saturating_sub(1));

    let mut desired_l_vec: Vec<usize> = if let Some(obj) = desired_l {
        let mut raw: Vec<isize> = obj.extract(py)?;
        raw.iter_mut()
            .map(|x| (*x + e as isize - 2).max(0) as usize)
            .collect()
    } else {
        let start = from_idx + e + 1;
        let end_exclusive = length_a.saturating_sub(e).saturating_add(2);
        if start >= end_exclusive {
            Vec::new()
        } else {
            (start..end_exclusive).collect()
        }
    };

    let mut valid_l = Vec::new();
    for dl in desired_l_vec.drain(..) {
        if acceptablelib2.is_empty() {
            continue;
        }
        let mut best = acceptablelib2[0];
        let mut best_diff = best.abs_diff(dl);
        for &cand in &acceptablelib2 {
            let d = cand.abs_diff(dl);
            if d < best_diff {
                best = cand;
                best_diff = d;
            }
        }
        valid_l.push(best);
    }
    valid_l.sort_unstable();
    valid_l.dedup();
    desired_l_vec = valid_l;

    if tau.saturating_mul(e + 1) > lengthacceptablelib {
        let out = PyDict::new_bound(py);
        out.set_item("A", &a_vec)?;
        out.set_item("Aest", vec![f64::NAN; length_a])?;
        out.set_item("B", &b_vec)?;
        out.set_item("rho", f64::NAN)?;
        out.set_item("sdevrho", f64::NAN)?;
        out.set_item("Lobs", f64::NAN)?;
        out.set_item("E", e)?;
        out.set_item("tau", tau)?;
        out.set_item("FULLinfo", f64::NAN)?;
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

    let n_neighbors = e + 1;
    let min_weight = 0.000001_f64;
    let mut rng = StdRng::seed_from_u64(42);

    let mut rho_mat = vec![vec![f64::NAN; iterations]; desired_l_vec.len()];
    let mut aest_sum = vec![0.0f64; length_a];
    let mut lpos: Vec<usize> = Vec::new();

    let lengtht = a_vec.iter().filter(|v| !v.is_nan()).count();

    for it in 0..iterations {
        let mut aest = vec![0.0f64; length_a];

        for (lidx, &l0) in desired_l_vec.iter().enumerate() {
            let mut l = l0;
            let min_l = from_idx + e + 1;
            if l < min_l {
                l = min_l;
            }
            if l >= lengtht {
                l = lengtht.saturating_sub(1);
            }
            if l < from_idx {
                continue;
            }
            let to = l;
            let lib_count = to - from_idx + 1;
            let mut lib_indices = vec![0usize; lib_count];
            for item in &mut lib_indices {
                let pick = rng.gen_range(0..acceptablelib.len());
                *item = acceptablelib[pick];
            }

            for &i in &acceptablelib {
                if i < from_idx || i >= b_vec.len() {
                    continue;
                }
                let mut point_lagged = vec![0.0f64; e];
                for k in 0..e {
                    point_lagged[k] = b_vec[i - tau * k];
                }

                let mut dist_and_idx: Vec<(f64, usize)> = Vec::with_capacity(lib_count);
                for &lib_i in &lib_indices {
                    let mut dist2 = 0.0;
                    for k in 0..e {
                        let d = b_vec[lib_i - tau * k] - point_lagged[k];
                        dist2 += d * d;
                    }
                    let mut d = dist2.sqrt();
                    if lib_i == i {
                        d = f64::INFINITY;
                    }
                    dist_and_idx.push((d, lib_i));
                }
                dist_and_idx.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(std::cmp::Ordering::Equal));
                let neigh = &dist_and_idx[..n_neighbors.min(dist_and_idx.len())];
                if neigh.is_empty() {
                    continue;
                }
                let distsv = neigh[0].0;
                let mut weights = vec![0.0f64; neigh.len()];
                if distsv != 0.0 && distsv.is_finite() {
                    let mut sumu = 0.0;
                    for (j, (d, _)) in neigh.iter().enumerate() {
                        let u = (-(*d) / distsv).exp();
                        weights[j] = u;
                        sumu += u;
                    }
                    let mut sumw = 0.0;
                    for w in &mut weights {
                        *w = (*w / sumu).max(min_weight);
                        sumw += *w;
                    }
                    for w in &mut weights {
                        *w /= sumw;
                    }
                } else {
                    let mut sumw = 0.0;
                    for (j, (d, _)) in neigh.iter().enumerate() {
                        weights[j] = if *d == 0.0 { 1.0 } else { min_weight };
                        sumw += weights[j];
                    }
                    for w in &mut weights {
                        *w /= sumw;
                    }
                }
                let mut pred = 0.0;
                for (j, (_, ni)) in neigh.iter().enumerate() {
                    pred += a_vec[*ni] * weights[j];
                }
                aest[i] = pred;
            }
            rho_mat[lidx][it] = get_rho_ccm(&a_vec, &aest, &acceptablelib);
            if it == 0 {
                lpos.push(l.saturating_sub(e).saturating_add(1));
            }
        }

        for i in 0..length_a {
            aest_sum[i] += aest[i];
        }
    }

    lpos.sort_unstable();
    lpos.dedup();

    let mut rho_means = vec![f64::NAN; desired_l_vec.len()];
    let mut rho_sdev = vec![f64::NAN; desired_l_vec.len()];
    for r in 0..desired_l_vec.len() {
        let row = &rho_mat[r];
        let vals: Vec<f64> = row.iter().copied().filter(|v| !v.is_nan()).collect();
        if vals.is_empty() {
            continue;
        }
        let mean = vals.iter().sum::<f64>() / vals.len() as f64;
        let var = vals.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / vals.len() as f64;
        rho_means[r] = mean;
        rho_sdev[r] = var.sqrt();
    }

    let mut aest_avg = vec![0.0f64; length_a];
    for i in 0..length_a {
        aest_avg[i] = aest_sum[i] / iterations as f64;
        if aest_avg[i] == 0.0 {
            aest_avg[i] = f64::NAN;
        }
    }

    let out = PyDict::new_bound(py);
    out.set_item("A", &a_vec)?;
    out.set_item("Aest", &aest_avg)?;
    out.set_item("B", &b_vec)?;
    out.set_item("rho", &rho_means)?;
    out.set_item("sdevrho", &rho_sdev)?;
    out.set_item("Lobs", &lpos)?;
    out.set_item("E", e)?;
    out.set_item("tau", tau)?;
    out.set_item("FULLinfo", &rho_mat)?;
    Ok(out.into_py(py))
}

#[pymodule]
fn _multispatialccm_rust(_py: Python<'_>, _m: &Bound<'_, PyModule>) -> PyResult<()> {
    _m.add_function(wrap_pyfunction!(ssr_pred_boot, _m)?)?;
    _m.add_function(wrap_pyfunction!(ccm_boot, _m)?)?;
    Ok(())
}
