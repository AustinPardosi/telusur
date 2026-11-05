//! Spherical k-means for the cluster index, and exact top-k for ground truth. Both run on a
//! real matrix multiply over chunks of rows: a dot-product loop per query over a whole base
//! is memory-bound and orders of magnitude slower.

use crate::{
    util::{Fbin, TopK},
    vector::quant::truncate_normalize,
};
use rand::{Rng, rngs::StdRng, seq::index::sample};
use rayon::prelude::*;

/// Rows per matrix multiply.
const CHUNK: usize = 4096;

/// Dot products of every row of `a` with every row of `b` (both `d` wide), as an
/// `a rows x b rows` matrix, row-major.
pub fn dots(a: &[f32], b: &[f32], d: usize) -> Vec<f32> {
    let (m, n) = (a.len() / d, b.len() / d);
    assert!(a.len() == m * d && b.len() == n * d, "rows of {d} floats");
    let mut c = vec![0f32; m * n];
    if m == 0 || n == 0 {
        return c;
    }
    // SAFETY: `a` is read as m x d with strides (d, 1) and `b` as d x n with strides (1, d),
    // that is b's rows as columns; `c` is m x n with strides (n, 1). The assert makes every
    // index those strides reach lie inside its slice, and `c` is a separate allocation.
    unsafe {
        matrixmultiply::sgemm(
            m,
            d,
            n,
            1.0,
            a.as_ptr(),
            d as isize,
            1,
            b.as_ptr(),
            1,
            d as isize,
            0.0,
            c.as_mut_ptr(),
            n as isize,
            1,
        );
    }
    c
}

/// The centroid with the largest dot product for each of `n` rows (`row(i)`, `d` wide),
/// ties to the lower centroid.
pub fn assign(
    n: usize,
    d: usize,
    row: impl Fn(usize) -> Vec<f32> + Sync,
    centroids: &[f32],
) -> Vec<u32> {
    let k = centroids.len() / d;
    let chunks: Vec<Vec<u32>> = (0..n.div_ceil(CHUNK))
        .into_par_iter()
        .map(|c| {
            let rows: Vec<f32> = (c * CHUNK..((c + 1) * CHUNK).min(n))
                .flat_map(&row)
                .collect();
            dots(&rows, centroids, d)
                .chunks_exact(k)
                .map(|s| (0..k).fold(0, |best, j| if s[j] > s[best] { j } else { best }) as u32)
                .collect()
        })
        .collect();
    chunks.concat()
}

/// Spherical k-means on `points` (normalized rows, `d` wide): `k` distinct points to start,
/// then `iters` rounds of assigning and moving each centroid to its points' normalized mean.
/// A cluster left empty restarts from a random point.
pub fn train(points: &[f32], d: usize, k: usize, iters: usize, rng: &mut StdRng) -> Vec<f32> {
    let n = points.len() / d;
    assert!(n >= k, "{n} points for {k} clusters");
    let point = |i: usize| &points[i * d..(i + 1) * d];
    let mut centroids: Vec<f32> = sample(rng, n, k)
        .into_iter()
        .flat_map(|i| point(i).to_vec())
        .collect();
    for _ in 0..iters {
        let assigned = assign(n, d, |i| point(i).to_vec(), &centroids);
        let mut sums = vec![0f32; k * d];
        let mut counts = vec![0usize; k];
        for (i, &c) in assigned.iter().enumerate() {
            let c = c as usize;
            counts[c] += 1;
            sums[c * d..(c + 1) * d]
                .iter_mut()
                .zip(point(i))
                .for_each(|(s, x)| *s += x);
        }
        for (c, centroid) in sums.chunks_exact_mut(d).enumerate() {
            let norm = centroid.iter().map(|x| x * x).sum::<f32>().sqrt();
            if counts[c] == 0 || norm == 0.0 {
                centroid.copy_from_slice(point(rng.random_range(0..n)));
            } else {
                centroid.iter_mut().for_each(|x| *x /= norm);
            }
        }
        centroids = sums;
    }
    centroids
}

/// For each query (rows of `queries`, `d` wide), the `k` base rows with the largest dot
/// product, best first, ties to the lower row. The base has `n` rows, `row(i)`.
pub fn exact_topk(
    n: usize,
    d: usize,
    row: impl Fn(usize) -> Vec<f32> + Sync,
    queries: &[f32],
    k: usize,
) -> Vec<Vec<u32>> {
    let nq = queries.len() / d;
    let fresh = || (0..nq).map(|_| TopK::new(k)).collect::<Vec<_>>();
    let tops = (0..n.div_ceil(CHUNK))
        .into_par_iter()
        .fold(fresh, |mut tops, c| {
            let start = c * CHUNK;
            let rows: Vec<f32> = (start..((c + 1) * CHUNK).min(n)).flat_map(&row).collect();
            let s = dots(queries, &rows, d);
            let width = rows.len() / d;
            for (q, top) in tops.iter_mut().enumerate() {
                for (j, &score) in s[q * width..(q + 1) * width].iter().enumerate() {
                    if score >= top.threshold() {
                        top.push((start + j) as u32, score);
                    }
                }
            }
            tops
        })
        .reduce(fresh, |mut a, b| {
            for (x, y) in a.iter_mut().zip(b) {
                y.into_sorted()
                    .into_iter()
                    .for_each(|(doc, s)| x.push(doc, s));
            }
            a
        });
    tops.into_iter()
        .map(|t| t.into_sorted().into_iter().map(|(doc, _)| doc).collect())
        .collect()
}

/// Exact top `k` base rows for every query by cosine over the first `dims` dims: the full
/// ground truth at the vectors' own dims, the rerank ground truth at the rerank dims.
pub fn ground_truth(base: &Fbin, queries: &Fbin, dims: usize, k: usize) -> Vec<Vec<u32>> {
    let q: Vec<f32> = (0..queries.n)
        .flat_map(|i| truncate_normalize(queries.row(i), dims))
        .collect();
    exact_topk(
        base.n,
        dims,
        |i| truncate_normalize(base.row(i), dims),
        &q,
        k,
    )
}
