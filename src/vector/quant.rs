//! Matryoshka truncation, 1-bit codes, and dot products.

/// The first `dims` values of `v`, scaled to unit length. A zero vector stays zero.
pub fn truncate_normalize(v: &[f32], dims: usize) -> Vec<f32> {
    let mut out = v[..dims].to_vec();
    let norm = dot(&out, &out).sqrt();
    if norm > 0.0 {
        out.iter_mut().for_each(|x| *x /= norm);
    }
    out
}

/// One bit per dim, 1 when the value is above 0. Byte j holds dims 8j..8j+7, least
/// significant bit first, so its low nibble is group 2j and its high nibble group 2j + 1.
pub fn binarize(v: &[f32], out: &mut [u8]) {
    for (byte, dims) in out.iter_mut().zip(v.as_chunks::<8>().0) {
        *byte = dims
            .iter()
            .enumerate()
            .fold(0, |b, (i, &x)| b | u8::from(x > 0.0) << i);
    }
}

/// Adds in 16 lanes: a plain `.sum()` must add in order, which keeps the compiler from
/// vectorizing it.
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let ((a16, a_rest), (b16, b_rest)) = (a.as_chunks::<16>(), b.as_chunks::<16>());
    let mut lanes = [0f32; 16];
    for (x, y) in a16.iter().zip(b16) {
        for i in 0..16 {
            lanes[i] += x[i] * y[i];
        }
    }
    lanes.iter().sum::<f32>() + a_rest.iter().zip(b_rest).map(|(x, y)| x * y).sum::<f32>()
}
