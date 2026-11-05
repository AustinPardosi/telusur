//! Matryoshka truncation, 1-bit codes, and dot products.

use half::f16;

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

pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

pub fn dot_f16(a: &[f16], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x.to_f32() * y).sum()
}
