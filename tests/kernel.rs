use rand::{Rng, SeedableRng, rngs::StdRng};
use telusur::vector::{
    kernel::{QueryLut, pack_blocks, scan_scalar},
    quant::truncate_normalize,
};

/// The scalar kernel, which works on transposed blocks, equals summing one table entry per
/// group straight off each code.
#[test]
fn scalar_kernel_equals_naive_lut_sum() {
    let mut rng = StdRng::seed_from_u64(1);
    for dims in [64, 128, 256, 512, 1024] {
        let bytes = dims / 8;
        let n = 100; // 3 full blocks and a ragged one of 4
        let q: Vec<f32> = (0..dims).map(|_| rng.random_range(-1.0..1.0)).collect();
        let lut = QueryLut::new(&truncate_normalize(&q, dims));
        let codes: Vec<u8> = (0..n * bytes).map(|_| rng.random()).collect();

        let blocks = pack_blocks(&codes, bytes);
        let mut scores = vec![0u16; n.div_ceil(32) * 32];
        scan_scalar(&lut, &blocks, &mut scores);

        for (i, code) in codes.chunks_exact(bytes).enumerate() {
            let naive: u32 = (0..dims / 4)
                .map(|g| u32::from(lut.tables[g][usize::from(code[g / 2] >> (4 * (g % 2)) & 15)]))
                .sum();
            assert_eq!(u32::from(scores[i]), naive, "dims {dims} code {i}");
        }
        assert!(
            scores[n..].iter().all(|&s| s == scores[n]),
            "padding slots score as zero codes"
        );
    }
}
