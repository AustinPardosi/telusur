use rand::{Rng, SeedableRng, rngs::StdRng};
use telusur::vector::{
    kernel::{Kernel, QueryLut, pack_blocks, scan, scan_scalar},
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

/// Every kernel this CPU can run gives exactly the scalar kernel's scores.
#[test]
fn every_kernel_equals_scalar() {
    let kernels = Kernel::available();
    eprintln!("kernels on this CPU: {kernels:?}");
    // With TELUSUR_KERNEL set (CI forces avx512 under Intel SDE), that kernel must be here.
    assert!(kernels.contains(&Kernel::default_for_process().unwrap()));
    let mut rng = StdRng::seed_from_u64(2);
    for dims in [64, 128, 256, 512, 1024] {
        let bytes = dims / 8;
        for n in [1, 31, 32, 33, 100, 1000] {
            let q: Vec<f32> = (0..dims).map(|_| rng.random_range(-1.0..1.0)).collect();
            let lut = QueryLut::new(&truncate_normalize(&q, dims));
            let codes: Vec<u8> = (0..n * bytes).map(|_| rng.random()).collect();
            let blocks = pack_blocks(&codes, bytes);
            let mut want = vec![0u16; n.div_ceil(32) * 32];
            scan_scalar(&lut, &blocks, &mut want);
            for &k in &kernels {
                let mut got = vec![u16::MAX; want.len()];
                scan(k, &lut, &blocks, &mut got).unwrap();
                assert_eq!(got, want, "{k:?} dims {dims} n {n}");
            }
        }
    }
}

/// The largest possible sum, 256 groups x 255 at 1024 dims, still fits in a u16.
#[test]
fn every_kernel_handles_the_overflow_edge() {
    let lut = QueryLut::new(&truncate_normalize(&[1.0; 1024], 1024));
    assert!(lut.tables.iter().all(|t| t[15] == 255));
    let blocks = pack_blocks(&[0xFF; 1024 / 8 * 32], 1024 / 8);
    for k in Kernel::available() {
        let mut out = vec![0u16; 32];
        scan(k, &lut, &blocks, &mut out).unwrap();
        assert!(out.iter().all(|&s| s == 256 * 255), "{k:?}: {out:?}");
    }
}
