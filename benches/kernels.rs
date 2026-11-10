//! B1: codes scored per second by each kernel this CPU can run, at 256 and 512 dims.
//! GB/s is codes/s times the code size (32 or 64 bytes). Also f16 rerank rows scored per
//! second, hot in cache and at random rows of a buffer far bigger than the caches.

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use half::f16;
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::hint::black_box;
use telusur::{
    util::hardware,
    vector::{
        kernel::{Kernel, QueryLut, dot_f16, dot_f16_scalar, pack_blocks, scan},
        quant::truncate_normalize,
    },
};

/// One large cluster's worth of codes.
const CODES: usize = 65_536;

fn kernels(c: &mut Criterion) {
    eprintln!("machine: {}", hardware());
    let mut rng = StdRng::seed_from_u64(0);
    for dims in [256, 512] {
        let bytes = dims / 8;
        let q: Vec<f32> = (0..dims).map(|_| rng.random_range(-1.0..1.0)).collect();
        let lut = QueryLut::new(&truncate_normalize(&q, dims));
        let codes: Vec<u8> = (0..CODES * bytes).map(|_| rng.random()).collect();
        let blocks = pack_blocks(&codes, bytes);
        let mut out = vec![0u16; CODES];
        let mut group = c.benchmark_group(format!("scan-{dims}"));
        group.throughput(Throughput::Elements(CODES as u64));
        for k in Kernel::available() {
            group.bench_function(format!("{k:?}").to_lowercase(), |b| {
                b.iter(|| scan(k, &lut, black_box(&blocks), &mut out).unwrap())
            });
        }
        group.finish();
    }
}

/// What the exact filter path does: one `dot_f16` per allowed doc, 10,000 docs, with the
/// kernel this CPU runs and with the scalar reference. "random" picks rows of a 512 MB buffer;
/// "hot" reuses 64 rows that stay in cache.
fn rerank(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(1);
    for dims in [768, 1024] {
        let rows = (512 << 20) / (dims * 2);
        let pattern: Vec<f16> = (0..65_536)
            .map(|_| f16::from_f32(rng.random_range(-1.0..1.0)))
            .collect();
        let base: Vec<f16> = pattern.iter().copied().cycle().take(rows * dims).collect();
        let q: Vec<f32> = (0..dims).map(|_| rng.random_range(-1.0..1.0)).collect();
        let q = truncate_normalize(&q, dims);
        let random: Vec<usize> = (0..10_000).map(|_| rng.random_range(0..rows)).collect();
        let hot: Vec<usize> = random.iter().map(|r| r % 64).collect();
        let mut group = c.benchmark_group(format!("rerank-{dims}"));
        group.throughput(Throughput::Elements(random.len() as u64));
        type Dot = fn(&[f16], &[f32]) -> f32;
        for (kernel, dot) in [("best", dot_f16 as Dot), ("scalar", dot_f16_scalar)] {
            for (rows, picks) in [("hot", &hot), ("random", &random)] {
                group.bench_function(format!("{kernel}-{rows}"), |b| {
                    b.iter(|| {
                        picks
                            .iter()
                            .map(|&r| dot(&base[r * dims..][..dims], black_box(&q)))
                            .sum::<f32>()
                    })
                });
            }
        }
        group.finish();
    }
}

criterion_group!(benches, kernels, rerank);
criterion_main!(benches);
