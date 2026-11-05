//! B1: codes scored per second by each kernel this CPU can run, at 256 and 512 dims.
//! GB/s is codes/s times the code size (32 or 64 bytes).

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::hint::black_box;
use telusur::{
    util::hardware,
    vector::{
        kernel::{Kernel, QueryLut, pack_blocks, scan},
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

criterion_group!(benches, kernels);
criterion_main!(benches);
