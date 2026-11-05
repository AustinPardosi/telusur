use anyhow::{Context, Result};
use memmap2::Mmap;
use std::{cmp::Ordering, collections::BinaryHeap, fs, fs::File, path::Path, process::Command};

/// The k best (doc, score): higher score first, ties to the smaller doc id.
pub struct TopK {
    k: usize,
    heap: BinaryHeap<Entry>,
}

/// Ordered so the heap's root is the worst entry kept.
#[derive(PartialEq)]
struct Entry(f32, u32);

impl Eq for Entry {}

impl Ord for Entry {
    fn cmp(&self, other: &Self) -> Ordering {
        other.0.total_cmp(&self.0).then(self.1.cmp(&other.1))
    }
}

impl PartialOrd for Entry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl TopK {
    pub fn new(k: usize) -> Self {
        Self {
            k,
            heap: BinaryHeap::with_capacity(k + 1),
        }
    }

    /// A score must beat this to get in: minus infinity until k are kept.
    pub fn threshold(&self) -> f32 {
        match self.heap.peek() {
            _ if self.heap.len() < self.k => f32::NEG_INFINITY,
            Some(worst) => worst.0,
            None => f32::INFINITY, // k == 0
        }
    }

    pub fn push(&mut self, doc: u32, score: f32) {
        let e = Entry(score, doc);
        if self.heap.len() < self.k {
            self.heap.push(e);
        } else if self.heap.peek().is_some_and(|worst| e < *worst) {
            self.heap.pop();
            self.heap.push(e);
        }
    }

    /// Best first.
    pub fn into_sorted(self) -> Vec<(u32, f32)> {
        self.heap
            .into_sorted_vec()
            .into_iter()
            .map(|Entry(s, d)| (d, s))
            .collect()
    }
}

/// Nearest-rank percentile of sorted values, `p` in 0..=100.
pub fn percentile(sorted: &[f64], p: f64) -> f64 {
    let rank = (p / 100.0 * sorted.len() as f64).ceil() as usize;
    sorted[rank.clamp(1, sorted.len()) - 1]
}

/// Memory-maps an index file read-only.
pub fn mmap(path: &Path) -> Result<Mmap> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    // SAFETY: index files are written once at build time and never modified afterwards
    // (docs/adr/0001-indexes-are-immutable.md), so the mapped bytes cannot change under us.
    unsafe { Mmap::map(&file) }.with_context(|| format!("mmap {}", path.display()))
}

/// CPU model, cores, RAM and OS, printed with every benchmark.
pub fn hardware() -> String {
    let (cpu, ram_bytes) = if cfg!(target_os = "macos") {
        let out = Command::new("sysctl")
            .args(["-n", "machdep.cpu.brand_string", "hw.memsize"])
            .output()
            .map(|o| String::from_utf8_lossy(&o.stdout).into_owned())
            .unwrap_or_default();
        let mut lines = out.lines();
        let cpu = lines.next().unwrap_or("unknown CPU").to_string();
        (
            cpu,
            lines
                .next()
                .and_then(|l| l.parse::<u64>().ok())
                .unwrap_or(0),
        )
    } else {
        let field = |file: &str, key: &str| {
            fs::read_to_string(file).ok().and_then(|s| {
                s.lines()
                    .find(|l| l.starts_with(key))
                    .and_then(|l| l.split(':').nth(1))
                    .map(|v| v.trim().to_string())
            })
        };
        let cpu = field("/proc/cpuinfo", "model name").unwrap_or_else(|| "unknown CPU".into());
        let kb = field("/proc/meminfo", "MemTotal")
            .and_then(|v| v.trim_end_matches(" kB").parse::<u64>().ok());
        (cpu, kb.unwrap_or(0) * 1024)
    };
    let cores = std::thread::available_parallelism().map_or(0, |n| n.get());
    format!(
        "{cpu}, {cores} cores, {} GB RAM, {} {}",
        ram_bytes >> 30,
        std::env::consts::OS,
        std::env::consts::ARCH
    )
}
