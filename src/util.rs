use anyhow::{Context, Result, ensure};
use bytemuck::Pod;
use memmap2::Mmap;
use std::{
    cmp::Ordering,
    collections::BinaryHeap,
    fs,
    fs::File,
    io::{BufWriter, Write},
    marker::PhantomData,
    path::Path,
    process::Command,
};

/// A memory-mapped file read as a slice of `T`.
pub struct MmapSlice<T> {
    map: Mmap,
    _t: PhantomData<T>,
}

impl<T: Pod> MmapSlice<T> {
    pub fn open(path: &Path) -> Result<Self> {
        let map = mmap(path)?;
        ensure!(
            map.len() % size_of::<T>() == 0,
            "{} is not a whole number of items",
            path.display()
        );
        Ok(Self {
            map,
            _t: PhantomData,
        })
    }

    pub fn as_slice(&self) -> &[T] {
        // A mapping starts on a page boundary, so it is aligned for any T.
        bytemuck::cast_slice(&self.map)
    }
}

/// A `.fbin` file (`u32 n, u32 d`, then n x d little-endian f32 rows), memory-mapped so a
/// build streams it instead of loading it.
pub struct Fbin {
    map: Mmap,
    pub n: usize,
    pub d: usize,
}

impl Fbin {
    pub fn open(path: &Path) -> Result<Self> {
        let map = mmap(path)?;
        ensure!(map.len() >= 8, "{} has no .fbin header", path.display());
        let n = u32::from_le_bytes(map[..4].try_into().unwrap()) as usize;
        let d = u32::from_le_bytes(map[4..8].try_into().unwrap()) as usize;
        ensure!(
            map.len() == 8 + n * d * 4,
            "{} should hold {n} x {d} floats",
            path.display()
        );
        Ok(Self { map, n, d })
    }

    pub fn row(&self, i: usize) -> &[f32] {
        // The rows start 8 bytes into a page-aligned mapping, so they are aligned for f32.
        let all: &[f32] = bytemuck::cast_slice(&self.map[8..]);
        &all[i * self.d..(i + 1) * self.d]
    }
}

/// Writes rows of `k` ids as an `.ibin` file (`u32 n, u32 k`, then n x k little-endian u32),
/// the ground truth format.
pub fn write_ibin(path: &Path, k: usize, rows: &[Vec<u32>]) -> Result<()> {
    let mut w = BufWriter::new(File::create(path)?);
    w.write_all(&(rows.len() as u32).to_le_bytes())?;
    w.write_all(&(k as u32).to_le_bytes())?;
    for row in rows {
        ensure!(row.len() == k, "row of {} ids in a k = {k} file", row.len());
        w.write_all(bytemuck::cast_slice(row))?;
    }
    w.flush()?;
    Ok(())
}

/// Reads an `.ibin` file into rows.
pub fn read_ibin(path: &Path) -> Result<Vec<Vec<u32>>> {
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    ensure!(bytes.len() >= 8, "{} has no .ibin header", path.display());
    let n = u32::from_le_bytes(bytes[..4].try_into().unwrap()) as usize;
    let k = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
    ensure!(
        bytes.len() == 8 + n * k * 4,
        "{} should hold {n} x {k} ids",
        path.display()
    );
    let ids: Vec<u32> = bytes[8..]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&b| u32::from_le_bytes(b))
        .collect();
    Ok(ids.chunks(k.max(1)).map(<[u32]>::to_vec).collect())
}

/// Writes rows of `d` floats as a `.fbin` file.
pub fn write_fbin(
    path: &Path,
    d: usize,
    rows: impl ExactSizeIterator<Item = Vec<f32>>,
) -> Result<()> {
    let mut w = BufWriter::new(File::create(path)?);
    w.write_all(&(rows.len() as u32).to_le_bytes())?;
    w.write_all(&(d as u32).to_le_bytes())?;
    for row in rows {
        ensure!(
            row.len() == d,
            "row of {} floats in a {d}-dim file",
            row.len()
        );
        w.write_all(bytemuck::cast_slice(&row))?;
    }
    w.flush()?;
    Ok(())
}

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
    // A laptop on battery may be in Low Power Mode, which slows the CPU, so say which.
    let power = if cfg!(target_os = "macos") {
        let pmset = |args: &[&str]| {
            Command::new("pmset")
                .args(args)
                .output()
                .map(|o| String::from_utf8_lossy(&o.stdout).into_owned())
                .unwrap_or_default()
        };
        let source = match pmset(&["-g", "ps"]).contains("AC Power") {
            true => "AC power",
            false => "battery",
        };
        let low = pmset(&["-g"])
            .lines()
            .any(|l| l.split_whitespace().eq(["lowpowermode", "1"]));
        format!(", {source}{}", if low { ", Low Power Mode" } else { "" })
    } else {
        String::new()
    };
    let cores = std::thread::available_parallelism().map_or(0, |n| n.get());
    format!(
        "{cpu}, {cores} cores, {} GB RAM, {} {}{power}",
        ram_bytes >> 30,
        std::env::consts::OS,
        std::env::consts::ARCH
    )
}
