use anyhow::{Context, Result};
use memmap2::Mmap;
use std::{fs, fs::File, path::Path, process::Command};

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
