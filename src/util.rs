use anyhow::{Context, Result};
use memmap2::Mmap;
use std::{fs::File, path::Path};

/// Memory-maps an index file read-only.
pub fn mmap(path: &Path) -> Result<Mmap> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    // SAFETY: index files are written once at build time and never modified afterwards
    // (docs/adr/0001-indexes-are-immutable.md), so the mapped bytes cannot change under us.
    unsafe { Mmap::map(&file) }.with_context(|| format!("mmap {}", path.display()))
}
