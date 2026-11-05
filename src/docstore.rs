//! Doc store: external id, title, url and snippet per doc, memory-mapped.
//!
//! `docs.bin` holds each doc's four fields as `varint len, bytes`, in doc id order.
//! `docs.off` holds n + 1 little-endian u64 offsets into it.

use crate::{
    text::codec::{get_varint, put_varint},
    util::mmap,
};
use anyhow::Result;
use memmap2::Mmap;
use std::{
    fs::{self, File},
    io::{BufWriter, Write},
    path::Path,
};

const SNIPPET_CHARS: usize = 200;

/// The first ~200 characters of `text`, cut at a word boundary.
pub fn snippet(text: &str) -> &str {
    let Some((end, _)) = text.char_indices().nth(SNIPPET_CHARS) else {
        return text.trim_end();
    };
    let head = &text[..end];
    if text[end..].starts_with(char::is_whitespace) {
        return head.trim_end();
    }
    match head.rfind(char::is_whitespace) {
        Some(i) => head[..i].trim_end(),
        None => head, // one long word: hard cut
    }
}

pub struct Writer {
    data: BufWriter<File>,
    offsets: Vec<u8>,
    end: u64,
    rec: Vec<u8>,
}

impl Writer {
    pub fn create(dir: &Path) -> Result<Self> {
        Ok(Self {
            data: BufWriter::new(File::create(dir.join("docs.bin"))?),
            offsets: 0u64.to_le_bytes().to_vec(),
            end: 0,
            rec: Vec::new(),
        })
    }

    /// Adds the next doc: external id, title, url, snippet.
    pub fn add(&mut self, fields: [&str; 4]) -> Result<()> {
        self.rec.clear();
        for f in fields {
            put_varint(&mut self.rec, f.len() as u64);
            self.rec.extend_from_slice(f.as_bytes());
        }
        self.data.write_all(&self.rec)?;
        self.end += self.rec.len() as u64;
        self.offsets.extend(self.end.to_le_bytes());
        Ok(())
    }

    pub fn finish(mut self, dir: &Path) -> Result<()> {
        self.data.flush()?;
        fs::write(dir.join("docs.off"), self.offsets)?;
        Ok(())
    }
}

pub struct DocStore {
    data: Mmap,
    offsets: Mmap,
}

impl DocStore {
    pub fn open(dir: &Path) -> Result<Self> {
        Ok(Self {
            data: mmap(&dir.join("docs.bin"))?,
            offsets: mmap(&dir.join("docs.off"))?,
        })
    }

    /// External id, title, url and snippet of `doc`.
    pub fn get(&self, doc: u32) -> Result<[&str; 4]> {
        let off = |i: usize| u64::from_le_bytes(self.offsets[i * 8..i * 8 + 8].try_into().unwrap());
        let mut pos = off(doc as usize) as usize;
        let mut field = || -> Result<&str> {
            let len = get_varint(&self.data, &mut pos) as usize;
            pos += len;
            Ok(std::str::from_utf8(&self.data[pos - len..pos])?)
        };
        Ok([field()?, field()?, field()?, field()?])
    }
}
