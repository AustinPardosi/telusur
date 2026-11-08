//! Filters: keyword fields and one crawl date field, each value mapped to a roaring bitmap of
//! the docs that have it. Values for one field are alternatives, different fields must all
//! hold, and a doc missing a filtered field never matches (it is in none of the field's
//! bitmaps). No numeric filters.
//!
//! Files per field f: `filter.f.fst` (value to `offset << 32 | len` in the bitmap file) and
//! `filter.f.bin` (serialized bitmaps). The crawl date field is keyed by `YYYY-MM-DD`, whose
//! byte order is date order, so a date range is a range of keys. `filters.json` names the
//! fields.

use crate::util::mmap;
use anyhow::{Context, Result, bail, ensure};
use fst::Streamer;
use memmap2::Mmap;
use roaring::RoaringBitmap;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::{BTreeMap, HashMap},
    fs::{self, File},
    io::BufWriter,
    path::Path,
    sync::OnceLock,
};

/// What a request asks of one field.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(untagged)]
pub enum Filter {
    /// Any of these values.
    Any(Vec<String>),
    /// Crawl dates in an inclusive range of whole UTC days, `YYYY-MM-DD`; either end may be open.
    Range {
        gte: Option<String>,
        lte: Option<String>,
    },
}

/// Field name to filter. Every entry must hold.
pub type Filters = BTreeMap<String, Filter>;

#[derive(Default, Serialize, Deserialize)]
struct Fields {
    keyword: Vec<String>,
    date: Option<String>,
}

/// Collects field values doc by doc while an index is built.
pub struct Builder {
    fields: Fields,
    values: BTreeMap<String, BTreeMap<String, RoaringBitmap>>,
}

impl Builder {
    pub fn new(keyword: &[String], date: Option<&str>) -> Result<Self> {
        let names = keyword.iter().map(String::as_str).chain(date);
        for name in names.clone() {
            ensure!(
                !name.is_empty()
                    && name
                        .chars()
                        .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-'),
                "filter field names are letters, digits, _ and -; got {name:?}"
            );
        }
        let mut sorted: Vec<&str> = names.collect();
        sorted.sort_unstable();
        ensure!(
            sorted.windows(2).all(|w| w[0] != w[1]),
            "a filter field is named twice"
        );
        Ok(Self {
            fields: Fields {
                keyword: keyword.to_vec(),
                date: date.map(String::from),
            },
            values: sorted
                .into_iter()
                .map(|f| (f.to_string(), BTreeMap::new()))
                .collect(),
        })
    }

    /// Records the declared fields of `doc` from its corpus line's other fields.
    pub fn add(&mut self, doc: u32, fields: &serde_json::Map<String, Value>) -> Result<()> {
        for name in &self.fields.keyword {
            let values: Vec<&str> = match fields.get(name) {
                None | Some(Value::Null) => continue,
                Some(Value::String(s)) => vec![s],
                Some(Value::Array(a)) => a
                    .iter()
                    .map(|v| {
                        v.as_str()
                            .with_context(|| format!("{name} values must be strings"))
                    })
                    .collect::<Result<_>>()?,
                Some(v) => bail!(
                    "{name} must be a string or a list of strings (no numeric filters), got {v}"
                ),
            };
            for v in values {
                self.values
                    .get_mut(name)
                    .unwrap()
                    .entry(v.to_string())
                    .or_default()
                    .insert(doc);
            }
        }
        if let Some(name) = &self.fields.date {
            match fields.get(name) {
                None | Some(Value::Null) => {}
                Some(Value::String(s)) => {
                    let day = utc_day(s).with_context(|| format!("{name} {s:?}"))?;
                    self.values
                        .get_mut(name)
                        .unwrap()
                        .entry(day.to_string())
                        .or_default()
                        .insert(doc);
                }
                Some(v) => bail!("{name} must be a date string, got {v}"),
            }
        }
        Ok(())
    }

    pub fn finish(self, dir: &Path) -> Result<()> {
        for (field, values) in &self.values {
            let mut fst = fst::MapBuilder::new(BufWriter::new(File::create(
                dir.join(format!("filter.{field}.fst")),
            )?))?;
            let mut bitmaps = Vec::new();
            for (value, docs) in values {
                let offset = bitmaps.len() as u64;
                docs.serialize_into(&mut bitmaps)?;
                let len = bitmaps.len() as u64 - offset;
                ensure!(
                    offset < 1 << 32 && len < 1 << 32,
                    "{field} bitmaps outgrew 4 GB"
                );
                fst.insert(value, offset << 32 | len)?;
            }
            fst.finish()?;
            fs::write(dir.join(format!("filter.{field}.bin")), bitmaps)?;
        }
        fs::write(dir.join("filters.json"), serde_json::to_vec(&self.fields)?)?;
        Ok(())
    }
}

/// The UTC day of a crawl date: `YYYY-MM-DD`, or a UTC timestamp like `2013-05-18T07:12:47Z`.
pub fn utc_day(s: &str) -> Result<&str> {
    let (day, rest) = s.split_at_checked(10).unwrap_or((s, ""));
    let ok = day.len() == 10
        && day.char_indices().all(|(i, c)| {
            if i == 4 || i == 7 {
                c == '-'
            } else {
                c.is_ascii_digit()
            }
        })
        && (1..=12).contains(&day[5..7].parse::<u32>().unwrap_or(0))
        && (1..=31).contains(&day[8..10].parse::<u32>().unwrap_or(0));
    ensure!(ok, "not a YYYY-MM-DD date");
    ensure!(
        rest.is_empty() || (rest.starts_with('T') && rest.ends_with('Z')),
        "timestamps must be UTC (end in Z)"
    );
    Ok(day)
}

struct Field {
    values: fst::Map<Mmap>,
    bitmaps: Mmap,
}

impl Field {
    /// The bitmap stored at `at` (`offset << 32 | len`). Unchecked: index files are written
    /// once by `Builder` and never change (docs/adr/0001-indexes-are-immutable.md), and roaring
    /// stays memory safe on bad input, so skipping its validation only skips work.
    fn bitmap(&self, at: u64) -> Result<RoaringBitmap> {
        let bytes = &self.bitmaps[(at >> 32) as usize..][..at as u32 as usize];
        Ok(RoaringBitmap::deserialize_unchecked_from(bytes)?)
    }
}

/// The crawl days of the date field in order, with a segment tree of their bitmaps so a date
/// range is the union of a few precomputed blocks of days, not of every day in it (a wide
/// range of FineWeb took milliseconds that way). Node i is the union of nodes 2i and 2i + 1;
/// the leaves, from `days.len()` on, are single days.
/// ponytail: built in memory on the first range query and kept, about (log2 days + 1) copies
/// of the dated docs; store it in the index files if that ever matters.
struct DayTree {
    days: Vec<String>,
    nodes: Vec<RoaringBitmap>,
}

impl DayTree {
    fn build(field: &Field) -> Result<Self> {
        let (mut days, mut leaves) = (Vec::new(), Vec::new());
        let mut stream = field.values.stream();
        while let Some((day, at)) = stream.next() {
            days.push(String::from_utf8(day.to_vec())?);
            leaves.push(field.bitmap(at)?);
        }
        let mut nodes = vec![RoaringBitmap::new(); days.len()];
        nodes.append(&mut leaves);
        for i in (1..days.len()).rev() {
            nodes[i] = &nodes[2 * i] | &nodes[2 * i + 1];
        }
        Ok(Self { days, nodes })
    }

    /// The docs crawled on days `gte..=lte`, either end open (`YYYY-MM-DD`, compared as text,
    /// which is date order).
    fn range(&self, gte: Option<&str>, lte: Option<&str>) -> RoaringBitmap {
        let n = self.days.len();
        let mut l = n + gte.map_or(0, |g| self.days.partition_point(|d| d.as_str() < g));
        let mut r = n + lte.map_or(n, |t| self.days.partition_point(|d| d.as_str() <= t));
        let mut docs = RoaringBitmap::new();
        while l < r {
            if l % 2 == 1 {
                docs |= &self.nodes[l];
                l += 1;
            }
            if r % 2 == 1 {
                r -= 1;
                docs |= &self.nodes[r];
            }
            (l, r) = (l / 2, r / 2);
        }
        docs
    }
}

/// The filter fields of an index, opened.
#[derive(Default)]
pub struct FilterIndex {
    date: Option<String>,
    fields: HashMap<String, Field>,
    days: OnceLock<DayTree>,
}

impl FilterIndex {
    pub fn open(dir: &Path) -> Result<Self> {
        let Ok(json) = fs::read(dir.join("filters.json")) else {
            return Ok(Self::default());
        };
        let names: Fields = serde_json::from_slice(&json)?;
        let mut fields = HashMap::new();
        for name in names.keyword.iter().chain(&names.date) {
            let field = Field {
                values: fst::Map::new(mmap(&dir.join(format!("filter.{name}.fst")))?)?,
                bitmaps: mmap(&dir.join(format!("filter.{name}.bin")))?,
            };
            fields.insert(name.clone(), field);
        }
        Ok(Self {
            date: names.date,
            fields,
            days: OnceLock::new(),
        })
    }

    /// The crawl date field, if the index has one.
    pub fn date_field(&self) -> Option<&str> {
        self.date.as_deref()
    }

    /// Every value of `field` with its number of docs, in value order (date order for the
    /// crawl date field).
    pub fn value_counts(&self, field: &str) -> Result<Vec<(String, u64)>> {
        let Some(f) = self.fields.get(field) else {
            bail!("{field} is not a filter field of this index");
        };
        let mut out = Vec::with_capacity(f.values.len());
        let mut stream = f.values.stream();
        while let Some((value, at)) = stream.next() {
            let docs = f.bitmap(at)?.len();
            out.push((String::from_utf8_lossy(value).into_owned(), docs));
        }
        Ok(out)
    }

    /// The docs that pass every filter, or `None` when there are no filters.
    pub fn allowed(&self, filters: &Filters) -> Result<Option<RoaringBitmap>> {
        let mut allowed: Option<RoaringBitmap> = None;
        for (name, filter) in filters {
            let Some(field) = self.fields.get(name) else {
                bail!("{name} is not a filter field of this index (declare it when building)");
            };
            let mut docs = RoaringBitmap::new();
            match filter {
                Filter::Any(values) => {
                    for v in values {
                        if let Some(at) = field.values.get(v) {
                            docs |= field.bitmap(at)?;
                        }
                    }
                }
                Filter::Range { gte, lte } => {
                    ensure!(
                        self.date.as_ref() == Some(name),
                        "{name} is not the crawl date field, so it takes a list of values"
                    );
                    let from = (gte.as_deref())
                        .map(|d| utc_day(d).with_context(|| format!("{name} gte {d:?}")))
                        .transpose()?;
                    let to = (lte.as_deref())
                        .map(|d| utc_day(d).with_context(|| format!("{name} lte {d:?}")))
                        .transpose()?;
                    let tree = match self.days.get() {
                        Some(tree) => tree,
                        None => {
                            let tree = DayTree::build(field)?;
                            self.days.get_or_init(|| tree)
                        }
                    };
                    docs = tree.range(from, to);
                }
            }
            allowed = Some(match allowed {
                None => docs,
                Some(a) => a & docs,
            });
        }
        Ok(allowed)
    }
}
