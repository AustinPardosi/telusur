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
use fst::{IntoStreamer, Streamer};
use memmap2::Mmap;
use roaring::RoaringBitmap;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::{BTreeMap, HashMap},
    fs::{self, File},
    io::BufWriter,
    path::Path,
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

/// The filter fields of an index, opened.
#[derive(Default)]
pub struct FilterIndex {
    date: Option<String>,
    fields: HashMap<String, Field>,
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
        })
    }

    /// The docs that pass every filter, or `None` when there are no filters.
    pub fn allowed(&self, filters: &Filters) -> Result<Option<RoaringBitmap>> {
        let mut allowed: Option<RoaringBitmap> = None;
        for (name, filter) in filters {
            let Some(field) = self.fields.get(name) else {
                bail!("{name} is not a filter field of this index (declare it when building)");
            };
            let bitmap = |v: u64| {
                RoaringBitmap::deserialize_from(
                    &field.bitmaps[(v >> 32) as usize..][..v as u32 as usize],
                )
            };
            let mut docs = RoaringBitmap::new();
            match filter {
                Filter::Any(values) => {
                    for v in values {
                        if let Some(at) = field.values.get(v) {
                            docs |= bitmap(at)?;
                        }
                    }
                }
                Filter::Range { gte, lte } => {
                    ensure!(
                        self.date.as_ref() == Some(name),
                        "{name} is not the crawl date field, so it takes a list of values"
                    );
                    let mut range = field.values.range();
                    if let Some(from) = gte {
                        range = range
                            .ge(utc_day(from).with_context(|| format!("{name} gte {from:?}"))?);
                    }
                    if let Some(to) = lte {
                        range =
                            range.le(utc_day(to).with_context(|| format!("{name} lte {to:?}"))?);
                    }
                    let mut stream = range.into_stream();
                    while let Some((_, at)) = stream.next() {
                        docs |= bitmap(at)?;
                    }
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
