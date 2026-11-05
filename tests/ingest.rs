use arrow_array::{ArrayRef, RecordBatch, StringArray};
use parquet::arrow::ArrowWriter;
use serde_json::{Value, json};
use std::{fs, path::PathBuf, sync::Arc};
use telusur::{
    engine::{self, BuildOptions, Index, Mode, Request},
    filter::{Filter, Filters},
    ingest::{self, MAX_CHARS, domain},
};

#[test]
fn domains() {
    for (url, want) in [
        ("https://www.Example.com/a/b", Some("example.com")),
        ("http://example.com:8080/x", Some("example.com")),
        (
            "http://user:pa:ss@WWW.Sub.Example.org/x",
            Some("sub.example.org"),
        ),
        (
            "https://news.example.co.uk?q=a/b",
            Some("news.example.co.uk"),
        ),
        ("https://example.com#frag/x", Some("example.com")),
        ("https://www.www.example.com/", Some("www.example.com")),
        ("https://wwwexample.com/", Some("wwwexample.com")),
        ("http://[2001:DB8::1]:80/x", Some("[2001:db8::1]")),
        ("HTTP://ÉCOLE.example/", Some("école.example")),
        ("example.net/path", Some("example.net")),
        ("https:///nohost", None),
        ("", None),
    ] {
        assert_eq!(domain(url).as_deref(), want, "{url}");
    }
}

fn strings(v: &[Option<&str>]) -> ArrayRef {
    Arc::new(StringArray::from(v.to_vec()))
}

#[test]
fn fineweb_fixture() {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("ingest");
    fs::create_dir_all(&dir).unwrap();
    let long_ascii = "x ".repeat(60_000); // 119,999 chars once collapsed
    let long_wide = "é".repeat(150_000); // 2 bytes per char
    let rows: [(Option<&str>, &str, &str, Option<&str>); 7] = [
        (
            Some("  Hello \n\t world  "),
            "a",
            "https://www.Example.com:8080/p?q=1#f",
            Some("2013-05-18T07:24:47Z"),
        ),
        (Some("Hello world"), "b", "https://other.com/", None), // duplicate once normalized
        (Some("rust simd"), "c", "http://u:p@Sub.Example.org/x", None),
        (None, "d", "https://null.example/", None),
        (Some(" \n "), "e", "https://blank.example/", None),
        (
            Some(&long_ascii),
            "f",
            "https://long.example/",
            Some("2014-01-02T00:00:00Z"),
        ),
        (Some(&long_wide), "g", "", None),
    ];
    let batch = RecordBatch::try_from_iter([
        ("text", strings(&rows.map(|r| r.0))),
        ("id", strings(&rows.map(|r| Some(r.1)))),
        ("dump", strings(&rows.map(|_| Some("CC-MAIN-2013-20")))),
        ("url", strings(&rows.map(|r| Some(r.2)))),
        ("date", strings(&rows.map(|r| r.3))),
    ])
    .unwrap();
    let parquet = dir.join("shard.parquet");
    let mut w =
        ArrowWriter::try_new(fs::File::create(&parquet).unwrap(), batch.schema(), None).unwrap();
    w.write(&batch).unwrap();
    w.close().unwrap();

    let out = dir.join("corpus.jsonl");
    let report = ingest::fineweb(&parquet, &out, None).unwrap();
    let docs: Vec<Value> = fs::read_to_string(&out)
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(
        (
            report.docs_read,
            report.docs_written,
            report.duplicates,
            report.empty,
            report.cut
        ),
        (7, 4, 1, 2, 2)
    );
    let ids: Vec<&str> = docs.iter().map(|d| d["id"].as_str().unwrap()).collect();
    assert_eq!(ids, ["a", "c", "f", "g"], "first occurrence kept");
    assert_eq!(
        docs[0],
        json!({
            "id": "a", "title": "", "text": "Hello world",
            "url": "https://www.Example.com:8080/p?q=1#f",
            "domain": "example.com", "crawl_date": "2013-05-18T07:24:47Z",
        })
    );
    assert_eq!(docs[1]["domain"], "sub.example.org");
    assert!(docs[1].get("crawl_date").is_none());
    assert!(docs[3].get("domain").is_none(), "no url, no domain");
    let chars = |d: &Value| d["text"].as_str().unwrap().chars().count();
    // Cut at 100,000 chars; a trailing space at the cut is trimmed.
    assert_eq!(chars(&docs[2]), MAX_CHARS - 1);
    assert_eq!(chars(&docs[3]), MAX_CHARS, "cut by chars, not bytes");
    assert!(report.docs_per_sec.total > 0.0 && report.docs_per_day > 0.0);

    let limited = ingest::fineweb(&parquet, &dir.join("two.jsonl"), Some(2)).unwrap();
    assert_eq!(limited.docs_written, 2);
    assert_eq!(
        fs::read_to_string(dir.join("two.jsonl"))
            .unwrap()
            .lines()
            .count(),
        2
    );

    // The output indexes with domain and crawl date filters.
    let opts = BuildOptions {
        keyword_fields: vec!["domain".into()],
        date_field: Some("crawl_date".into()),
        ..BuildOptions::default()
    };
    engine::build(&out, &dir.join("index"), &opts).unwrap();
    let index = Index::open(&dir.join("index")).unwrap();
    let search = |filters: Filters| {
        let req = Request {
            query: "hello rust x".into(),
            mode: Mode::Keyword,
            filters,
            ..Request::default()
        };
        let res = index.search(&req).unwrap();
        res.results.into_iter().map(|h| h.id).collect::<Vec<_>>()
    };
    let by_domain = [(
        "domain".to_string(),
        Filter::Any(vec!["example.com".into()]),
    )];
    assert_eq!(search(by_domain.into()), ["a"]);
    let by_date = [(
        "crawl_date".to_string(),
        Filter::Range {
            gte: Some("2014-01-01".into()),
            lte: None,
        },
    )];
    assert_eq!(search(by_date.into()), ["f"]);
}
