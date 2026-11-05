use telusur::text::analyzer::{analyze, stem};

/// Martin Porter's official vocabulary and output lists
/// (https://tartarus.org/martin/PorterStemmer/voc.txt and output.txt), pasted side by side.
/// Lucene 10.5.0's PorterStemFilter also reproduces every pair.
#[test]
fn porter_reproduces_official_vocabulary() {
    let mut n = 0;
    for line in include_str!("data/porter.txt").lines() {
        let (word, want) = line.split_once(' ').unwrap();
        assert_eq!(stem(word), want, "{word}");
        n += 1;
    }
    assert_eq!(n, 23_531);
}

/// Inputs and the tokens Lucene 10.5.0 made for them with Anserini's DefaultEnglishAnalyzer
/// chain, captured with scripts/lucene_tokens.java.
#[test]
fn token_streams_match_lucene() {
    let cases = [
        (
            "John's car and the Smiths' house aren't O'Neil's.",
            "john car smith hous aren't o'neil",
        ),
        (
            "It’s Mary’s book, isn't it? JAMES'S dog's",
            "mari book isn't jame dog",
        ),
        (
            "Pi is 3.14159, about 1,000,000 times 2.5e10 and v1.2.3 or 4.5%",
            "pi 3.14159 about 1,000,000 time 2.5e10 v1.2.3 4.5",
        ),
        (
            "state-of-the-art e-mail co-operation well-known 50-year-old",
            "state art e mail co oper well known 50 year old",
        ),
        (
            "The quick brown fox jumps over the lazy dog, and that is not such a big deal for them.",
            "quick brown fox jump over lazi dog big deal them",
        ),
        (
            "Running runners ran easily; generalizations, relational, hopefulness",
            "run runner ran easili gener relat hope",
        ),
        (
            "U.S.A. and e.g. Ph.D. foo_bar snake_case_name",
            "u.s.a e.g ph.d foo_bar snake_case_nam",
        ),
        (
            "Visit https://example.com/path?q=1 or mail me@example.org #hashtag @user",
            "visit http example.com path q 1 mail me example.org hashtag user",
        ),
        (
            "Café naïve résumé Ελληνικά ΣΟΦΟΣ İstanbul",
            "café naïv résumé ελληνικά σοφοσ istanbul",
        ),
        (
            "中文分词 日本語のテキスト 한국어",
            "中 文 分 词 日 本 語 の テキスト 한국어",
        ),
        (
            "COVID-19 SARS-CoV-2 IL-6 p53 3D 2019-nCoV",
            "covid 19 sar cov 2 il 6 p53 3d 2019 ncov",
        ),
        ("Title Line\nbody text here", "titl line bodi text here"),
    ];
    for (input, want) in cases {
        assert_eq!(analyze(input).join(" "), want, "{input}");
    }

    // StandardTokenizer cuts words longer than 255 chars into pieces (Lucene: 255, 255, 90).
    let long = "x".repeat(600);
    let pieces = analyze(&format!("lorem {long} ipsum"));
    let lens: Vec<usize> = pieces.iter().map(String::len).collect();
    assert_eq!(lens, [5, 255, 255, 90, 5]);
}

/// Lucene keeps emoji and drops number symbols like ½ (`emoji 🙂 here 👍🏽`); the
/// letter-or-digit rule does the opposite. Pinned so a change is deliberate.
#[test]
fn known_difference_from_lucene() {
    assert_eq!(
        analyze("Emoji 🙂 here 👍🏽 and ⅕ ½ ①").join(" "),
        "emoji here ⅕ ½ ①"
    );
}
