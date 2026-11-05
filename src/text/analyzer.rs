//! Anserini's `DefaultEnglishAnalyzer`: Unicode word boundaries, English possessive removal,
//! lowercase, Lucene's English stopwords, then the original Porter stemmer. Docs and queries
//! both go through `analyze`.
//!
//! One known difference from Lucene's `StandardTokenizer`: a word must contain a letter or
//! digit, so emoji are dropped (Lucene keeps them) and number symbols like `½` or `①` are
//! kept (Lucene drops them).

use unicode_segmentation::UnicodeSegmentation;

/// Lucene's `EnglishAnalyzer.ENGLISH_STOP_WORDS_SET`.
pub const STOPWORDS: [&str; 33] = [
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into", "is", "it",
    "no", "not", "of", "on", "or", "such", "that", "the", "their", "then", "there", "these",
    "they", "this", "to", "was", "will", "with",
];

/// `StandardTokenizer`'s default max token length. Longer words are cut into pieces this long.
const MAX_TOKEN_CHARS: usize = 255;

pub fn analyze(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    for word in text.unicode_words() {
        let mut rest = word;
        while !rest.is_empty() {
            let end = rest
                .char_indices()
                .nth(MAX_TOKEN_CHARS)
                .map_or(rest.len(), |(i, _)| i);
            let (piece, tail) = rest.split_at(end);
            rest = tail;
            // Lucene lowercases one char at a time: no final sigma rule, and İ becomes i
            // (the first char of Rust's two-char mapping).
            let lower: String = strip_possessive(piece)
                .chars()
                .map(|c| c.to_lowercase().next().unwrap())
                .collect();
            if !STOPWORDS.contains(&lower.as_str()) {
                tokens.push(stem(&lower));
            }
        }
    }
    tokens
}

/// Lucene's `EnglishPossessiveFilter`: drops a trailing `'s` (any of three apostrophes).
fn strip_possessive(t: &str) -> &str {
    t.strip_suffix(['s', 'S'])
        .and_then(|r| r.strip_suffix(['\'', '\u{2019}', '\u{FF07}']))
        .unwrap_or(t)
}

/// The original Porter stemmer, ported from Lucene's `PorterStemmer` (Porter's own reference
/// code, including its two departures from the paper: bli -> ble and logi -> log). Like Java
/// it works on UTF-16 units, and anything but a, e, i, o, u and y is a consonant. Words of
/// one or two units are left alone.
pub fn stem(word: &str) -> String {
    let b: Vec<u16> = word.encode_utf16().collect();
    if b.len() < 3 {
        return word.to_string();
    }
    let mut p = Porter {
        k: b.len() as isize - 1,
        j: 0,
        b,
    };
    p.step1ab();
    p.step1c();
    p.step2();
    p.step3();
    p.step4();
    p.step5();
    p.b.truncate(p.k as usize + 1);
    String::from_utf16(&p.b).expect("stemming only removes ASCII, never half a surrogate pair")
}

struct Porter {
    b: Vec<u16>,
    /// Index of the last char of the word so far.
    k: isize,
    /// Index of the last char before the suffix `ends` last matched.
    j: isize,
}

impl Porter {
    fn byte(&self, i: isize) -> u8 {
        u8::try_from(self.b[i as usize]).unwrap_or(0)
    }

    fn cons(&self, i: isize) -> bool {
        match self.byte(i) {
            b'a' | b'e' | b'i' | b'o' | b'u' => false,
            b'y' => i == 0 || !self.cons(i - 1),
            _ => true,
        }
    }

    /// The number of vowel-consonant sequences in `b[0..=j]`.
    fn m(&self) -> usize {
        let (mut n, mut i) = (0, 0);
        loop {
            if i > self.j {
                return n;
            }
            if !self.cons(i) {
                break;
            }
            i += 1;
        }
        i += 1;
        loop {
            loop {
                if i > self.j {
                    return n;
                }
                if self.cons(i) {
                    break;
                }
                i += 1;
            }
            i += 1;
            n += 1;
            loop {
                if i > self.j {
                    return n;
                }
                if !self.cons(i) {
                    break;
                }
                i += 1;
            }
            i += 1;
        }
    }

    fn vowel_in_stem(&self) -> bool {
        (0..=self.j).any(|i| !self.cons(i))
    }

    fn double_cons(&self, i: isize) -> bool {
        i >= 1 && self.b[i as usize] == self.b[i as usize - 1] && self.cons(i)
    }

    /// Consonant, vowel, consonant ending at `i`, where the last is not w, x or y.
    fn cvc(&self, i: isize) -> bool {
        i >= 2
            && self.cons(i)
            && !self.cons(i - 1)
            && self.cons(i - 2)
            && !matches!(self.byte(i), b'w' | b'x' | b'y')
    }

    /// Whether the word ends with `s`. On a match, sets `j` to just before it.
    fn ends(&mut self, s: &str) -> bool {
        let o = self.k - s.len() as isize + 1;
        if o < 0
            || !s
                .bytes()
                .zip(&self.b[o as usize..])
                .all(|(c, &u)| u == u16::from(c))
        {
            return false;
        }
        self.j = o - 1;
        true
    }

    /// Replaces everything after `j` with `s`.
    fn set_to(&mut self, s: &str) {
        self.b.truncate((self.j + 1) as usize);
        self.b.extend(s.bytes().map(u16::from));
        self.k = self.b.len() as isize - 1;
    }

    fn r(&mut self, s: &str) {
        if self.m() > 0 {
            self.set_to(s);
        }
    }

    /// Replaces the first matching suffix if the stem before it has m > 0.
    fn replace_first(&mut self, rules: &[(&str, &str)]) {
        if let Some(&(_, to)) = rules.iter().find(|(from, _)| self.ends(from)) {
            self.r(to);
        }
    }

    /// Plurals and -ed or -ing.
    fn step1ab(&mut self) {
        if self.byte(self.k) == b's' {
            if self.ends("sses") {
                self.k -= 2;
            } else if self.ends("ies") {
                self.set_to("i");
            } else if self.byte(self.k - 1) != b's' {
                self.k -= 1;
            }
        }
        if self.ends("eed") {
            if self.m() > 0 {
                self.k -= 1;
            }
        } else if (self.ends("ed") || self.ends("ing")) && self.vowel_in_stem() {
            self.k = self.j;
            if self.ends("at") {
                self.set_to("ate");
            } else if self.ends("bl") {
                self.set_to("ble");
            } else if self.ends("iz") {
                self.set_to("ize");
            } else if self.double_cons(self.k) {
                if !matches!(self.byte(self.k), b'l' | b's' | b'z') {
                    self.k -= 1;
                }
            } else if self.m() == 1 && self.cvc(self.k) {
                self.set_to("e");
            }
        }
    }

    /// Terminal y to i when there is another vowel in the stem.
    fn step1c(&mut self) {
        if self.ends("y") && self.vowel_in_stem() {
            self.b[self.k as usize] = u16::from(b'i');
        }
    }

    /// Double suffixes to single ones.
    fn step2(&mut self) {
        self.replace_first(&[
            ("ational", "ate"),
            ("tional", "tion"),
            ("enci", "ence"),
            ("anci", "ance"),
            ("izer", "ize"),
            ("bli", "ble"),
            ("alli", "al"),
            ("entli", "ent"),
            ("eli", "e"),
            ("ousli", "ous"),
            ("ization", "ize"),
            ("ation", "ate"),
            ("ator", "ate"),
            ("alism", "al"),
            ("iveness", "ive"),
            ("fulness", "ful"),
            ("ousness", "ous"),
            ("aliti", "al"),
            ("iviti", "ive"),
            ("biliti", "ble"),
            ("logi", "log"),
        ]);
    }

    /// -ic-, -full, -ness and similar.
    fn step3(&mut self) {
        self.replace_first(&[
            ("icate", "ic"),
            ("ative", ""),
            ("alize", "al"),
            ("iciti", "ic"),
            ("ical", "ic"),
            ("ful", ""),
            ("ness", ""),
        ]);
    }

    /// Drops -ant, -ence and the rest when the stem has m > 1.
    fn step4(&mut self) {
        const SUFFIXES: [&str; 19] = [
            "al", "ance", "ence", "er", "ic", "able", "ible", "ant", "ement", "ment", "ent", "ion",
            "ou", "ism", "ate", "iti", "ous", "ive", "ize",
        ];
        let found = SUFFIXES.iter().any(|s| {
            self.ends(s) && (*s != "ion" || self.j >= 0 && matches!(self.byte(self.j), b's' | b't'))
        });
        if found && self.m() > 1 {
            self.k = self.j;
        }
    }

    /// Drops a final -e and turns -ll into -l, when m allows.
    fn step5(&mut self) {
        self.j = self.k;
        if self.byte(self.k) == b'e' {
            let m = self.m();
            if m > 1 || m == 1 && !self.cvc(self.k - 1) {
                self.k -= 1;
            }
        }
        if self.byte(self.k) == b'l' && self.double_cons(self.k) && self.m() > 1 {
            self.k -= 1;
        }
    }
}
