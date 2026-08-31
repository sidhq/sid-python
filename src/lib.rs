use std::collections::{HashMap, HashSet};

use alyze::analyze::{
    AnalysisOptions, Analyzer, LanguageWithStopwords, ReusableBuffer, StopwordRemoval,
    TokenizerOptions,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

const K1: f64 = 1.2;
const B: f64 = 0.75;
const UNIGRAM_WEIGHT: f64 = 1.0;
const BIGRAM_WEIGHT: f64 = 1.0;
const SUPPORTED_LANGUAGES: &str = "danish, dutch, english, finnish, french, german, generic, hungarian, italian, norwegian, portuguese, russian, spanish, swedish";

#[derive(Clone, Copy, Debug)]
enum Language {
    Generic,
    Stopwords(LanguageWithStopwords),
}

impl Language {
    fn parse(value: &str) -> Result<Self, String> {
        let stopwords = match value {
            "generic" => return Ok(Self::Generic),
            "danish" => LanguageWithStopwords::Danish,
            "dutch" => LanguageWithStopwords::Dutch,
            "english" => LanguageWithStopwords::English,
            "finnish" => LanguageWithStopwords::Finnish,
            "french" => LanguageWithStopwords::French,
            "german" => LanguageWithStopwords::German,
            "hungarian" => LanguageWithStopwords::Hungarian,
            "italian" => LanguageWithStopwords::Italian,
            "norwegian" => LanguageWithStopwords::Norwegian,
            "portuguese" => LanguageWithStopwords::Portuguese,
            "russian" => LanguageWithStopwords::Russian,
            "spanish" => LanguageWithStopwords::Spanish,
            "swedish" => LanguageWithStopwords::Swedish,
            _ => {
                return Err(format!(
                    "unsupported language {value:?}; supported values: {SUPPORTED_LANGUAGES}"
                ));
            }
        };
        Ok(Self::Stopwords(stopwords))
    }

    fn analyzer(self) -> Analyzer {
        Analyzer::new(AnalysisOptions {
            tokenizer: TokenizerOptions::UAX29Word(Default::default()),
            maximum_token_length: None,
            case_sensitive: false,
            stopword_removal: match self {
                Self::Generic => None,
                Self::Stopwords(language) => Some(StopwordRemoval::ForLanguage(language)),
            },
            stemming: None,
            ascii_folding: false,
        })
    }
}

fn source_token_ranges(content: &str) -> Vec<(usize, usize)> {
    let analyzer = Analyzer::new(AnalysisOptions {
        tokenizer: TokenizerOptions::UAX29Word(Default::default()),
        maximum_token_length: None,
        case_sensitive: true,
        stopword_removal: None,
        stemming: None,
        ascii_folding: false,
    });
    let mut ranges = Vec::new();
    analyzer.analyze(content, &mut ReusableBuffer::new(), |token| {
        ranges.push((token.byte_range.start, token.byte_range.end));
        true
    });
    ranges
}

#[derive(Debug)]
struct QueryFeatures {
    unigram_ids: HashMap<String, usize>,
    bigram_ids: HashMap<String, HashMap<String, usize>>,
    bigram_count: usize,
}

fn query_features(query: &str, analyzer: Analyzer) -> QueryFeatures {
    let mut tokens = Vec::new();
    analyzer.analyze(query, &mut ReusableBuffer::new(), |token| {
        tokens.push(token.text.to_owned());
        true
    });

    let mut unigram_ids = HashMap::new();
    for token in &tokens {
        let next_id = unigram_ids.len();
        unigram_ids.entry(token.clone()).or_insert(next_id);
    }

    let mut bigram_ids: HashMap<String, HashMap<String, usize>> = HashMap::new();
    let mut seen_bigrams = HashSet::new();
    let mut bigram_count = 0;
    for pair in tokens.windows(2) {
        if seen_bigrams.insert((pair[0].clone(), pair[1].clone())) {
            bigram_ids
                .entry(pair[0].clone())
                .or_default()
                .insert(pair[1].clone(), bigram_count);
            bigram_count += 1;
        }
    }

    QueryFeatures {
        unigram_ids,
        bigram_ids,
        bigram_count,
    }
}

#[derive(Debug)]
struct ContentFeatures {
    filtered_positions: Vec<usize>,
    unigram_events: Vec<Vec<usize>>,
    bigram_events: Vec<Vec<(usize, usize)>>,
}

fn content_features(content: &str, analyzer: Analyzer, query: &QueryFeatures) -> ContentFeatures {
    let mut filtered_positions = Vec::new();
    let mut unigram_events = vec![Vec::new(); query.unigram_ids.len()];
    let mut bigram_events = vec![Vec::new(); query.bigram_count];
    let mut previous: Option<(String, usize)> = None;

    analyzer.analyze(content, &mut ReusableBuffer::new(), |token| {
        filtered_positions.push(token.position);
        if let Some(&feature_id) = query.unigram_ids.get(token.text) {
            unigram_events[feature_id].push(token.position);
        }
        if let Some((previous_text, previous_position)) = previous.as_ref()
            && let Some(next_tokens) = query.bigram_ids.get(previous_text.as_str())
            && let Some(&feature_id) = next_tokens.get(token.text)
        {
            bigram_events[feature_id].push((*previous_position, token.position));
        }
        previous = Some((token.text.to_owned(), token.position));
        true
    });

    ContentFeatures {
        filtered_positions,
        unigram_events,
        bigram_events,
    }
}

fn window_starts(source_token_count: usize, window_size: usize, stride: usize) -> Vec<usize> {
    let final_start = source_token_count - window_size;
    let mut starts: Vec<_> = (0..=final_start).step_by(stride).collect();
    if starts.last().copied() != Some(final_start) {
        starts.push(final_start);
    }
    starts
}

fn count_positions(positions: &[usize], start: usize, end: usize) -> usize {
    let left = positions.partition_point(|&position| position < start);
    let right = positions.partition_point(|&position| position < end);
    right - left
}

fn count_bigrams(events: &[(usize, usize)], start: usize, end: usize) -> usize {
    let left = events.partition_point(|&(left_position, _)| left_position < start);
    let right = events.partition_point(|&(_, right_position)| right_position < end);
    right.saturating_sub(left)
}

fn bm25_term_score(tf: usize, document_length: usize, average_length: f64, idf: f64) -> f64 {
    if tf == 0 || average_length == 0.0 {
        return 0.0;
    }
    let tf = tf as f64;
    let length_ratio = document_length as f64 / average_length;
    idf * (tf * (K1 + 1.0)) / (tf + K1 * (1.0 - B + B * length_ratio))
}

fn idf(document_count: usize, document_frequency: usize) -> f64 {
    (1.0 + (document_count as f64 - document_frequency as f64 + 0.5)
        / (document_frequency as f64 + 0.5))
        .ln()
}

fn score_unigrams(
    starts: &[usize],
    window_size: usize,
    document_lengths: &[usize],
    average_length: f64,
    events_by_feature: &[Vec<usize>],
    scores: &mut [f64],
) {
    for events in events_by_feature {
        let document_frequency = starts
            .iter()
            .filter(|&&start| count_positions(events, start, start + window_size) > 0)
            .count();
        if document_frequency == 0 {
            continue;
        }
        let feature_idf = idf(starts.len(), document_frequency);
        for (window_index, &start) in starts.iter().enumerate() {
            let tf = count_positions(events, start, start + window_size);
            scores[window_index] += UNIGRAM_WEIGHT
                * bm25_term_score(
                    tf,
                    document_lengths[window_index],
                    average_length,
                    feature_idf,
                );
        }
    }
}

fn score_bigrams(
    starts: &[usize],
    window_size: usize,
    document_lengths: &[usize],
    average_length: f64,
    events_by_feature: &[Vec<(usize, usize)>],
    scores: &mut [f64],
) {
    for events in events_by_feature {
        let document_frequency = starts
            .iter()
            .filter(|&&start| count_bigrams(events, start, start + window_size) > 0)
            .count();
        if document_frequency == 0 {
            continue;
        }
        let feature_idf = idf(starts.len(), document_frequency);
        for (window_index, &start) in starts.iter().enumerate() {
            let tf = count_bigrams(events, start, start + window_size);
            scores[window_index] += BIGRAM_WEIGHT
                * bm25_term_score(
                    tf,
                    document_lengths[window_index],
                    average_length,
                    feature_idf,
                );
        }
    }
}

fn byte_span_to_codepoints(content: &str, start_byte: usize, end_byte: usize) -> (usize, usize) {
    let start = content[..start_byte].chars().count();
    let end = start + content[start_byte..end_byte].chars().count();
    (start, end)
}

fn select_snippet(
    query: &str,
    content: &str,
    window_size: usize,
    stride: usize,
    language: Language,
) -> (usize, usize) {
    let source_ranges = source_token_ranges(content);
    if source_ranges.len() <= window_size {
        return (0, content.chars().count());
    }

    let analyzer = language.analyzer();
    let query = query_features(query, analyzer);
    let content_features = content_features(content, analyzer, &query);
    let starts = window_starts(source_ranges.len(), window_size, stride);

    let unigram_lengths: Vec<_> = starts
        .iter()
        .map(|&start| {
            count_positions(
                &content_features.filtered_positions,
                start,
                start + window_size,
            )
        })
        .collect();
    let bigram_lengths: Vec<_> = starts
        .iter()
        .map(|&start| {
            let filtered_count = count_positions(
                &content_features.filtered_positions,
                start,
                start + window_size,
            );
            filtered_count.saturating_sub(1)
        })
        .collect();
    let average_unigram_length = unigram_lengths.iter().sum::<usize>() as f64 / starts.len() as f64;
    let average_bigram_length = bigram_lengths.iter().sum::<usize>() as f64 / starts.len() as f64;

    let mut scores = vec![0.0; starts.len()];
    score_unigrams(
        &starts,
        window_size,
        &unigram_lengths,
        average_unigram_length,
        &content_features.unigram_events,
        &mut scores,
    );
    score_bigrams(
        &starts,
        window_size,
        &bigram_lengths,
        average_bigram_length,
        &content_features.bigram_events,
        &mut scores,
    );

    let best_index = scores
        .iter()
        .enumerate()
        .max_by(|(left_index, left), (right_index, right)| {
            left.total_cmp(right)
                .then_with(|| right_index.cmp(left_index))
        })
        .map(|(index, _)| index)
        .unwrap_or(0);
    let best_start = starts[best_index];
    let start_byte = source_ranges[best_start].0;
    let end_byte = source_ranges[best_start + window_size - 1].1;
    byte_span_to_codepoints(content, start_byte, end_byte)
}

/// Select the most relevant fixed-size source-token window with unigram and bigram BM25.
#[pyfunction(signature = (query, content, window_size=50, stride=10, language="english"))]
fn bm25_snippet_with_stride(
    py: Python<'_>,
    query: &str,
    content: &str,
    window_size: isize,
    stride: isize,
    language: &str,
) -> PyResult<(usize, usize)> {
    if window_size <= 0 {
        return Err(PyValueError::new_err(
            "window_size must be greater than zero",
        ));
    }
    if stride <= 0 {
        return Err(PyValueError::new_err("stride must be greater than zero"));
    }
    let language = Language::parse(language).map_err(PyValueError::new_err)?;
    Ok(py.detach(|| {
        select_snippet(
            query,
            content,
            window_size as usize,
            stride as usize,
            language,
        )
    }))
}

#[pymodule]
fn _snippet(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(bm25_snippet_with_stride, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn select(query: &str, content: &str, size: usize, stride: usize) -> (usize, usize) {
        select_snippet(
            query,
            content,
            size,
            stride,
            Language::parse("english").unwrap(),
        )
    }

    #[test]
    fn bm25_formula_is_exact() {
        let actual = bm25_term_score(3, 10, 8.0, idf(5, 2));
        let expected_idf = (1.0_f64 + 3.5 / 2.5).ln();
        let expected = expected_idf * (3.0 * 2.2) / (3.0 + 1.2 * (0.25 + 0.75 * 1.25));
        assert!((actual - expected).abs() < 1e-12);
    }

    #[test]
    fn unigram_and_bigram_streams_normalize_independently() {
        let starts = [0, 2];
        let mut scores = [0.0, 0.0];
        let unigram_events = vec![vec![0, 2]];
        let bigram_events = vec![vec![(0, 1)]];

        score_unigrams(&starts, 3, &[3, 1], 2.0, &unigram_events, &mut scores);
        score_bigrams(&starts, 3, &[2, 0], 1.0, &bigram_events, &mut scores);

        let unigram_idf = idf(2, 2);
        let bigram_idf = idf(2, 1);
        let expected_first =
            bm25_term_score(2, 3, 2.0, unigram_idf) + bm25_term_score(1, 2, 1.0, bigram_idf);
        let expected_second = bm25_term_score(1, 1, 2.0, unigram_idf);
        assert!((scores[0] - expected_first).abs() < 1e-12);
        assert!((scores[1] - expected_second).abs() < 1e-12);
    }

    #[test]
    fn coherent_phrase_beats_scattered_terms() {
        let content = "contingent filler filler fee filler filler agreement pad pad pad contingent fee agreement tail tail tail";
        let (start, end) = select("contingent fee agreement", content, 6, 2);
        assert!(content[start..end].contains("contingent fee agreement"));
    }

    #[test]
    fn single_word_query_uses_unigrams() {
        let content = "zero one two three four target six seven eight nine ten eleven";
        let (start, end) = select("target", content, 4, 2);
        assert!(content[start..end].contains("target"));
    }

    #[test]
    fn stopwords_are_removed_before_bigrams() {
        let content = "alpha x x x beta x x x alpha and beta x x x x";
        let (start, end) = select("alpha and beta", content, 5, 2);
        assert!(content[start..end].contains("alpha and beta"));
    }

    #[test]
    fn duplicate_query_terms_do_not_multiply_weight() {
        let content = "zero one target three four five six target eight nine ten eleven";
        assert_eq!(
            select("target", content, 4, 2),
            select("target target target", content, 4, 2)
        );
    }

    #[test]
    fn earliest_tie_and_final_window_are_deterministic() {
        let content = "a b c d e f g h i j k";
        assert_eq!(select("missing", content, 4, 3), (0, 7));
        let (start, end) = select("k", content, 4, 20);
        assert_eq!(&content[start..end], "h i j k");
    }

    #[test]
    fn edge_inputs_have_fixed_fallbacks() {
        assert_eq!(select("query", "!!! ...", 5, 1), (0, 7));
        assert_eq!(select("query", "one two", 2, 1), (0, 7));
        assert_eq!(select("", "one two three four", 2, 1), (0, 7));
        assert_eq!(select("the and", "one two three four", 2, 1), (0, 7));
    }

    #[test]
    fn invalid_languages_list_supported_values() {
        let error = Language::parse("klingon").unwrap_err();
        assert!(error.contains("english"));
        assert!(error.contains("generic"));
        assert!(error.contains("swedish"));
    }

    #[test]
    fn unicode_offsets_are_python_codepoints() {
        for content in [
            "😀 zero one café target three four",
            "e\u{301} zero one target three four",
            "漢字 zero one target three four",
            "ภาษาไทย zero one target three four",
        ] {
            let (start, end) = select("target", content, 3, 1);
            let chars: Vec<_> = content.chars().collect();
            let selected: String = chars[start..end].iter().collect();
            assert!(selected.contains("target"));
        }
    }

    #[test]
    fn randomized_spans_are_valid_and_deterministic() {
        let alphabet = ["alpha", "βeta", "漢字", "😀", "the", "target"];
        let mut state = 7_u64;
        for _ in 0..250 {
            let mut words = Vec::new();
            for _ in 0..80 {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                words.push(alphabet[(state as usize) % alphabet.len()]);
            }
            let content = words.join(" ");
            let span = select("alpha target", &content, 11, 3);
            assert_eq!(span, select("alpha target", &content, 11, 3));
            assert!(span.0 < span.1 && span.1 <= content.chars().count());
            let slice: String = content.chars().skip(span.0).take(span.1 - span.0).collect();
            assert!(!slice.starts_with(char::is_whitespace));
            assert!(!slice.ends_with(char::is_whitespace));
        }
    }

    #[test]
    fn selector_is_thread_safe() {
        let handles: Vec<_> = (0..16)
            .map(|_| {
                std::thread::spawn(|| {
                    for _ in 0..100 {
                        assert_eq!(select("target", "a b c target e f", 3, 1), (2, 12));
                    }
                })
            })
            .collect();
        for handle in handles {
            handle.join().unwrap();
        }
    }
}
