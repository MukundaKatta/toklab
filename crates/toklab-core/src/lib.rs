//! Pure-Rust core for `toklab`. Thin wrapper around
//! [tiktoken-rs](https://crates.io/crates/tiktoken-rs) that adds:
//!
//! - **Bulk APIs** (`count_many`, optional rayon parallelism) — the win over
//!   pure-Python `tiktoken` is in long lists where Python interpreter
//!   overhead dominates.
//! - **Length-budgeting helpers** (`fits`, `truncate_to`) so the common
//!   patterns are one call instead of three.
//! - **Model-name lookup** that maps OpenAI model names to encodings via
//!   `tiktoken_rs::get_bpe_from_model`.
//!
//! Encodings supported out of the box: `cl100k_base` (GPT-3.5, GPT-4,
//! text-embedding-3-*) and `o200k_base` (GPT-4o family).

#![deny(unsafe_code)]
#![warn(missing_docs)]
#![warn(rust_2018_idioms)]

use rayon::prelude::*;
use thiserror::Error;
use tiktoken_rs::CoreBPE;

/// Crate-wide result alias.
pub type Result<T> = std::result::Result<T, TokenizerError>;

/// All errors surfaced by `toklab-core`.
#[derive(Error, Debug)]
pub enum TokenizerError {
    /// Unknown encoding name passed to [`Tokenizer::for_encoding`].
    #[error("unknown encoding: {0} (expected cl100k_base or o200k_base)")]
    UnknownEncoding(String),
    /// tiktoken-rs failed to load BPE tables. Should be unreachable for the
    /// bundled encodings; surfaces if a future version makes them optional.
    #[error("tiktoken-rs error: {0}")]
    Tiktoken(String),
}

/// Wraps a `CoreBPE` for one specific encoding.
pub struct Tokenizer {
    bpe: CoreBPE,
    encoding_name: String,
}

impl Tokenizer {
    /// Construct from an OpenAI model name (`"gpt-4"`, `"gpt-4o"`, etc.)
    /// using `tiktoken_rs::get_bpe_from_model`.
    pub fn for_model(model: &str) -> Result<Self> {
        let bpe = tiktoken_rs::get_bpe_from_model(model)
            .map_err(|e| TokenizerError::Tiktoken(e.to_string()))?;
        Ok(Self {
            bpe,
            encoding_name: encoding_for_model(model).to_string(),
        })
    }

    /// Construct from an encoding name. Accepts `"cl100k_base"` and
    /// `"o200k_base"`.
    pub fn for_encoding(name: &str) -> Result<Self> {
        let bpe =
            match name {
                "cl100k_base" => tiktoken_rs::cl100k_base()
                    .map_err(|e| TokenizerError::Tiktoken(e.to_string()))?,
                "o200k_base" => tiktoken_rs::o200k_base()
                    .map_err(|e| TokenizerError::Tiktoken(e.to_string()))?,
                other => return Err(TokenizerError::UnknownEncoding(other.to_string())),
            };
        Ok(Self {
            bpe,
            encoding_name: name.to_string(),
        })
    }

    /// Encoding name (`"cl100k_base"` or `"o200k_base"`).
    pub fn encoding_name(&self) -> &str {
        &self.encoding_name
    }

    /// Count BPE tokens in `text`, ignoring special tokens.
    pub fn count(&self, text: &str) -> usize {
        self.bpe.encode_ordinary(text).len()
    }

    /// Bulk count. With `parallel = true` distributes across rayon's pool.
    pub fn count_many(&self, texts: &[&str], parallel: bool) -> Vec<usize> {
        if parallel {
            texts
                .par_iter()
                .map(|t| self.bpe.encode_ordinary(t).len())
                .collect()
        } else {
            texts
                .iter()
                .map(|t| self.bpe.encode_ordinary(t).len())
                .collect()
        }
    }

    /// Encode to BPE token IDs (ordinary mode, no special tokens).
    pub fn encode(&self, text: &str) -> Vec<u32> {
        // tiktoken-rs 0.6 returns Vec<Rank> where Rank == u32; if a future
        // version changes this we'll catch it here.
        self.bpe.encode_ordinary(text)
    }

    /// Decode a slice of BPE token IDs back to a string.
    pub fn decode(&self, tokens: &[u32]) -> Result<String> {
        self.bpe
            .decode(tokens.to_vec())
            .map_err(|e| TokenizerError::Tiktoken(e.to_string()))
    }

    /// True iff `text` encodes to `<= budget` BPE tokens.
    pub fn fits(&self, text: &str, budget: usize) -> bool {
        self.count(text) <= budget
    }

    /// Encode `text`, truncate to the first `budget` tokens, and decode back.
    /// If `text` already fits, returns it unchanged. Boundary handling is
    /// whatever tiktoken-rs's `decode` does on a mid-token cut, which is
    /// well-defined for cl100k/o200k since each token decodes to a complete
    /// UTF-8 sequence in the merged-vocabulary case.
    pub fn truncate_to(&self, text: &str, budget: usize) -> Result<String> {
        let mut tokens = self.bpe.encode_ordinary(text);
        if tokens.len() <= budget {
            return Ok(text.to_string());
        }
        tokens.truncate(budget);
        self.bpe
            .decode(tokens)
            .map_err(|e| TokenizerError::Tiktoken(e.to_string()))
    }
}

/// Map an OpenAI model name to its encoding name. Used for diagnostics so
/// callers can see which encoding their model resolved to.
fn encoding_for_model(model: &str) -> &'static str {
    if model.starts_with("gpt-4o") || model.starts_with("o1") || model.starts_with("o3") {
        "o200k_base"
    } else {
        "cl100k_base"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_simple_text() {
        let tok = Tokenizer::for_encoding("cl100k_base").unwrap();
        let text = "hello world";
        let toks = tok.encode(text);
        let decoded = tok.decode(&toks).unwrap();
        assert_eq!(decoded, text);
    }

    #[test]
    fn count_matches_encode_len() {
        let tok = Tokenizer::for_encoding("cl100k_base").unwrap();
        let text = "the quick brown fox jumps over the lazy dog";
        assert_eq!(tok.count(text), tok.encode(text).len());
    }

    #[test]
    fn count_many_serial_and_parallel_agree() {
        let tok = Tokenizer::for_encoding("cl100k_base").unwrap();
        let texts: Vec<&str> = vec!["hi", "world", "lorem ipsum dolor sit amet"];
        let serial = tok.count_many(&texts, false);
        let par = tok.count_many(&texts, true);
        assert_eq!(serial, par);
    }

    #[test]
    fn for_model_gpt4_is_cl100k() {
        let tok = Tokenizer::for_model("gpt-4").unwrap();
        assert_eq!(tok.encoding_name(), "cl100k_base");
    }

    #[test]
    fn for_model_gpt4o_is_o200k() {
        let tok = Tokenizer::for_model("gpt-4o").unwrap();
        assert_eq!(tok.encoding_name(), "o200k_base");
    }

    #[test]
    fn unknown_encoding_rejected() {
        assert!(Tokenizer::for_encoding("unknown_base").is_err());
    }

    #[test]
    fn fits_and_truncate() {
        let tok = Tokenizer::for_encoding("cl100k_base").unwrap();
        let text = "the quick brown fox";
        let n = tok.count(text);
        assert!(tok.fits(text, n));
        assert!(tok.fits(text, n + 1));
        assert!(!tok.fits(text, n - 1));

        let truncated = tok.truncate_to(text, 2).unwrap();
        assert!(tok.count(&truncated) <= 2);
        assert!(truncated.len() <= text.len());
    }

    #[test]
    fn truncate_returns_input_when_fits() {
        let tok = Tokenizer::for_encoding("cl100k_base").unwrap();
        let text = "hi";
        assert_eq!(tok.truncate_to(text, 100).unwrap(), text);
    }

    #[test]
    fn empty_text_is_zero_tokens() {
        let tok = Tokenizer::for_encoding("cl100k_base").unwrap();
        assert_eq!(tok.count(""), 0);
        assert_eq!(tok.encode(""), Vec::<u32>::new());
    }

    #[test]
    fn unicode_text_round_trips() {
        let tok = Tokenizer::for_encoding("cl100k_base").unwrap();
        let text = "你好世界 🌍";
        let toks = tok.encode(text);
        assert_eq!(tok.decode(&toks).unwrap(), text);
    }

    #[test]
    fn count_many_handles_empty_list() {
        let tok = Tokenizer::for_encoding("cl100k_base").unwrap();
        let empty: Vec<&str> = vec![];
        assert!(tok.count_many(&empty, false).is_empty());
        assert!(tok.count_many(&empty, true).is_empty());
    }
}
