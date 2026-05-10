# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1] - 2026-05-10

### Added
- `Tokenizer::for_model` now falls back to encoding-name routing when
  `tiktoken_rs::get_bpe_from_model` doesn't recognize the model name.
  This means new families (`gpt-5*`, `o4*`, `chatgpt-4o*`) resolve to a
  working tokenizer without waiting for the bundled mapping to update.
- Routing table for `o200k_base` extended: `gpt-5`, `o3`, `o4`,
  `chatgpt-4o` prefixes added alongside the existing `gpt-4o`, `o1`.

## [0.1.0] - 2026-05-09

### Added

- Initial public release.
- Rust core crate `toklab-core` wrapping
  [tiktoken-rs](https://crates.io/crates/tiktoken-rs) for cl100k and o200k
  encodings; no network at runtime.
- `Tokenizer::for_model("gpt-4")` mapping plus explicit
  `Tokenizer::for_encoding("cl100k_base"|"o200k_base")`.
- Bulk counting (`count_many`) with optional parallel execution via rayon.
- `fits(text, budget)` and `truncate_to(text, budget)` length-budgeting
  helpers.
- Python package `toklab` with PyO3 bindings releasing the GIL on every
  bulk call.
- abi3-py310 wheel: one wheel for CPython 3.10 through 3.13.

[Unreleased]: https://github.com/MukundaKatta/toklab/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/MukundaKatta/toklab/releases/tag/v0.1.1
[0.1.0]: https://github.com/MukundaKatta/toklab/releases/tag/v0.1.0
