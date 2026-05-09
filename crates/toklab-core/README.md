# toklab-core

Pure-Rust core for [toklab](https://github.com/MukundaKatta/toklab):
bulk tokenizer + counter for OpenAI BPE encodings, wrapping
[tiktoken-rs](https://crates.io/crates/tiktoken-rs).

```rust
use toklab_core::Tokenizer;

let tok = Tokenizer::for_model("gpt-4")?;
assert_eq!(tok.count("hello world"), 2);
let many = tok.count_many(&["hello", "world"], false);
assert_eq!(many, vec![1, 1]);
# Ok::<(), toklab_core::TokenizerError>(())
```

## License

Dual-licensed under MIT or Apache-2.0.
