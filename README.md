# toklab

Fast bulk tokenizer + token counter for OpenAI BPE encodings.
Rust core wrapping [tiktoken-rs](https://crates.io/crates/tiktoken-rs),
Python frontend.

## The problem

You need to count tokens for 100k strings to plan a context budget,
truncate inputs, or estimate cost. Pure-Python `tiktoken` does this fine
for one-shot calls, but a Python loop over a long list spends most of
its time in interpreter overhead and per-call init.

`toklab` keeps the same encoding (it uses `tiktoken-rs`, which ships the
exact byte tables from the official tiktoken release) but exposes a bulk
API that releases the GIL and parallelizes across cores.

## Install

```bash
pip install toklab
```

## 30-second quickstart

```python
from toklab import Tokenizer

tok = Tokenizer.for_model("gpt-4")

print(tok.count("hello world"))               # 2

texts = ["hello", "world", "lorem ipsum"]
print(tok.count_many(texts))                   # [1, 1, 4]
print(tok.count_many(texts, parallel=True))    # same, distributed across cores

# Length-budgeting helpers.
print(tok.fits("hello world", budget=5))       # True
print(tok.truncate_to("a long sentence" * 100, budget=20))
```

## API

```python
class Tokenizer:
    @classmethod
    def for_model(cls, model: str) -> Tokenizer: ...

    @classmethod
    def for_encoding(cls, name: str) -> Tokenizer: ...
    # name in {"cl100k_base", "o200k_base"}

    def count(self, text: str) -> int: ...
    def count_many(self, texts: list[str], *, parallel: bool = False) -> list[int]: ...

    def encode(self, text: str) -> list[int]: ...
    def decode(self, tokens: list[int]) -> str: ...

    def fits(self, text: str, *, budget: int) -> bool: ...
    def truncate_to(self, text: str, *, budget: int) -> str: ...
```

## License

Dual-licensed under MIT or Apache-2.0 at your option.
