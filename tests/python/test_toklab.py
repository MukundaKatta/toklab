"""End-to-end tests for the Python facade."""

from __future__ import annotations

import pytest
from toklab import Tokenizer, ToklabError, __version__


def test_version_present() -> None:
    assert isinstance(__version__, str)
    assert __version__ != ""


def test_for_model_gpt4_resolves_cl100k() -> None:
    t = Tokenizer.for_model("gpt-4")
    assert t.encoding_name == "cl100k_base"


def test_for_model_gpt4o_resolves_o200k() -> None:
    t = Tokenizer.for_model("gpt-4o")
    assert t.encoding_name == "o200k_base"


def test_for_encoding_explicit() -> None:
    t = Tokenizer.for_encoding("cl100k_base")
    assert t.encoding_name == "cl100k_base"


def test_count_simple() -> None:
    t = Tokenizer.for_model("gpt-4")
    assert t.count("hello world") == 2
    assert t.count("") == 0


def test_count_many_serial_and_parallel_match() -> None:
    t = Tokenizer.for_model("gpt-4")
    texts = ["hi", "world", "lorem ipsum dolor sit amet"] * 4
    serial = t.count_many(texts)
    parallel = t.count_many(texts, parallel=True)
    assert serial == parallel
    assert all(n > 0 for n in serial)


def test_count_many_empty_list() -> None:
    t = Tokenizer.for_model("gpt-4")
    assert t.count_many([]) == []
    assert t.count_many([], parallel=True) == []


def test_encode_decode_round_trip() -> None:
    t = Tokenizer.for_model("gpt-4")
    text = "the quick brown fox jumps over the lazy dog"
    tokens = t.encode(text)
    assert all(isinstance(x, int) for x in tokens)
    assert t.decode(tokens) == text


def test_unicode_round_trip() -> None:
    t = Tokenizer.for_model("gpt-4")
    text = "你好世界 🌍"
    assert t.decode(t.encode(text)) == text


def test_fits_boundary() -> None:
    t = Tokenizer.for_model("gpt-4")
    n = t.count("hello world")
    assert t.fits("hello world", budget=n)
    assert t.fits("hello world", budget=n + 1)
    assert not t.fits("hello world", budget=n - 1)


def test_truncate_to_returns_input_when_fits() -> None:
    t = Tokenizer.for_model("gpt-4")
    assert t.truncate_to("hi", budget=100) == "hi"


def test_truncate_to_shrinks_long_text() -> None:
    t = Tokenizer.for_model("gpt-4")
    text = "the quick brown fox jumps over the lazy dog"
    out = t.truncate_to(text, budget=3)
    assert t.count(out) <= 3
    assert len(out) < len(text)


def test_truncate_to_zero_is_empty() -> None:
    t = Tokenizer.for_model("gpt-4")
    assert t.truncate_to("hello world", budget=0) == ""


def test_truncate_to_mid_multibyte_char_is_safe() -> None:
    # Cutting between the BPE tokens of a multi-byte character must not raise
    # an invalid-UTF-8 error; the result stays within budget and is valid.
    t = Tokenizer.for_model("gpt-4")
    text = "你好世界 🌍"
    full = t.count(text)
    for budget in range(full + 1):
        out = t.truncate_to(text, budget=budget)
        assert t.count(out) <= budget
    assert t.truncate_to(text, budget=full) == text


def test_unknown_encoding_rejected() -> None:
    with pytest.raises(ValueError, match="unknown encoding"):
        Tokenizer.for_encoding("not_a_thing")


def test_repr_includes_encoding() -> None:
    t = Tokenizer.for_model("gpt-4")
    text = repr(t)
    assert "cl100k_base" in text


def test_native_error_class_exposed() -> None:
    assert issubclass(ToklabError, Exception)


def test_count_matches_encode_len() -> None:
    t = Tokenizer.for_model("gpt-4")
    text = "lorem ipsum dolor sit amet, consectetur"
    assert t.count(text) == len(t.encode(text))


def test_two_models_independent() -> None:
    t4 = Tokenizer.for_model("gpt-4")
    t4o = Tokenizer.for_model("gpt-4o")
    text = "the quick brown fox"
    # Different encodings tokenize the same string to different IDs.
    assert t4.encode(text) != t4o.encode(text)
