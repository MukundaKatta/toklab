//! PyO3 bindings exposing `toklab_core` as `toklab._native`.
//!
//! Bulk paths release the GIL via `py.allow_threads`; single calls do too
//! to keep behavior consistent under threaded callers.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyString;

use toklab_core::{Tokenizer, TokenizerError};

pyo3::create_exception!(_native, ToklabError, pyo3::exceptions::PyException);

fn map_err(e: TokenizerError) -> PyErr {
    match e {
        TokenizerError::UnknownEncoding(_) => PyValueError::new_err(e.to_string()),
        other => ToklabError::new_err(other.to_string()),
    }
}

#[pyclass(name = "Tokenizer", module = "toklab._native")]
struct PyTokenizer {
    inner: Tokenizer,
}

#[pymethods]
impl PyTokenizer {
    /// Construct from an OpenAI model name.
    #[staticmethod]
    fn for_model(model: &str) -> PyResult<Self> {
        Ok(Self {
            inner: Tokenizer::for_model(model).map_err(map_err)?,
        })
    }

    /// Construct from an encoding name (`cl100k_base` or `o200k_base`).
    #[staticmethod]
    fn for_encoding(name: &str) -> PyResult<Self> {
        Ok(Self {
            inner: Tokenizer::for_encoding(name).map_err(map_err)?,
        })
    }

    /// Encoding name (`cl100k_base` or `o200k_base`).
    #[getter]
    fn encoding_name(&self) -> &str {
        self.inner.encoding_name()
    }

    fn count(&self, py: Python<'_>, text: &str) -> usize {
        let owned = text.to_owned();
        py.allow_threads(move || self.inner.count(&owned))
    }

    #[pyo3(signature = (texts, parallel=false))]
    fn count_many(&self, py: Python<'_>, texts: Vec<String>, parallel: bool) -> Vec<usize> {
        py.allow_threads(move || {
            let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
            self.inner.count_many(&refs, parallel)
        })
    }

    fn encode(&self, py: Python<'_>, text: &str) -> Vec<u32> {
        let owned = text.to_owned();
        py.allow_threads(move || self.inner.encode(&owned))
    }

    fn decode(&self, py: Python<'_>, tokens: Vec<u32>) -> PyResult<String> {
        py.allow_threads(move || self.inner.decode(&tokens))
            .map_err(map_err)
    }

    #[pyo3(signature = (text, *, budget))]
    fn fits(&self, py: Python<'_>, text: &str, budget: usize) -> bool {
        let owned = text.to_owned();
        py.allow_threads(move || self.inner.fits(&owned, budget))
    }

    #[pyo3(signature = (text, *, budget))]
    fn truncate_to(&self, py: Python<'_>, text: &str, budget: usize) -> PyResult<String> {
        let owned = text.to_owned();
        py.allow_threads(move || self.inner.truncate_to(&owned, budget))
            .map_err(map_err)
    }

    fn __repr__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyString>> {
        Ok(PyString::new(
            py,
            &format!("Tokenizer(encoding='{}')", self.inner.encoding_name()),
        ))
    }
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("ToklabError", m.py().get_type::<ToklabError>())?;
    m.add_class::<PyTokenizer>()?;
    Ok(())
}
