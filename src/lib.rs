use pyo3::prelude::*;

mod analysis;
mod bridge;
mod graph;

#[pymodule]
fn deeptracking_llamaindex(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<bridge::PyLlamaIndexBridge>()?;
    Ok(())
}
