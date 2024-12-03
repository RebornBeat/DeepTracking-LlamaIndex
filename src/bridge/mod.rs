use crate::analysis::CodeTracker;
use pyo3::prelude::*;

pub struct LlamaIndexBridge {
    tracker: CodeTracker,
}

impl LlamaIndexBridge {
    pub fn new(root_path: PathBuf) -> Result<Self, crate::analysis::TrackerError> {
        Ok(LlamaIndexBridge {
            tracker: CodeTracker::new(root_path)?,
        })
    }

    pub fn analyze_and_index(&mut self, py: Python<'_>) -> PyResult<()> {
        self.tracker
            .analyze()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

        let llama_index = py.import("llama_index")?;

        // Convert dependencies to LlamaIndex documents
        let deps = self
            .tracker
            .get_dependencies(&self.tracker.root_path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

        // Create LlamaIndex documents
        let documents = py.eval("[]", None, None)?;
        for dep in deps {
            let doc = py.eval(&format!("{{ 'text': '{}' }}", dep), None, None)?;
            documents.call_method1("append", (doc,))?;
        }

        // Create index
        let index = llama_index.call_method1("GPTVectorStoreIndex", (documents,))?;

        Ok(())
    }
}

#[pyclass]
pub struct PyLlamaIndexBridge(LlamaIndexBridge);

#[pymethods]
impl PyLlamaIndexBridge {
    #[new]
    fn new(root_path: String) -> PyResult<Self> {
        let bridge = LlamaIndexBridge::new(PathBuf::from(root_path))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(PyLlamaIndexBridge(bridge))
    }

    fn analyze_and_index(&mut self, py: Python<'_>) -> PyResult<()> {
        self.0.analyze_and_index(py)
    }

    fn query(&self, py: Python<'_>, query: String) -> PyResult<Vec<String>> {
        // Implement query functionality
        Ok(vec![])
    }
}
