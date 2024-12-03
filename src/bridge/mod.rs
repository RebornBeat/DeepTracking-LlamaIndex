use crate::analysis::CodeTracker;
use crate::graph::DependencyGraph;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use serde_json::Value;
use std::collections::HashMap;
use std::path::PathBuf;

#[derive(Debug)]
pub struct LlamaIndexBridge {
    tracker: CodeTracker,
    index: Option<PyObject>,
    storage_context: Option<PyObject>,
}

#[derive(Debug)]
pub struct QueryResult {
    response: String,
    source_nodes: Vec<SourceNode>,
    relationships: Vec<RelationshipInfo>,
}

#[derive(Debug)]
struct SourceNode {
    content: String,
    metadata: HashMap<String, Value>,
    score: f32,
}

#[derive(Debug)]
struct RelationshipInfo {
    source: PathBuf,
    target: PathBuf,
    relationship_type: String,
    metadata: Value,
}

impl LlamaIndexBridge {
    pub fn new(root_path: PathBuf) -> Result<Self, crate::analysis::TrackerError> {
        Ok(LlamaIndexBridge {
            tracker: CodeTracker::new(root_path)?,
            index: None,
            storage_context: None,
        })
    }

    pub fn initialize(&mut self, py: Python<'_>) -> PyResult<()> {
        // Initialize LlamaIndex components
        let llama_index = py.import("llama_index")?;

        // Create storage context
        let storage_context = llama_index.getattr("StorageContext")?.call_method0("from_defaults")?;

        // Create service context with settings
        let service_context = llama_index
            .getattr("ServiceContext")?
            .call_method1(
                "from_defaults",
                (PyDict::new(py).apply(|d| {
                    let _ = d.set_item("chunk_size", 1024);
                    let _ = d.set_item("chunk_overlap", 128);
                }),),
            )?;

        // Analyze codebase
        self.tracker
            .analyze()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

        // Convert dependencies to documents with metadata
        let documents = self.create_documents(py)?;

        // Create vector store index
        let index = llama_index.getattr("VectorStoreIndex")?.call_method1(
            "from_documents",
            (documents, PyDict::new(py).apply(|d| {
                let _ = d.set_item("storage_context", storage_context.clone());
                let _ = d.set_item("service_context", service_context);
            }),),
        )?;

        self.index = Some(index);
        self.storage_context = Some(storage_context);

        Ok(())
    }

    fn create_documents(&self, py: Python<'_>) -> PyResult<PyObject> {
        let llama_index = py.import("llama_index")?;
        let documents = PyList::empty(py);

        // Get all analyzed files and their metadata
        let analyzed_files = self.tracker.get_analyzed_files()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

        for (file_path, metadata) in analyzed_files {
            let content = std::fs::read_to_string(&file_path)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

            // Create document with metadata
            let doc = llama_index.getattr("Document")?.call1((
                content,
                PyDict::new(py).apply(|d| {
                    let _ = d.set_item("file_path", file_path.to_string_lossy().to_string());
                    let _ = d.set_item("metadata", metadata);
                }),
            ))?;

            documents.append(doc)?;
        }

        Ok(documents.into())
    }

    pub fn query(&self, py: Python<'_>, query: &str) -> PyResult<QueryResult> {
        let index = self.index.as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Index not initialized"))?;

        // Enhance query with dependency context
        let enhanced_query = self.enhance_query(query)?;

        // Create query engine with custom settings
        let query_engine = index.call_method1(
            "as_query_engine",
            (PyDict::new(py).apply(|d| {
                let _ = d.set_item("similarity_top_k", 5);
                let _ = d.set_item("response_mode", "tree_summarize");
            }),),
        )?;

        // Execute query
        let response = query_engine.call_method1("query", (enhanced_query,))?;

        // Extract response text and source nodes
        let response_text = response.getattr("response")?.extract::<String>()?;
        let source_nodes = self.extract_source_nodes(py, response.getattr("source_nodes")?)?;

        // Get relationship information for the source nodes
        let relationships = self.extract_relationships(&source_nodes)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

        Ok(QueryResult {
            response: response_text,
            source_nodes,
            relationships,
        })
    }

    fn enhance_query(&self, query: &str) -> Result<String, String> {
        // Enhance the query with dependency context
        let mut enhanced = String::from(query);

        // Add structural hints
        enhanced.push_str("\nConsider code dependencies and relationships such as:");
        enhanced.push_str("\n- Function calls and definitions");
        enhanced.push_str("\n- Module imports and usage");
        enhanced.push_str("\n- Type relationships and inheritance");
        enhanced.push_str("\n- File organization and project structure");

        Ok(enhanced)
    }

    fn extract_source_nodes(&self, py: Python<'_>, nodes: PyObject) -> PyResult<Vec<SourceNode>> {
        let nodes_list = nodes.extract::<Vec<PyObject>>()?;
        let mut source_nodes = Vec::new();

        for node in nodes_list {
            let content = node.getattr(py, "text")?.extract::<String>()?;
            let metadata = node
                .getattr(py, "metadata")?
                .extract::<HashMap<String, Value>>()?;
            let score = node
                .getattr(py, "score")?
                .extract::<f32>()
                .unwrap_or(0.0);

            source_nodes.push(SourceNode {
                content,
                metadata,
                score,
            });
        }

        Ok(source_nodes)
    }

    fn extract_relationships(&self, source_nodes: &[SourceNode]) -> Result<Vec<RelationshipInfo>, String> {
        let mut relationships = Vec::new();

        for node in source_nodes {
            if let Some(file_path) = node.metadata.get("file_path") {
                if let Some(path_str) = file_path.as_str() {
                    let path = PathBuf::from(path_str);

                    // Get dependencies for this file
                    let deps = self.tracker.get_dependencies(&path)?;

                    for dep in deps {
                        relationships.push(RelationshipInfo {
                            source: path.clone(),
                            target: PathBuf::from(dep),
                            relationship_type: "dependency".to_string(),
                            metadata: json!({
                                "type": "code_dependency",
                                "score": node.score,
                            }),
                        });
                    }
                }
            }
        }

        Ok(relationships)
    }

    pub fn persist_index(&self, py: Python<'_>, path: &str) -> PyResult<()> {
        if let Some(storage_context) = &self.storage_context {
            storage_context.call_method1("persist", (path,))?;
        }
        Ok(())
    }

    pub fn load_index(&mut self, py: Python<'_>, path: &str) -> PyResult<()> {
        let llama_index = py.import("llama_index")?;

        // Load storage context
        let storage_context = llama_index
            .getattr("StorageContext")?
            .call_method1("from_defaults", (PyDict::new(py).apply(|d| {
                let _ = d.set_item("persist_dir", path);
            }),))?;

        // Load index
        let index = llama_index.getattr("load_index_from_storage")?.call1((storage_context.clone(),))?;

        self.index = Some(index);
        self.storage_context = Some(storage_context);

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

    fn initialize(&mut self, py: Python<'_>) -> PyResult<()> {
        self.0.initialize(py)
    }

    fn query(&self, py: Python<'_>, query: String) -> PyResult<PyObject> {
        let result = self.0.query(py, &query)?;

        // Convert result to Python dictionary
        let result_dict = PyDict::new(py);
        result_dict.set_item("response", result.response)?;
        result_dict.set_item("source_nodes", Self::source_nodes_to_py(py, &result.source_nodes)?)?;
        result_dict.set_item("relationships", Self::relationships_to_py(py, &result.relationships)?)?;

        Ok(result_dict.into())
    }

    fn persist_index(&self, py: Python<'_>, path: String) -> PyResult<()> {
        self.0.persist_index(py, &path)
    }

    fn load_index(&mut self, py: Python<'_>, path: String) -> PyResult<()> {
        self.0.load_index(py, &path)
    }
}

impl PyLlamaIndexBridge {
    fn source_nodes_to_py(py: Python<'_>, nodes: &[SourceNode]) -> PyResult<PyObject> {
        let list = PyList::empty(py);
        for node in nodes {
            let dict = PyDict::new(py);
            dict.set_item("content", &node.content)?;
            dict.set_item("metadata", &node.metadata)?;
            dict.set_item("score", node.score)?;
            list.append(dict)?;
        }
        Ok(list.into())
    }

    fn relationships_to_py(py: Python<'_>, relationships: &[RelationshipInfo]) -> PyResult<PyObject> {
        let list = PyList::empty(py);
        for rel in relationships {
            let dict = PyDict::new(py);
            dict.set_item("source", rel.source.to_string_lossy().to_string())?;
            dict.set_item("target", rel.target.to_string_lossy().to_string())?;
            dict.set_item("relationship_type", &rel.relationship_type)?;
            dict.set_item("metadata", &rel.metadata)?;
            list.append(dict)?;
        }
        Ok(list.into())
    }
}
