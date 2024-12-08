use crate::analyzers::manager::{AnalyzerManager, FileEntry, ProjectStructure};
use crate::graph::DependencyGraph;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

#[derive(Debug)]
pub struct LlamaIndexBridge {
    analyzer: AnalyzerManager,
    graph: Arc<RwLock<DependencyGraph>>,
    index: Option<PyObject>,
    storage_context: Option<PyObject>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct QueryResult {
    pub response: String,
    pub source_nodes: Vec<SourceNode>,
    pub project_context: ProjectContext,
    pub project_structure: ProjectStructure,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SourceNode {
    pub file_path: PathBuf,
    pub content: String,
    pub metadata: NodeMetadata,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct NodeMetadata {
    pub language: Option<String>,
    pub dependencies: Vec<String>,
    pub imports: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ProjectContext {
    pub analyzed_files: usize,
    pub file_relationships: HashMap<String, Vec<String>>,
    pub dependency_paths: Vec<DependencyPath>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DependencyPath {
    pub source: String,
    pub target: String,
    pub path: Vec<String>,
}

impl LlamaIndexBridge {
    pub fn new(root_path: PathBuf) -> Result<Self, String> {
        Ok(Self {
            analyzer: AnalyzerManager::new(&root_path)?,
            graph: Arc::new(RwLock::new(DependencyGraph::new())),
            index: None,
            storage_context: None,
        })
    }

    pub fn initialize(&mut self, py: Python<'_>) -> PyResult<()> {
        // Analyze project and build dependency graph
        let analysis_result = self
            .analyzer
            .analyze_project(&self.analyzer.root_path())
            .map_err(BridgeError::AnalyzerError)?;

        self.graph
            .write()
            .map_err(|e| BridgeError::GraphError(e.to_string()))?
            .add_dependencies(analysis_result.dependencies)
            .map_err(BridgeError::GraphError)?;

        // Initialize LlamaIndex components
        let llama_index = py.import("llama_index")?;

        // Create storage context
        let storage_context = llama_index
            .getattr("StorageContext")?
            .call_method0("from_defaults")?;

        let service_context_dict = PyDict::new(py);
        service_context_dict.set_item("chunk_size", 1024)?;
        service_context_dict.set_item("chunk_overlap", 128)?;

        // Create service context
        let service_context = llama_index
            .getattr("ServiceContext")?
            .call_method1("from_defaults", (service_context_dict,))?;

        // Create documents with metadata
        let documents = self.create_documents(py, &analysis_result.project_structure)?;

        let index_dict = PyDict::new(py);
        index_dict.set_item("storage_context", storage_context.clone())?;
        index_dict.set_item("service_context", service_context)?;

        // Create index
        let index = llama_index
            .getattr("VectorStoreIndex")?
            .call_method1("from_documents", (documents, index_dict))?;

        self.index = Some(index.into());
        self.storage_context = Some(storage_context.into());

        Ok(())
    }

    fn create_documents(
        &self,
        py: Python<'_>,
        project_structure: &ProjectStructure,
    ) -> PyResult<PyObject> {
        let llama_index = py.import("llama_index")?;
        let documents = PyList::empty(py);

        for file_entry in &project_structure.files {
            let metadata = self.create_metadata(py, file_entry)?;

            if let Ok(content) = std::fs::read_to_string(&file_entry.path) {
                let doc = llama_index
                    .getattr("Document")?
                    .call1((content, metadata))?;
                documents.append(doc)?;
            }
        }

        Ok(documents.into())
    }

    fn create_metadata(&self, py: Python<'_>, file_entry: &FileEntry) -> PyResult<PyObject> {
        let metadata = PyDict::new(py);

        metadata.set_item("file_path", file_entry.path.to_string_lossy().to_string())?;
        metadata.set_item("file_type", &file_entry.file_type)?;

        if let Some(ref file_metadata) = file_entry.metadata {
            metadata.set_item("language", &file_metadata.language)?;
            metadata.set_item("dependencies", &file_metadata.dependencies)?;
            metadata.set_item("size", file_metadata.size)?;

            // Add graph-based relationships
            if let Ok(graph) = self.graph.read() {
                if let Ok(relationships) = graph.get_direct_relationships(&file_entry.path) {
                    metadata.set_item("relationships", relationships)?;
                }
            }
        }

        Ok(metadata.into())
    }

    pub fn query(&self, py: Python<'_>, query: String) -> PyResult<QueryResult> {
        let index = self.index.as_ref().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Index not initialized")
        })?;

        // Enhance query with relationship context
        let enhanced_query = self
            .enhance_query(query)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;

        let query_dict = PyDict::new(py);
        query_dict.set_item("similarity_top_k", 5)?;
        query_dict.set_item("response_mode", "tree_summarize")?;

        // Create query engine
        let query_engine = index.call_method1(py, "as_query_engine", (query_dict,))?;

        // Execute query
        let response = query_engine.call_method1(py, "query", (enhanced_query,))?;

        // Process results
        let response_text = response.getattr(py, "response")?.extract::<String>(py)?;
        let source_nodes = response.getattr(py, "source_nodes")?;
        let extracted_nodes = self.extract_source_nodes(py, source_nodes)?;
        let project_context = self
            .build_project_context(&extracted_nodes)
            .map_err(BridgeError::AnalyzerError)?;

        Ok(QueryResult {
            response: response_text,
            source_nodes: extracted_nodes,
            project_context,
            project_structure: ProjectStructure {
                root: self.analyzer.root_path().to_string_lossy().to_string(),
                files: Vec::new(),
            },
        })
    }

    fn enhance_query(&self, query: String) -> Result<String, String> {
        let mut enhanced = query.clone();

        // Add structural context
        enhanced.push_str("\nConsider the following relationships:");
        enhanced.push_str("\n- Direct function calls and definitions");
        enhanced.push_str("\n- Module imports and dependencies");
        enhanced.push_str("\n- Class inheritance and usage patterns");

        // Add specific relationship hints based on query content
        if query.contains("function") || query.contains("calls") {
            enhanced.push_str("\nFocus on function call relationships and dependencies");
        }

        if query.contains("import") || query.contains("module") {
            enhanced.push_str("\nFocus on module relationships and import hierarchy");
        }

        Ok(enhanced)
    }

    fn extract_source_nodes(&self, py: Python<'_>, nodes: PyObject) -> PyResult<Vec<SourceNode>> {
        let nodes_list = nodes.extract::<Vec<PyObject>>(py)?;
        let mut source_nodes = Vec::new();

        for node in nodes_list {
            let content = node.getattr(py, "text")?.extract::<String>(py)?;
            let metadata = node
                .getattr(py, "metadata")?
                .extract::<HashMap<String, PyObject>>(py)?;

            source_nodes.push(SourceNode {
                file_path: PathBuf::from(
                    metadata
                        .get("file_path")
                        .ok_or_else(|| {
                            PyErr::new::<pyo3::exceptions::PyValueError, _>("Missing file_path")
                        })?
                        .extract::<String>(py)?,
                ),
                content,
                metadata: self.extract_node_metadata(py, &metadata)?,
            });
        }

        Ok(source_nodes)
    }

    fn extract_node_metadata(
        &self,
        py: Python<'_>,
        metadata: &HashMap<String, PyObject>,
    ) -> PyResult<NodeMetadata> {
        Ok(NodeMetadata {
            language: metadata
                .get("language")
                .map(|l| l.extract(py))
                .transpose()?,
            dependencies: metadata
                .get("dependencies")
                .map(|d| d.extract(py))
                .transpose()?
                .unwrap_or_default(),
            imports: metadata
                .get("imports")
                .map(|i| i.extract(py))
                .transpose()?
                .unwrap_or_default(),
        })
    }

    fn build_project_context(&self, source_nodes: &[SourceNode]) -> Result<ProjectContext, String> {
        let mut context = ProjectContext {
            analyzed_files: 0,
            file_relationships: HashMap::new(),
            dependency_paths: Vec::new(),
        };

        let graph = self
            .graph
            .read()
            .map_err(|e| format!("Failed to acquire read lock: {}", e))?;

        for node in source_nodes {
            // Handle direct relationships
            if let Ok(relationships) = graph.get_direct_relationships(&node.file_path) {
                context.file_relationships.insert(
                    node.file_path.to_string_lossy().to_string(),
                    relationships.values().flatten().cloned().collect(),
                );
            }

            // Handle dependency paths
            if let Ok(dependencies) = graph.get_dependencies(&node.file_path) {
                for dep_path in dependencies {
                    context.dependency_paths.push(DependencyPath {
                        source: node.file_path.to_string_lossy().to_string(),
                        target: dep_path.clone(),
                        path: vec![node.file_path.to_string_lossy().to_string(), dep_path],
                    });
                }
            }
        }

        context.analyzed_files = source_nodes.len();
        Ok(context)
    }

    fn get_dependency_context(
        &self,
        file_path: &Path,
    ) -> Result<HashMap<String, Vec<String>>, String> {
        let graph = self
            .graph
            .read()
            .map_err(|e| format!("Failed to acquire read lock: {}", e))?;
        let path_buf = file_path.to_path_buf();
        let mut context = HashMap::new();

        if let Ok(deps) = graph.get_dependencies(&path_buf) {
            context.insert("direct_dependencies".to_string(), deps);
        }

        if let Ok(reverse_deps) = graph.get_callers(&path_buf) {
            context.insert(
                "reverse_dependencies".to_string(),
                reverse_deps
                    .iter()
                    .map(|p| p.to_string_lossy().to_string())
                    .collect(),
            );
        }

        if let Ok(related) = graph.get_direct_relationships(&path_buf) {
            for (rel_type, targets) in related {
                context.insert(rel_type, targets);
            }
        }

        Ok(context)
    }

    fn preprocess_query(&self, query: &str) -> String {
        let mut preprocessed = query.to_string();

        // Add code-specific context
        preprocessed = format!(
            "{}\n\nConsider the following code context when responding:\n{}",
            preprocessed,
            "- Focus on code structure and relationships\n\
             - Consider both direct and indirect dependencies\n\
             - Look for patterns in module organization\n\
             - Examine function calls and data flow"
        );

        preprocessed
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
        let result = self.0.query(py, query)?;

        // Convert result to Python dictionary
        let result_dict = PyDict::new(py);
        result_dict.set_item("response", result.response)?;
        result_dict.set_item(
            "source_nodes",
            Self::source_nodes_to_py(py, &result.source_nodes)?,
        )?;
        result_dict.set_item(
            "project_context",
            Self::project_context_to_py(py, &result.project_context)?,
        )?;

        Ok(result_dict.into())
    }

    fn persist_index(&self, py: Python<'_>, path: String) -> PyResult<()> {
        if let Some(storage_context) = &self.0.storage_context {
            storage_context.call_method1(py, "persist", (path,))?;
        }
        Ok(())
    }

    fn load_index(&mut self, py: Python<'_>, path: String) -> PyResult<()> {
        let llama_index = py.import("llama_index")?;

        let storage_dict = PyDict::new(py);
        storage_dict.set_item("persist_dir", path)?;

        // Load storage context
        let storage_context = llama_index
            .getattr("StorageContext")?
            .call_method1("from_defaults", (storage_dict,))?;

        // Load index
        let index = llama_index
            .getattr("load_index_from_storage")?
            .call1((storage_context.clone(),))?;

        self.0.index = Some(index.into_py(py));
        self.0.storage_context = Some(storage_context.into_py(py));

        Ok(())
    }
}

impl PyLlamaIndexBridge {
    fn source_nodes_to_py(py: Python<'_>, nodes: &[SourceNode]) -> PyResult<PyObject> {
        let list = PyList::empty(py);

        for node in nodes {
            let dict = PyDict::new(py);
            dict.set_item("file_path", node.file_path.to_string_lossy().to_string())?;
            dict.set_item("content", &node.content)?;
            dict.set_item("metadata", Self::metadata_to_py(py, &node.metadata)?)?;
            list.append(dict)?;
        }

        Ok(list.into())
    }

    fn metadata_to_py(py: Python<'_>, metadata: &NodeMetadata) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        dict.set_item("language", &metadata.language)?;
        dict.set_item("dependencies", &metadata.dependencies)?;
        dict.set_item("imports", &metadata.imports)?;
        Ok(dict.into())
    }

    fn project_context_to_py(py: Python<'_>, context: &ProjectContext) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        dict.set_item("analyzed_files", context.analyzed_files)?;

        // Convert file relationships
        let relationships = PyDict::new(py);
        for (file, deps) in &context.file_relationships {
            relationships.set_item(file, deps)?;
        }
        dict.set_item("file_relationships", relationships)?;

        // Convert dependency paths
        let paths = PyList::empty(py);
        for path in &context.dependency_paths {
            let path_dict = PyDict::new(py);
            path_dict.set_item("source", &path.source)?;
            path_dict.set_item("target", &path.target)?;
            path_dict.set_item("path", &path.path)?;
            paths.append(path_dict)?;
        }
        dict.set_item("dependency_paths", paths)?;

        Ok(dict.into())
    }
}

// Add error handling types
#[derive(Debug)]
pub enum BridgeError {
    AnalyzerError(String),
    GraphError(String),
    PyInterfaceError(String),
}

impl From<String> for BridgeError {
    fn from(s: String) -> Self {
        BridgeError::AnalyzerError(s)
    }
}

impl From<BridgeError> for PyErr {
    fn from(error: BridgeError) -> PyErr {
        match error {
            BridgeError::AnalyzerError(s) => PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(s),
            BridgeError::GraphError(s) => PyErr::new::<pyo3::exceptions::PyValueError, _>(s),
            BridgeError::PyInterfaceError(s) => {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(s)
            }
        }
    }
}
