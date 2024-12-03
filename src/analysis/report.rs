use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Serialize, Deserialize)]
pub struct CombinedReport {
    pub project_structure: ProjectStructure,
    pub llama_index_response: LlamaIndexResponse,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ProjectStructure {
    pub root: PathBuf,
    pub entries: Vec<FileEntry>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct FileEntry {
    pub path: PathBuf,
    pub entry_type: EntryType,
    pub children: Vec<FileEntry>,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum EntryType {
    Directory,
    File { extension: Option<String> },
}

#[derive(Debug, Serialize, Deserialize)]
pub struct LlamaIndexResponse {
    pub response: String,
    pub source_nodes: Vec<SourceNode>,
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

pub struct ReportGenerator {
    root_path: PathBuf,
}

impl ReportGenerator {
    pub fn new(root_path: PathBuf) -> Self {
        Self { root_path }
    }

    pub fn generate_report(&self, llama_response: PyObject) -> Result<CombinedReport, String> {
        let project_structure = self.build_project_structure()?;
        let llama_index_response = self.process_llama_response(llama_response)?;

        Ok(CombinedReport {
            project_structure,
            llama_index_response,
        })
    }

    fn build_project_structure(&self) -> Result<ProjectStructure, String> {
        let mut entries = Vec::new();
        self.build_directory_structure(&self.root_path, &mut entries)?;

        Ok(ProjectStructure {
            root: self.root_path.clone(),
            entries,
        })
    }

    fn build_directory_structure(
        &self,
        dir: &Path,
        entries: &mut Vec<FileEntry>,
    ) -> Result<(), String> {
        for entry in fs::read_dir(dir).map_err(|e| e.to_string())? {
            let entry = entry.map_err(|e| e.to_string())?;
            let path = entry.path();

            // Skip ignored directories/files
            if self.should_ignore(&path) {
                continue;
            }

            let mut file_entry = if path.is_dir() {
                FileEntry {
                    path: path
                        .strip_prefix(&self.root_path)
                        .map_err(|e| e.to_string())?
                        .to_path_buf(),
                    entry_type: EntryType::Directory,
                    children: Vec::new(),
                }
            } else {
                FileEntry {
                    path: path
                        .strip_prefix(&self.root_path)
                        .map_err(|e| e.to_string())?
                        .to_path_buf(),
                    entry_type: EntryType::File {
                        extension: path
                            .extension()
                            .and_then(|ext| ext.to_str())
                            .map(String::from),
                    },
                    children: Vec::new(),
                }
            };

            if path.is_dir() {
                self.build_directory_structure(&path, &mut file_entry.children)?;
            }

            entries.push(file_entry);
        }

        // Sort entries (directories first, then files alphabetically)
        entries.sort_by(|a, b| match (&a.entry_type, &b.entry_type) {
            (EntryType::Directory, EntryType::File { .. }) => std::cmp::Ordering::Less,
            (EntryType::File { .. }, EntryType::Directory) => std::cmp::Ordering::Greater,
            _ => a.path.cmp(&b.path),
        });

        Ok(())
    }

    fn should_ignore(&self, path: &Path) -> bool {
        let ignored = [
            ".git",
            "target",
            "node_modules",
            "__pycache__",
            ".deeptracking-cache",
        ];

        path.components().any(|c| {
            c.as_os_str()
                .to_str()
                .map(|s| ignored.contains(&s))
                .unwrap_or(false)
        })
    }

    fn process_llama_response(&self, response: PyObject) -> Result<LlamaIndexResponse, String> {
        Python::with_gil(|py| {
            // Extract response text and source nodes from LlamaIndex response
            let response_text = response
                .getattr(py, "response")?
                .extract::<String>(py)
                .map_err(|e| e.to_string())?;

            let source_nodes = response
                .getattr(py, "source_nodes")?
                .extract::<Vec<PyObject>>(py)
                .map_err(|e| e.to_string())?;

            let mut nodes = Vec::new();
            for node in source_nodes {
                let metadata = node
                    .getattr(py, "metadata")?
                    .extract::<HashMap<String, PyObject>>(py)
                    .map_err(|e| e.to_string())?;

                nodes.push(SourceNode {
                    file_path: PathBuf::from(
                        metadata
                            .get("file_path")
                            .ok_or("Missing file_path")?
                            .extract::<String>(py)
                            .map_err(|e| e.to_string())?,
                    ),
                    content: node
                        .getattr(py, "text")?
                        .extract::<String>(py)
                        .map_err(|e| e.to_string())?,
                    metadata: NodeMetadata {
                        language: metadata
                            .get("language")
                            .map(|l| l.extract::<String>(py))
                            .transpose()
                            .map_err(|e| e.to_string())?,
                        dependencies: metadata
                            .get("dependencies")
                            .map(|d| d.extract::<Vec<String>>(py))
                            .transpose()
                            .map_err(|e| e.to_string())?
                            .unwrap_or_default(),
                        imports: metadata
                            .get("imports")
                            .map(|i| i.extract::<Vec<String>>(py))
                            .transpose()
                            .map_err(|e| e.to_string())?
                            .unwrap_or_default(),
                    },
                });
            }

            Ok(LlamaIndexResponse {
                response: response_text,
                source_nodes: nodes,
            })
        })
        .map_err(|e| e.to_string())
    }
}
