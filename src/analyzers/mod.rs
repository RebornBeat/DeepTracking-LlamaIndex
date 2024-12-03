mod python;
mod rust;

pub use python::PythonAnalyzer;
pub use rust::RustAnalyzer;

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dependency {
    pub source: PathBuf,
    pub target: PathBuf,
    pub dependency_type: DependencyType,
    pub metadata: DependencyMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DependencyType {
    Import,
    FunctionCall,
    Inheritance,
    Usage,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DependencyMetadata {
    pub line_number: Option<usize>,
    pub description: Option<String>,
}

pub trait CodeAnalyzer {
    fn analyze(&self, path: &Path) -> Result<Vec<Dependency>, String>;
    fn supported_extensions(&self) -> Vec<&'static str>;
}
