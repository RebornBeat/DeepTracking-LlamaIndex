pub mod manager;
mod python;
mod rust;

pub use python::PythonAnalyzer;
pub use rust::RustAnalyzer;

use serde::{Deserialize, Serialize};
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dependency {
    pub source: PathBuf,
    pub target: PathBuf,
    pub dependency_type: DependencyType,
    pub metadata: DependencyMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize, Hash, Eq, PartialEq)]
pub enum DependencyType {
    Import,
    FunctionCall,
    Inheritance,
    Usage,
    FunctionDefinition,
    TypeUsage,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, Eq, PartialEq)]
pub struct DependencyMetadata {
    pub line_number: Option<usize>,
    pub description: Option<String>,
    pub relationships: Option<Vec<String>>,
    pub context: Option<serde_json::Value>,
}

impl Hash for DependencyMetadata {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.line_number.hash(state);
        self.description.hash(state);
        self.relationships.hash(state);
    }
}

pub trait CodeAnalyzer: Send + Sync + std::fmt::Debug {
    fn analyze(&self, path: &Path) -> Result<Vec<Dependency>, String>;
    fn supported_extensions(&self) -> Vec<&'static str>;
}
