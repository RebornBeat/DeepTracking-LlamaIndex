mod analyzers;
mod errors;
mod manager;
mod report;

pub use analyzers::{CodeAnalyzer, Dependency, DependencyMetadata, DependencyType};
pub use errors::TrackerError;
pub use manager::AnalyzerManager;
pub use report::{CombinedReport, ReportGenerator};

use crate::graph::DependencyGraph;
use std::path::PathBuf;

pub struct CodeTracker {
    pub(crate) root_path: PathBuf,
    analyzers: Vec<Box<dyn analyzers::CodeAnalyzer>>,
    graph: DependencyGraph,
}

impl CodeTracker {
    pub fn new(root_path: PathBuf) -> Result<Self, TrackerError> {
        Ok(CodeTracker {
            root_path,
            analyzers: vec![
                Box::new(analyzers::RustAnalyzer::new()),
                Box::new(analyzers::PythonAnalyzer::new()),
            ],
            graph: DependencyGraph::new(),
        })
    }

    pub fn analyze(&mut self) -> Result<(), TrackerError> {
        for analyzer in &self.analyzers {
            let results = analyzer
                .analyze(&self.root_path)
                .map_err(|e| TrackerError::Analysis(e.to_string()))?;
            self.graph
                .add_dependencies(results)
                .map_err(|e| TrackerError::Graph(e.to_string()))?;
        }
        Ok(())
    }

    pub fn get_dependencies(&self, file_path: &PathBuf) -> Result<Vec<String>, TrackerError> {
        self.graph
            .get_dependencies(file_path)
            .map_err(|e| TrackerError::Graph(e.to_string()))
    }
}
