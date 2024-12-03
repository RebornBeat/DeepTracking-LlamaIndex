use crate::analysis::analyzers::{Dependency, DependencyMetadata, DependencyType};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    pub path: PathBuf,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    pub source: PathBuf,
    pub target: PathBuf,
    pub edge_type: DependencyType,
    pub metadata: DependencyMetadata,
}

pub struct DependencyGraph {
    nodes: HashMap<PathBuf, Node>,
    edges: Vec<Edge>,
}

impl DependencyGraph {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: Vec::new(),
        }
    }

    pub fn add_dependencies(&mut self, deps: Vec<Dependency>) -> Result<(), String> {
        for dep in deps {
            self.add_node(&dep.source)?;
            self.add_node(&dep.target)?;
            self.add_edge(Edge {
                source: dep.source,
                target: dep.target,
                edge_type: dep.dependency_type,
                metadata: dep.metadata,
            })?;
        }
        Ok(())
    }

    pub fn get_dependencies(&self, file: &PathBuf) -> Result<Vec<String>, String> {
        let mut deps = Vec::new();
        if let Some(node) = self.nodes.get(file) {
            for edge in &self.edges {
                if edge.source == node.path {
                    deps.push(edge.target.to_string_lossy().into_owned());
                }
            }
        }
        Ok(deps)
    }

    fn add_node(&mut self, path: &PathBuf) -> Result<(), String> {
        if !self.nodes.contains_key(path) {
            self.nodes.insert(
                path.clone(),
                Node {
                    path: path.clone(),
                    metadata: HashMap::new(),
                },
            );
        }
        Ok(())
    }

    fn add_edge(&mut self, edge: Edge) -> Result<(), String> {
        self.edges.push(edge);
        Ok(())
    }
}
