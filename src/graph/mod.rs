use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    pub path: PathBuf,
    pub metadata: HashMap<String, Value>,
    pub node_type: NodeType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NodeType {
    File,
    Function,
    Module,
    Type,
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
    // Index for quick lookups
    node_index: HashMap<String, HashSet<PathBuf>>,
    edge_index: HashMap<PathBuf, HashSet<Edge>>,
}

impl DependencyGraph {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: Vec::new(),
            node_index: HashMap::new(),
            edge_index: HashMap::new(),
        }
    }

    pub fn add_dependencies(&mut self, deps: Vec<Dependency>) -> Result<(), String> {
        for dep in deps {
            self.add_node(&dep.source, NodeType::File)?;
            self.add_node(&dep.target, NodeType::File)?;

            let edge = Edge {
                source: dep.source.clone(),
                target: dep.target.clone(),
                edge_type: dep.dependency_type,
                metadata: dep.metadata,
            };

            self.add_edge(edge)?;
        }
        Ok(())
    }

    pub fn get_dependencies(&self, file: &PathBuf) -> Result<Vec<String>, String> {
        let mut deps = Vec::new();

        if let Some(edges) = self.edge_index.get(file) {
            for edge in edges {
                deps.push(edge.target.to_string_lossy().into_owned());
            }
        }

        Ok(deps)
    }

    pub fn get_callers(&self, target: &PathBuf) -> Result<Vec<PathBuf>, String> {
        let mut callers = Vec::new();

        for edge in &self.edges {
            if edge.target == *target && edge.edge_type == DependencyType::FunctionCall {
                callers.push(edge.source.clone());
            }
        }

        Ok(callers)
    }

    pub fn get_function_dependencies(
        &self,
        function_name: &str,
    ) -> Result<Vec<Dependency>, String> {
        let mut deps = Vec::new();

        // Find all edges where this function is involved
        for edge in &self.edges {
            let source_name = edge
                .source
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("");

            let target_name = edge
                .target
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("");

            if source_name.contains(function_name) || target_name.contains(function_name) {
                deps.push(Dependency {
                    source: edge.source.clone(),
                    target: edge.target.clone(),
                    dependency_type: edge.edge_type.clone(),
                    metadata: edge.metadata.clone(),
                });
            }
        }

        Ok(deps)
    }

    pub fn get_llamaindex_metadata(&self, file: &PathBuf) -> Result<Value, String> {
        let mut metadata = json!({
            "file_path": file.to_string_lossy(),
            "dependencies": self.get_dependencies(file)?,
            "callers": self.get_callers(file)?,
            "relationships": {
                "direct": self.get_direct_relationships(file)?,
                "indirect": self.get_indirect_relationships(file)?,
            },
            "metrics": self.calculate_metrics(file)?,
        });

        if let Some(node) = self.nodes.get(file) {
            if let Some(node_metadata) = metadata.as_object_mut() {
                node_metadata.extend(node.metadata.iter().map(|(k, v)| (k.clone(), v.clone())));
            }
        }

        Ok(metadata)
    }

    fn get_direct_relationships(
        &self,
        file: &PathBuf,
    ) -> Result<HashMap<String, Vec<String>>, String> {
        let mut relationships = HashMap::new();

        if let Some(edges) = self.edge_index.get(file) {
            for edge in edges {
                let rel_type = match edge.edge_type {
                    DependencyType::FunctionCall => "calls",
                    DependencyType::Import => "imports",
                    DependencyType::Inheritance => "inherits",
                    DependencyType::Usage => "uses",
                    DependencyType::FunctionDefinition => "defines",
                    DependencyType::TypeUsage => "uses_type",
                };

                relationships
                    .entry(rel_type.to_string())
                    .or_insert_with(Vec::new)
                    .push(edge.target.to_string_lossy().into_owned());
            }
        }

        Ok(relationships)
    }

    fn get_indirect_relationships(
        &self,
        file: &PathBuf,
    ) -> Result<Vec<IndirectRelationship>, String> {
        let mut relationships = Vec::new();
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();

        queue.push_back((file.clone(), 0));
        visited.insert(file.clone());

        while let Some((current, depth)) = queue.pop_front() {
            if depth > 3 {
                // Limit indirect relationship depth
                continue;
            }

            if let Some(edges) = self.edge_index.get(&current) {
                for edge in edges {
                    if !visited.contains(&edge.target) {
                        visited.insert(edge.target.clone());
                        queue.push_back((edge.target.clone(), depth + 1));

                        relationships.push(IndirectRelationship {
                            path: edge.target.to_string_lossy().into_owned(),
                            relationship_type: edge.edge_type.clone(),
                            depth,
                            intermediate_nodes: self.get_path_between(file, &edge.target)?,
                        });
                    }
                }
            }
        }

        Ok(relationships)
    }

    fn calculate_metrics(&self, file: &PathBuf) -> Result<DependencyMetrics, String> {
        let mut metrics = DependencyMetrics {
            incoming_dependencies: 0,
            outgoing_dependencies: 0,
            cyclomatic_complexity: 0,
            depth_of_inheritance: 0,
            coupling_factor: 0.0,
        };

        // Calculate incoming dependencies
        metrics.incoming_dependencies = self.get_callers(file)?.len();

        // Calculate outgoing dependencies
        if let Some(edges) = self.edge_index.get(file) {
            metrics.outgoing_dependencies = edges.len();
        }

        // Calculate coupling factor
        let total_files = self.nodes.len() as f64;
        if total_files > 0.0 {
            metrics.coupling_factor = (metrics.incoming_dependencies
                + metrics.outgoing_dependencies) as f64
                / total_files;
        }

        // Calculate inheritance depth if applicable
        metrics.depth_of_inheritance = self.calculate_inheritance_depth(file)?;

        Ok(metrics)
    }

    fn calculate_inheritance_depth(&self, file: &PathBuf) -> Result<usize, String> {
        let mut depth = 0;
        let mut current = file.clone();
        let mut visited = HashSet::new();

        while let Some(parent) = self.find_parent(&current)? {
            if !visited.insert(parent.clone()) {
                return Err("Cyclic inheritance detected".to_string());
            }
            depth += 1;
            current = parent;
        }

        Ok(depth)
    }

    fn find_parent(&self, file: &PathBuf) -> Result<Option<PathBuf>, String> {
        if let Some(edges) = self.edge_index.get(file) {
            for edge in edges {
                if edge.edge_type == DependencyType::Inheritance {
                    return Ok(Some(edge.target.clone()));
                }
            }
        }
        Ok(None)
    }

    fn get_path_between(&self, start: &PathBuf, end: &PathBuf) -> Result<Vec<PathBuf>, String> {
        let mut path = Vec::new();
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        let mut parent_map = HashMap::new();

        queue.push_back(start.clone());
        visited.insert(start.clone());

        while let Some(current) = queue.pop_front() {
            if current == *end {
                // Reconstruct path
                let mut current = current;
                while current != *start {
                    path.push(current.clone());
                    current = parent_map.get(&current).unwrap().clone();
                }
                path.push(start.clone());
                path.reverse();
                return Ok(path);
            }

            if let Some(edges) = self.edge_index.get(&current) {
                for edge in edges {
                    if !visited.contains(&edge.target) {
                        visited.insert(edge.target.clone());
                        queue.push_back(edge.target.clone());
                        parent_map.insert(edge.target.clone(), current.clone());
                    }
                }
            }
        }

        Ok(Vec::new())
    }

    pub fn get_node_metadata(&self, path: &PathBuf) -> Option<&HashMap<String, Value>> {
        self.nodes.get(path).map(|node| &node.metadata)
    }

    pub fn update_node_metadata(
        &mut self,
        path: &PathBuf,
        metadata: HashMap<String, Value>,
    ) -> Result<(), String> {
        if let Some(node) = self.nodes.get_mut(path) {
            node.metadata.extend(metadata);
            Ok(())
        } else {
            Err("Node not found".to_string())
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct IndirectRelationship {
    path: String,
    relationship_type: DependencyType,
    depth: usize,
    intermediate_nodes: Vec<PathBuf>,
}

#[derive(Debug, Serialize, Deserialize)]
struct DependencyMetrics {
    incoming_dependencies: usize,
    outgoing_dependencies: usize,
    cyclomatic_complexity: usize,
    depth_of_inheritance: usize,
    coupling_factor: f64,
}
