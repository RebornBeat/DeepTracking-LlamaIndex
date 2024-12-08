use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("Storage error: {0}")]
    Storage(String),
    #[error("Index error: {0}")]
    Index(String),
    #[error("Vector operation error: {0}")]
    VectorOp(String),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    pub vector_dimension: usize,
    pub max_items: usize,
    pub index_type: IndexType,
    pub similarity_threshold: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IndexType {
    Flat,
    IVF,
    HNSW,
}

#[derive(Debug)]
pub struct VectorIndex {
    vectors: Vec<Vec<f32>>,
    metadata: HashMap<usize, IndexMetadata>,
    config: IndexConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexMetadata {
    pub id: usize,
    pub path: String,
    pub modality: String,
    pub attributes: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct IndexConfig {
    pub num_trees: usize,
    pub max_items_per_node: usize,
    pub search_k: usize,
}

impl VectorIndex {
    pub fn new(config: IndexConfig) -> Self {
        Self {
            vectors: Vec::new(),
            metadata: HashMap::new(),
            config,
        }
    }

    pub fn add(&mut self, vector: Vec<f32>, metadata: IndexMetadata) -> Result<(), StoreError> {
        let id = self.vectors.len();
        self.vectors.push(vector);
        self.metadata.insert(id, metadata);
        Ok(())
    }

    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(usize, f32)>, StoreError> {
        let mut scores: Vec<(usize, f32)> = self
            .vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (i, cosine_similarity(query, v)))
            .collect();

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        Ok(scores.into_iter().take(k).collect())
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}
