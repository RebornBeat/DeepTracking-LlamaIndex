use crate::indexing::embeddings::Embeddings;
use crate::llm::Model;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Clone, Serialize, Deserialize)]
pub struct VectorStore {
    embeddings: HashMap<String, Vec<f32>>,
    metadata: HashMap<String, NodeMetadata>,
    model: Arc<dyn Model>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct NodeMetadata {
    pub file_path: PathBuf,
    pub language: Option<String>,
    pub content_hash: String,
    pub relationships: HashMap<String, Vec<String>>,
}

impl VectorStore {
    pub fn new(model: Arc<dyn Model>) -> Self {
        Self {
            embeddings: HashMap::new(),
            metadata: HashMap::new(),
            model,
        }
    }

    pub async fn add_text(&mut self, text: &str, metadata: NodeMetadata) -> Result<(), String> {
        let embedding = self.model.embed_text(text).await?;
        let key = metadata.file_path.to_string_lossy().to_string();

        self.embeddings.insert(key.clone(), embedding);
        self.metadata.insert(key, metadata);

        Ok(())
    }

    pub async fn search(&self, query: &str, top_k: usize) -> Result<Vec<SearchResult>, String> {
        let query_embedding = self.model.embed_text(query).await?;
        let mut results = Vec::new();

        for (key, embedding) in &self.embeddings {
            let similarity = cosine_similarity(&query_embedding, embedding);
            results.push(SearchResult {
                key: key.clone(),
                similarity,
                metadata: self.metadata.get(key).cloned(),
            });
        }

        results.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());
        Ok(results.into_iter().take(top_k).collect())
    }
}

#[derive(Clone, Debug)]
pub struct SearchResult {
    pub key: String,
    pub similarity: f32,
    pub metadata: Option<NodeMetadata>,
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}
