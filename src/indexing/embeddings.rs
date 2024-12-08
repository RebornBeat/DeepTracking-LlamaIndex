use crate::llm::Model;
use dashmap::DashMap;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokenizers::Tokenizer;
use tokio::sync::RwLock;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EmbeddingVector(Vec<f32>);

#[derive(Clone)]
pub struct Embeddings {
    model: Arc<dyn Model>,
    cache: Arc<DashMap<String, EmbeddingVector>>,
    tokenizer: Arc<Tokenizer>,
    config: EmbeddingConfig,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    pub dimension: usize,
    pub chunk_size: usize,
    pub chunk_overlap: usize,
    pub cache_capacity: usize,
    pub batch_size: usize,
}

#[derive(Clone, Debug)]
pub struct TextChunk {
    pub content: String,
    pub metadata: ChunkMetadata,
}

#[derive(Clone, Debug)]
pub struct ChunkMetadata {
    pub start_idx: usize,
    pub end_idx: usize,
    pub source_file: String,
    pub language: Option<String>,
}

impl Embeddings {
    pub fn new(model: Arc<dyn Model>, tokenizer: Arc<Tokenizer>, config: EmbeddingConfig) -> Self {
        Self {
            model,
            cache: Arc::new(DashMap::with_capacity(config.cache_capacity)),
            tokenizer,
            config,
        }
    }

    pub async fn embed_text(&self, text: &str) -> Result<EmbeddingVector, String> {
        // Check cache first
        if let Some(cached) = self.cache.get(text) {
            return Ok(cached.clone());
        }

        // Generate embedding
        let embedding = self.model.embed_text(text).await?;
        let embedding_vector = EmbeddingVector(embedding);

        // Cache the result
        self.cache
            .insert(text.to_string(), embedding_vector.clone());

        Ok(embedding_vector)
    }

    pub async fn embed_chunks(
        &self,
        chunks: Vec<TextChunk>,
    ) -> Result<Vec<(TextChunk, EmbeddingVector)>, String> {
        let mut results = Vec::with_capacity(chunks.len());

        // Process chunks in batches
        for chunk_batch in chunks.chunks(self.config.batch_size) {
            let mut batch_results = Vec::with_capacity(chunk_batch.len());

            for chunk in chunk_batch {
                let embedding = self.embed_text(&chunk.content).await?;
                batch_results.push((chunk.clone(), embedding));
            }

            results.extend(batch_results);
        }

        Ok(results)
    }

    pub fn chunk_text(&self, text: &str, metadata: ChunkMetadata) -> Vec<TextChunk> {
        let chunks = self.create_overlapping_chunks(text);

        chunks
            .into_iter()
            .enumerate()
            .map(|(i, content)| {
                let start_idx = i * (self.config.chunk_size - self.config.chunk_overlap);
                let end_idx = start_idx + content.len();

                TextChunk {
                    content,
                    metadata: ChunkMetadata {
                        start_idx,
                        end_idx,
                        source_file: metadata.source_file.clone(),
                        language: metadata.language.clone(),
                    },
                }
            })
            .collect()
    }

    fn create_overlapping_chunks(&self, text: &str) -> Vec<String> {
        let tokens = self.tokenizer.encode(text, false).unwrap();
        let token_ids = tokens.get_ids();

        if token_ids.len() <= self.config.chunk_size {
            return vec![text.to_string()];
        }

        let mut chunks = Vec::new();
        let mut start = 0;

        while start < token_ids.len() {
            let end = (start + self.config.chunk_size).min(token_ids.len());
            let chunk_tokens = &token_ids[start..end];

            let chunk_text = self
                .tokenizer
                .decode(chunk_tokens, true)
                .unwrap_or_default();

            chunks.push(chunk_text);

            start += self.config.chunk_size - self.config.chunk_overlap;
        }

        chunks
    }
}

// Similarity calculations
impl EmbeddingVector {
    pub fn cosine_similarity(&self, other: &EmbeddingVector) -> f32 {
        let dot_product: f32 = self.0.iter().zip(other.0.iter()).map(|(a, b)| a * b).sum();

        let norm_a: f32 = self.0.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = other.0.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot_product / (norm_a * norm_b)
        }
    }

    pub fn euclidean_distance(&self, other: &EmbeddingVector) -> f32 {
        self.0
            .iter()
            .zip(other.0.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt()
    }
}

// Cache management
#[derive(Default)]
pub struct EmbeddingCache {
    cache: Arc<DashMap<String, CacheEntry>>,
    config: CacheConfig,
}

#[derive(Clone, Debug)]
struct CacheEntry {
    vector: EmbeddingVector,
    last_accessed: std::time::SystemTime,
    access_count: u32,
}

#[derive(Clone, Debug)]
struct CacheConfig {
    max_size: usize,
    ttl: std::time::Duration,
    cleanup_interval: std::time::Duration,
}

impl EmbeddingCache {
    pub fn new(config: CacheConfig) -> Self {
        let cache = Self {
            cache: Arc::new(DashMap::new()),
            config,
        };

        // Start background cleanup task
        cache.start_cleanup_task();
        cache
    }

    fn start_cleanup_task(&self) {
        let cache = self.cache.clone();
        let config = self.config.clone();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(config.cleanup_interval);
            loop {
                interval.tick().await;
                Self::cleanup_expired_entries(&cache, &config);
            }
        });
    }

    fn cleanup_expired_entries(cache: &DashMap<String, CacheEntry>, config: &CacheConfig) {
        let now = std::time::SystemTime::now();
        cache.retain(|_, entry| {
            now.duration_since(entry.last_accessed)
                .map(|age| age < config.ttl)
                .unwrap_or(false)
        });
    }

    pub fn get(&self, key: &str) -> Option<EmbeddingVector> {
        self.cache.get_mut(key).map(|mut entry| {
            entry.last_accessed = std::time::SystemTime::now();
            entry.access_count += 1;
            entry.vector.clone()
        })
    }

    pub fn insert(&self, key: String, vector: EmbeddingVector) {
        if self.cache.len() >= self.config.max_size {
            self.evict_least_used();
        }

        self.cache.insert(
            key,
            CacheEntry {
                vector,
                last_accessed: std::time::SystemTime::now(),
                access_count: 1,
            },
        );
    }

    fn evict_least_used(&self) {
        if let Some((key, _)) = self
            .cache
            .iter()
            .min_by_key(|entry| entry.value().access_count)
        {
            self.cache.remove(&key);
        }
    }
}
