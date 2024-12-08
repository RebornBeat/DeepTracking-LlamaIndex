mod audio_store;
mod code_store;
mod common;
mod image_store;
mod video_store;

pub use audio_store::AudioVectorStore;
pub use code_store::CodeVectorStore;
use common::{StorageConfig, StoreError, VectorIndex};
pub use image_store::ImageVectorStore;
pub use video_store::VideoVectorStore;

use crate::indexing::embeddings::{EmbeddingGenerator, EmbeddingMetadata};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;

#[async_trait]
pub trait VectorStore: Send + Sync {
    type Item;
    type Config;
    type Query;
    type Result;

    async fn add(&mut self, item: Self::Item) -> Result<(), StoreError>;
    async fn search(&self, query: Self::Query) -> Result<Vec<Self::Result>, StoreError>;
    async fn save(&self, path: PathBuf) -> Result<(), StoreError>;
    async fn load(&mut self, path: PathBuf) -> Result<(), StoreError>;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoreMetadata {
    pub item_count: usize,
    pub vector_dimension: usize,
    pub index_type: String,
    pub modality: String,
}

pub struct StoreWithEmbeddings<T: EmbeddingGenerator> {
    store: VectorIndex,
    embedding_generator: Arc<T>,
    config: StorageConfig,
    metadata: StoreMetadata,
}

impl<T: EmbeddingGenerator> StoreWithEmbeddings<T> {
    pub fn new(embedding_generator: Arc<T>, config: StorageConfig) -> Self {
        Self {
            store: VectorIndex::new(config.clone()),
            embedding_generator,
            config,
            metadata: StoreMetadata {
                item_count: 0,
                vector_dimension: embedding_generator.dimension(),
                index_type: "HNSW".to_string(),
                modality: config.modality.to_string(),
            },
        }
    }

    pub async fn add_item(&mut self, item: T::Input) -> Result<(), StoreError> {
        // Generate embedding
        let embedding = self
            .embedding_generator
            .generate(&item, &self.config.embedding_config)
            .await
            .map_err(|e| StoreError::Generation(e.to_string()))?;

        // Create metadata
        let metadata = self.create_item_metadata(&item)?;

        // Add to store
        self.store.add_vector(embedding, metadata)?;
        self.metadata.item_count += 1;

        Ok(())
    }

    fn create_item_metadata(&self, item: &T::Input) -> Result<EmbeddingMetadata, StoreError> {
        // Implementation specific to each type
        todo!()
    }
}
