use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;

#[derive(Clone)]
pub struct StorageContext {
    persister: Arc<RwLock<Persister>>,
    settings: StorageSettings,
}

#[derive(Clone)]
pub struct ServiceContext {
    chunk_size: usize,
    chunk_overlap: usize,
    settings: ServiceSettings,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct StorageSettings {
    pub index_path: PathBuf,
    pub vector_store_path: PathBuf,
    pub metadata_path: PathBuf,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct ServiceSettings {
    pub chunk_size: usize,
    pub chunk_overlap: usize,
    pub similarity_top_k: usize,
    pub embedding_dimension: usize,
}

struct Persister {
    storage_path: PathBuf,
}

impl StorageContext {
    pub fn new(storage_path: PathBuf) -> Self {
        let settings = StorageSettings {
            index_path: storage_path.join("index"),
            vector_store_path: storage_path.join("vectors"),
            metadata_path: storage_path.join("metadata"),
        };

        Self {
            persister: Arc::new(RwLock::new(Persister {
                storage_path: storage_path.clone(),
            })),
            settings,
        }
    }

    pub async fn save(&self) -> Result<(), String> {
        let persister = self.persister.read().await;
        // Save vector store state
        self.save_vectors().await?;
        // Save metadata
        self.save_metadata().await?;
        Ok(())
    }

    pub async fn load(&self) -> Result<(), String> {
        let persister = self.persister.read().await;
        // Load vector store state
        self.load_vectors().await?;
        // Load metadata
        self.load_metadata().await?;
        Ok(())
    }

    async fn save_vectors(&self) -> Result<(), String> {
        // Implementation for saving vector store
        Ok(())
    }

    async fn load_vectors(&self) -> Result<(), String> {
        // Implementation for loading vector store
        Ok(())
    }

    async fn save_metadata(&self) -> Result<(), String> {
        // Implementation for saving metadata
        Ok(())
    }

    async fn load_metadata(&self) -> Result<(), String> {
        // Implementation for loading metadata
        Ok(())
    }
}

impl ServiceContext {
    pub fn new(settings: ServiceSettings) -> Self {
        Self {
            chunk_size: settings.chunk_size,
            chunk_overlap: settings.chunk_overlap,
            settings,
        }
    }

    pub fn default() -> Self {
        Self::new(ServiceSettings {
            chunk_size: 1024,
            chunk_overlap: 128,
            similarity_top_k: 5,
            embedding_dimension: 384,
        })
    }

    pub fn with_chunk_size(mut self, chunk_size: usize) -> Self {
        self.chunk_size = chunk_size;
        self.settings.chunk_size = chunk_size;
        self
    }

    pub fn with_chunk_overlap(mut self, chunk_overlap: usize) -> Self {
        self.chunk_overlap = chunk_overlap;
        self.settings.chunk_overlap = chunk_overlap;
        self
    }

    pub fn get_chunk_size(&self) -> usize {
        self.chunk_size
    }

    pub fn get_chunk_overlap(&self) -> usize {
        self.chunk_overlap
    }

    pub fn get_similarity_top_k(&self) -> usize {
        self.settings.similarity_top_k
    }

    pub fn get_embedding_dimension(&self) -> usize {
        self.settings.embedding_dimension
    }
}
