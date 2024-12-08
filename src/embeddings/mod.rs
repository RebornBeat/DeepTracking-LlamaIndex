use serde::{Deserialize, Serialize};
use std::path::PathBuf;

pub mod audio_embed;
pub mod code_embed;
pub mod image_embed;
pub mod video_embed;

pub use audio_embed::AudioEmbedding;
pub use code_embed::CodeEmbedding;
pub use image_embed::ImageEmbedding;
pub use video_embed::VideoEmbedding;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Embedding {
    Code(CodeEmbedding),
    Image(ImageEmbedding),
    Audio(AudioEmbedding),
    Video(VideoEmbedding),
}

#[async_trait::async_trait]
pub trait EmbeddingGenerator: Send + Sync {
    type Input;
    type Config;

    async fn generate(
        &self,
        input: &Self::Input,
        config: &Self::Config,
    ) -> Result<Vec<f32>, EmbeddingError>;
    fn dimension(&self) -> usize;
}

#[derive(Debug, thiserror::Error)]
pub enum EmbeddingError {
    #[error("Generation failed: {0}")]
    GenerationFailed(String),
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    #[error("Dimension mismatch: {0}")]
    DimensionMismatch(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingMetadata {
    pub path: PathBuf,
    pub modality: String,
    pub dimension: usize,
    pub generator_version: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}
