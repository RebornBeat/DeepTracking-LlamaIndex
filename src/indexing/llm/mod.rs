use crate::llm::Model;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

pub mod audio_enhancer;
pub mod code_enhancer;
pub mod image_enhancer;
pub mod video_enhancer;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EnhancedAnalysis {
    Code(EnhancedCodeAnalysis),
    Image(EnhancedImageAnalysis),
    Audio(EnhancedAudioAnalysis),
    Video(EnhancedVideoAnalysis),
}

#[async_trait::async_trait]
pub trait LLMEnhancer {
    type Input;
    type Output;
    type Config;

    async fn enhance(
        &self,
        base_analysis: Self::Input,
        config: &Self::Config,
    ) -> Result<Self::Output, EnhancementError>;
}

#[derive(Debug, thiserror::Error)]
pub enum EnhancementError {
    #[error("LLM processing failed: {0}")]
    LLMError(String),
    #[error("Invalid analysis: {0}")]
    InvalidAnalysis(String),
    #[error("Context error: {0}")]
    ContextError(String),
}
