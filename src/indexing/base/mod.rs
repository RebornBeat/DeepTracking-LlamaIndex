use serde::{Deserialize, Serialize};
use std::path::PathBuf;

pub mod audio;
pub mod code;
pub mod image;
pub mod video;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BaseAnalysis {
    Code(CodeAnalysis),
    Image(ImageAnalysis),
    Audio(AudioAnalysis),
    Video(VideoAnalysis),
}

pub trait BaseAnalyzer {
    type Config;
    type Output;

    fn analyze(&self, content: &[u8], config: &Self::Config)
        -> Result<Self::Output, AnalysisError>;
}

#[derive(Debug, thiserror::Error)]
pub enum AnalysisError {
    #[error("Invalid content format: {0}")]
    InvalidFormat(String),
    #[error("Analysis failed: {0}")]
    AnalysisFailed(String),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}
