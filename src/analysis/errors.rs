use thiserror::Error;

#[derive(Error, Debug)]
pub enum TrackerError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Analysis error: {0}")]
    Analysis(String),
    #[error("Graph error: {0}")]
    Graph(String),
}
