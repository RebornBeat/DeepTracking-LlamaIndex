use super::{EmbeddingError, EmbeddingGenerator, EmbeddingMetadata};
use crate::analyzers::manager::CodeAnalysis;
use crate::llm::Model;
use async_trait::async_trait;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct CodeEmbedding {
    pub vector: Vec<f32>,
    pub metadata: EmbeddingMetadata,
    pub features: CodeFeatures,
}

#[derive(Debug, Clone)]
pub struct CodeFeatures {
    pub syntactic_features: Vec<f32>,
    pub semantic_features: Vec<f32>,
    pub structural_features: Vec<f32>,
    pub dependency_features: Vec<f32>,
}

pub struct CodeEmbeddingGenerator {
    llm: Arc<dyn Model>,
    config: CodeEmbeddingConfig,
}

#[derive(Debug, Clone)]
pub struct CodeEmbeddingConfig {
    pub dimension: usize,
    pub feature_weights: FeatureWeights,
    pub context_window: usize,
}

#[derive(Debug, Clone)]
pub struct FeatureWeights {
    pub syntactic: f32,
    pub semantic: f32,
    pub structural: f32,
    pub dependency: f32,
}

impl CodeEmbeddingGenerator {
    pub fn new(llm: Arc<dyn Model>, config: CodeEmbeddingConfig) -> Self {
        Self { llm, config }
    }

    async fn generate_syntactic_features(
        &self,
        analysis: &CodeAnalysis,
    ) -> Result<Vec<f32>, EmbeddingError> {
        // Generate features based on code structure
        let tokens = analysis.get_tokens()?;
        let ast = analysis.get_ast()?;

        // Extract features from AST and tokens
        let mut features = Vec::new();
        features.extend(self.extract_token_features(&tokens)?);
        features.extend(self.extract_ast_features(&ast)?);

        Ok(features)
    }

    async fn generate_semantic_features(
        &self,
        analysis: &CodeAnalysis,
    ) -> Result<Vec<f32>, EmbeddingError> {
        // Use LLM to understand code semantics
        let context = self.prepare_semantic_context(analysis)?;
        let embedding = self.llm.embed_text(&context).await?;

        Ok(embedding)
    }

    async fn generate_structural_features(
        &self,
        analysis: &CodeAnalysis,
    ) -> Result<Vec<f32>, EmbeddingError> {
        // Analyze code structure
        let structure = analysis.get_structure()?;

        // Extract structural patterns
        let mut features = Vec::new();
        features.extend(self.analyze_dependencies(&structure)?);
        features.extend(self.analyze_hierarchy(&structure)?);

        Ok(features)
    }

    async fn generate_dependency_features(
        &self,
        analysis: &CodeAnalysis,
    ) -> Result<Vec<f32>, EmbeddingError> {
        // Analyze dependencies
        let deps = analysis.get_dependencies()?;

        // Convert dependencies to features
        let mut features = Vec::new();
        features.extend(self.analyze_import_patterns(&deps)?);
        features.extend(self.analyze_call_graphs(&deps)?);

        Ok(features)
    }
}

#[async_trait]
impl EmbeddingGenerator for CodeEmbeddingGenerator {
    type Input = CodeAnalysis;
    type Config = CodeEmbeddingConfig;

    async fn generate(
        &self,
        input: &Self::Input,
        config: &Self::Config,
    ) -> Result<Vec<f32>, EmbeddingError> {
        // Generate individual feature vectors
        let syntactic = self.generate_syntactic_features(input).await?;
        let semantic = self.generate_semantic_features(input).await?;
        let structural = self.generate_structural_features(input).await?;
        let dependency = self.generate_dependency_features(input).await?;

        // Combine features using weights
        let mut combined = Vec::with_capacity(config.dimension);
        combined.extend(self.weighted_combine(
            vec![
                (&syntactic, config.feature_weights.syntactic),
                (&semantic, config.feature_weights.semantic),
                (&structural, config.feature_weights.structural),
                (&dependency, config.feature_weights.dependency),
            ],
            config.dimension,
        )?);

        Ok(combined)
    }

    fn dimension(&self) -> usize {
        self.config.dimension
    }
}
