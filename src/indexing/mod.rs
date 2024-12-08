mod base;
mod common;
mod embeddings;
mod llm;
mod store;

use crate::analyzers::{PythonAnalyzer, RustAnalyzer};
use common::{IndexConfig, Metadata, Relationship};

pub enum ModalityType {
    Code,
    Image,
    Audio,
    Video,
}

pub struct ModalityConfig {
    modality: ModalityType,
    base_config: BaseConfig,
    llm_config: LLMConfig,
    store_config: StoreConfig,
}

pub struct Indexer {
    configs: HashMap<ModalityType, ModalityConfig>,
    stores: HashMap<ModalityType, Box<dyn Store>>,
    analyzers: HashMap<ModalityType, Box<dyn Analyzer>>,
    llm_enhancers: HashMap<ModalityType, Box<dyn LLMEnhancer>>,
}

impl Indexer {
    pub fn new() -> Self {
        let mut indexer = Self {
            configs: HashMap::new(),
            stores: HashMap::new(),
            analyzers: HashMap::new(),
            llm_enhancers: HashMap::new(),
        };

        // Initialize code modality
        indexer.setup_code_modality();
        indexer.setup_image_modality();
        indexer.setup_audio_modality();
        indexer.setup_video_modality();

        indexer
    }

    fn setup_code_modality(&mut self) {
        // Keep existing code analyzers and add LLM enhancement
        let code_analyzers: Vec<Box<dyn CodeAnalyzer>> = vec![
            Box::new(RustAnalyzer::new()),
            Box::new(PythonAnalyzer::new()),
        ];

        let code_config = ModalityConfig {
            modality: ModalityType::Code,
            base_config: BaseConfig::code_default(),
            llm_config: LLMConfig::code_default(),
            store_config: StoreConfig::code_default(),
        };

        self.configs.insert(ModalityType::Code, code_config);
        self.stores
            .insert(ModalityType::Code, Box::new(CodeStore::new(code_analyzers)));
    }

    pub async fn index_content(
        &mut self,
        content: &[u8],
        modality: ModalityType,
        metadata: Metadata,
    ) -> Result<IndexingResult, IndexError> {
        let config = self
            .configs
            .get(&modality)
            .ok_or(IndexError::UnsupportedModality)?;

        // 1. Base Analysis
        let base_analysis = self.analyze_content(content, &modality).await?;

        // 2. LLM Enhancement
        let enhanced_analysis = self.enhance_analysis(base_analysis, &modality).await?;

        // 3. Generate Embeddings
        let embeddings = self
            .generate_embeddings(enhanced_analysis, &modality)
            .await?;

        // 4. Store Results
        self.store_results(embeddings, metadata, &modality).await
    }

    async fn analyze_content(
        &self,
        content: &[u8],
        modality: &ModalityType,
    ) -> Result<BaseAnalysis, IndexError> {
        match modality {
            ModalityType::Code => self.analyze_code(content).await,
            ModalityType::Image => self.analyze_image(content).await,
            ModalityType::Audio => self.analyze_audio(content).await,
            ModalityType::Video => self.analyze_video(content).await,
        }
    }

    async fn enhance_analysis(
        &self,
        analysis: BaseAnalysis,
        modality: &ModalityType,
    ) -> Result<EnhancedAnalysis, IndexError> {
        let enhancer = self
            .llm_enhancers
            .get(modality)
            .ok_or(IndexError::UnsupportedModality)?;

        enhancer.enhance(analysis).await
    }
}
