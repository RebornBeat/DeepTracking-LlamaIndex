use super::{StorageConfig, StoreError, VectorIndex, VectorStore};
use crate::indexing::llm::EnhancedCodeAnalysis;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug)]
pub struct CodeVectorStore {
    index: VectorIndex,
    config: StorageConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeQuery {
    pub text: String,
    pub filters: CodeQueryFilters,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeQueryFilters {
    pub languages: Option<Vec<String>>,
    pub file_types: Option<Vec<String>>,
    pub relationship_types: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeSearchResult {
    pub analysis: EnhancedCodeAnalysis,
    pub similarity: f32,
    pub path: PathBuf,
}

impl CodeVectorStore {
    pub fn new(config: StorageConfig) -> Self {
        Self {
            index: VectorIndex::new(IndexConfig {
                num_trees: 10,
                max_items_per_node: 100,
                search_k: 50,
            }),
            config,
        }
    }

    fn create_vector(&self, analysis: &EnhancedCodeAnalysis) -> Result<Vec<f32>, StoreError> {
        let mut vector = Vec::with_capacity(self.config.vector_dimension);

        // 1. Pattern-based features (30% of vector)
        let pattern_features = self.encode_patterns(&analysis.patterns)?;
        vector.extend(pattern_features);

        // 2. Architectural features (25% of vector)
        let arch_features = self.encode_architecture(&analysis.architecture)?;
        vector.extend(arch_features);

        // 3. Relationship features (25% of vector)
        let rel_features = self.encode_relationships(&analysis.relationships)?;
        vector.extend(rel_features);

        // 4. Context features (20% of vector)
        let context_features = self.encode_context(&analysis.context)?;
        vector.extend(context_features);

        // Normalize the vector
        self.normalize_vector(&mut vector)?;

        Ok(vector)
    }

    fn encode_patterns(&self, patterns: &[CodePattern]) -> Result<Vec<f32>, StoreError> {
        let mut features = Vec::new();

        // Encode pattern types and their strengths
        let pattern_type_counts = patterns.iter().fold(HashMap::new(), |mut acc, p| {
            *acc.entry(&p.pattern_type).or_insert(0.0) += p.confidence;
            acc
        });

        // Encode pattern distribution
        features.extend(self.encode_distribution(pattern_type_counts, 10)?);

        // Encode pattern complexity
        features.push(patterns.len() as f32 / 100.0); // Normalized by max expected patterns

        // Encode pattern relationships
        let relationship_strength = patterns
            .iter()
            .map(|p| p.instances.len() as f32 * p.confidence)
            .sum::<f32>()
            / patterns.len() as f32;
        features.push(relationship_strength);

        Ok(features)
    }

    fn encode_architecture(&self, arch: &ArchitecturalInsights) -> Result<Vec<f32>, StoreError> {
        let mut features = Vec::new();

        // Encode design patterns
        features.extend(self.encode_design_patterns(&arch.design_patterns)?);

        // Encode component roles
        features.extend(self.encode_component_roles(&arch.component_roles)?);

        // Encode architectural style
        features.extend(self.encode_architectural_style(&arch.architectural_style)?);

        // Encode quality attributes
        features.extend(self.encode_quality_attributes(&arch.quality_attributes)?);

        Ok(features)
    }

    fn normalize_vector(&self, vector: &mut Vec<f32>) -> Result<(), StoreError> {
        let magnitude = (vector.iter().map(|x| x * x).sum::<f32>()).sqrt();
        if magnitude > 0.0 {
            for x in vector.iter_mut() {
                *x /= magnitude;
            }
        }
        Ok(())
    }

    fn apply_filters(
        &self,
        results: Vec<(usize, f32)>,
        filters: &CodeQueryFilters,
    ) -> Result<Vec<FilteredResult>, StoreError> {
        let mut filtered = Vec::new();

        for (idx, similarity) in results {
            let metadata = self
                .index
                .metadata
                .get(&idx)
                .ok_or(StoreError::Storage("Missing metadata".into()))?;

            if self.matches_filters(metadata, filters)? {
                filtered.push(FilteredResult {
                    index: idx,
                    similarity,
                    metadata: metadata.clone(),
                });
            }
        }

        Ok(filtered)
    }

    fn matches_filters(
        &self,
        metadata: &IndexMetadata,
        filters: &CodeQueryFilters,
    ) -> Result<bool, StoreError> {
        // Check language filter
        if let Some(langs) = &filters.languages {
            if !langs.contains(&metadata.attributes["language"]) {
                return Ok(false);
            }
        }

        // Check file type filter
        if let Some(types) = &filters.file_types {
            if !types.contains(&metadata.attributes["file_type"]) {
                return Ok(false);
            }
        }

        // Check relationship types
        if let Some(rel_types) = &filters.relationship_types {
            let has_matching_relationship = metadata
                .attributes
                .get("relationships")
                .map(|rels| rel_types.iter().any(|rt| rels.contains(rt)))
                .unwrap_or(false);

            if !has_matching_relationship {
                return Ok(false);
            }
        }

        Ok(true)
    }

    fn convert_to_search_results(
        &self,
        filtered: Vec<FilteredResult>,
    ) -> Result<Vec<CodeSearchResult>, StoreError> {
        let mut results = Vec::new();

        for result in filtered {
            let analysis = self.load_analysis(&result.metadata.path)?;

            results.push(CodeSearchResult {
                analysis,
                similarity: result.similarity,
                path: PathBuf::from(&result.metadata.path),
                context: self.build_code_context(&result)?,
            });
        }

        Ok(results)
    }

    fn build_code_context(&self, result: &FilteredResult) -> Result<CodeContext, StoreError> {
        // Build comprehensive code context including:
        // - Surrounding code
        // - Related functions
        // - Dependencies
        // - Usage patterns
        todo!()
    }

    fn create_metadata(
        &self,
        analysis: &EnhancedCodeAnalysis,
    ) -> Result<IndexMetadata, StoreError> {
        let mut attributes = HashMap::new();
        attributes.insert(
            "language".to_string(),
            analysis.base.functions[0].language.clone(),
        );
        attributes.insert(
            "pattern_count".to_string(),
            analysis.patterns.len().to_string(),
        );
        attributes.insert(
            "architectural_style".to_string(),
            analysis.architecture.architectural_style[0].to_string(),
        );
        attributes.insert(
            "complexity_score".to_string(),
            analysis.base.metrics.complexity.to_string(),
        );

        Ok(IndexMetadata {
            id: 0, // Will be set by index
            path: analysis.base.functions[0]
                .location
                .file
                .to_string_lossy()
                .into_owned(),
            modality: "code".to_string(),
            attributes,
        })
    }
}

#[async_trait]
impl VectorStore for CodeVectorStore {
    type Item = EnhancedCodeAnalysis;
    type Config = StorageConfig;
    type Query = CodeQuery;
    type Result = CodeSearchResult;

    async fn add(&mut self, item: Self::Item) -> Result<(), StoreError> {
        let vector = self.create_vector(&item)?;
        let metadata = self.create_metadata(&item)?;
        self.index.add(vector, metadata)
    }

    async fn search(&self, query: Self::Query) -> Result<Vec<Self::Result>, StoreError> {
        // Create query vector
        let query_vector = match query.text {
            Some(text) => self.create_query_vector(&text)?,
            None => return Err(StoreError::InvalidQuery("No query text provided".into())),
        };

        // Perform similarity search
        let results = self.index.search(&query_vector, self.config.max_results)?;

        // Apply filters
        let filtered_results = self.apply_filters(results, &query.filters)?;

        // Convert to search results
        let search_results = self.convert_to_search_results(filtered_results)?;

        Ok(search_results)
    }

    async fn save(&self, path: PathBuf) -> Result<(), StoreError> {
        let store_data = StoreData {
            vectors: self.index.vectors.clone(),
            metadata: self.index.metadata.clone(),
            config: self.config.clone(),
        };

        let file = File::create(path)?;
        serde_json::to_writer(file, &store_data).map_err(|e| StoreError::Storage(e.to_string()))?;

        Ok(())
    }

    async fn load(&mut self, path: PathBuf) -> Result<(), StoreError> {
        let file = File::open(path)?;
        let store_data: StoreData =
            serde_json::from_reader(file).map_err(|e| StoreError::Storage(e.to_string()))?;

        self.index.vectors = store_data.vectors;
        self.index.metadata = store_data.metadata;
        self.config = store_data.config;

        Ok(())
    }
}
