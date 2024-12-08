use super::{StorageConfig, StoreError, VectorIndex, VectorStore};
use crate::indexing::llm::EnhancedAudioAnalysis;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug)]
pub struct AudioVectorStore {
    index: VectorIndex,
    config: StorageConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioQuery {
    pub audio_features: Option<Vec<f32>>,
    pub semantic_description: Option<String>,
    pub filters: AudioQueryFilters,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioQueryFilters {
    pub content_types: Option<Vec<AudioContentType>>,
    pub duration_range: Option<(f32, f32)>,
    pub tempo_range: Option<(f32, f32)>,
    pub emotional_qualities: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AudioContentType {
    Music,
    Speech,
    Ambient,
    Effects,
    Mixed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioSearchResult {
    pub analysis: EnhancedAudioAnalysis,
    pub similarity: f32,
    pub path: PathBuf,
    pub segment_info: Option<AudioSegmentInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioSegmentInfo {
    pub start_time: f32,
    pub end_time: f32,
    pub confidence: f32,
    pub context: AudioContext,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioContext {
    pub surrounding_elements: Vec<AudioElement>,
    pub temporal_position: String,
    pub structural_role: String,
}

impl AudioVectorStore {
    pub fn new(config: StorageConfig) -> Self {
        Self {
            index: VectorIndex::new(IndexConfig {
                num_trees: 12, // Optimized for audio feature space
                max_items_per_node: 150,
                search_k: 100,
            }),
            config,
        }
    }

    fn create_vector(&self, analysis: &EnhancedAudioAnalysis) -> Result<Vec<f32>, StoreError> {
        let mut vector = Vec::with_capacity(self.config.vector_dimension);

        // 1. Spectral features (35% of vector)
        let spectral_features = self.encode_spectral_features(&analysis.base.spectral)?;
        vector.extend(spectral_features);

        // 2. Temporal features (25% of vector)
        let temporal_features = self.encode_temporal_features(&analysis.base.waveform)?;
        vector.extend(temporal_features);

        // 3. Content features (25% of vector)
        let content_features = self.encode_content_features(&analysis.content_interpretation)?;
        vector.extend(content_features);

        // 4. Emotional features (15% of vector)
        let emotional_features = self.encode_emotional_features(&analysis.emotional_analysis)?;
        vector.extend(emotional_features);

        // Normalize the vector
        self.normalize_vector(&mut vector)?;

        Ok(vector)
    }

    fn encode_spectral_features(
        &self,
        spectral: &SpectralAnalysis,
    ) -> Result<Vec<f32>, StoreError> {
        let mut features = Vec::new();

        // Encode frequency spectrum (compressed)
        let spectrum_features = self.compress_spectrum(&spectral.frequency_spectrum)?;
        features.extend(spectrum_features);

        // Encode MFCCs
        features.extend(&spectral.mfcc[..13]); // First 13 coefficients

        // Encode pitch features
        let pitch_features = self.encode_pitch_features(&spectral.pitch_contour)?;
        features.extend(pitch_features);

        Ok(features)
    }

    fn compress_spectrum(&self, spectrum: &[Vec<f32>]) -> Result<Vec<f32>, StoreError> {
        // Compress spectrum using log-scale averaging
        let mut compressed = Vec::new();
        let bands = self.calculate_frequency_bands(spectrum)?;

        for band in bands {
            let avg = band.iter().sum::<f32>() / band.len() as f32;
            compressed.push(avg);
        }

        Ok(compressed)
    }

    fn encode_temporal_features(
        &self,
        waveform: &WaveformAnalysis,
    ) -> Result<Vec<f32>, StoreError> {
        let mut features = Vec::new();

        // Encode amplitude envelope
        let env_features = self.encode_amplitude_envelope(&waveform.amplitude_envelope)?;
        features.extend(env_features);

        // Encode rhythm features
        let rhythm_features = self.encode_rhythm_features(waveform)?;
        features.extend(rhythm_features);

        Ok(features)
    }

    fn analyze_temporal_matches(
        &self,
        results: Vec<(usize, f32)>,
        filters: &AudioQueryFilters,
    ) -> Result<Vec<TemporalMatch>, StoreError> {
        let mut temporal_matches = Vec::new();

        for (idx, similarity) in results {
            let metadata = self
                .index
                .metadata
                .get(&idx)
                .ok_or(StoreError::Storage("Missing metadata".into()))?;

            if let Some(duration_range) = &filters.duration_range {
                let duration: f32 = metadata.attributes["duration"]
                    .parse()
                    .map_err(|_| StoreError::InvalidMetadata("Invalid duration".into()))?;

                if duration >= duration_range.0 && duration <= duration_range.1 {
                    temporal_matches.push(TemporalMatch {
                        index: idx,
                        similarity,
                        metadata: metadata.clone(),
                        temporal_info: self.extract_temporal_info(idx)?,
                    });
                }
            }
        }

        Ok(temporal_matches)
    }

    fn apply_audio_filters(
        &self,
        matches: Vec<TemporalMatch>,
        filters: &AudioQueryFilters,
    ) -> Result<Vec<FilteredAudioResult>, StoreError> {
        let mut filtered = Vec::new();

        for mat in matches {
            if self.matches_audio_filters(&mat.metadata, filters)? {
                filtered.push(FilteredAudioResult {
                    temporal_match: mat,
                    segments: self.find_matching_segments(&mat, filters)?,
                });
            }
        }

        Ok(filtered)
    }

    fn find_matching_segments(
        &self,
        match_result: &TemporalMatch,
        filters: &AudioQueryFilters,
    ) -> Result<Vec<AudioSegment>, StoreError> {
        let mut segments = Vec::new();

        // Get temporal analysis
        let temporal_data = self.load_temporal_data(match_result.index)?;

        // Find segments matching filters
        for segment in temporal_data.segments {
            if self.segment_matches_filters(&segment, filters)? {
                segments.push(segment);
            }
        }

        Ok(segments)
    }

    fn build_audio_search_results(
        &self,
        filtered: Vec<FilteredAudioResult>,
    ) -> Result<Vec<AudioSearchResult>, StoreError> {
        let mut results = Vec::new();

        for result in filtered {
            let analysis = self.load_audio_analysis(&result.temporal_match.metadata.path)?;

            results.push(AudioSearchResult {
                analysis,
                similarity: result.temporal_match.similarity,
                path: PathBuf::from(&result.temporal_match.metadata.path),
                segment_info: Some(AudioSegmentInfo {
                    segments: result.segments,
                    context: self.build_audio_context(&result)?,
                }),
            });
        }

        Ok(results)
    }

    fn create_metadata(
        &self,
        analysis: &EnhancedAudioAnalysis,
    ) -> Result<IndexMetadata, StoreError> {
        let mut attributes = HashMap::new();
        attributes.insert(
            "content_type".to_string(),
            analysis.content_interpretation.content_type.to_string(),
        );
        attributes.insert(
            "duration".to_string(),
            analysis.base.waveform.duration.to_string(),
        );
        attributes.insert(
            "emotional_mood".to_string(),
            analysis.emotional_analysis.overall_mood.clone(),
        );

        Ok(IndexMetadata {
            id: 0, // Will be set by index
            path: analysis
                .base
                .waveform
                .source_path
                .to_string_lossy()
                .into_owned(),
            modality: "audio".to_string(),
            attributes,
        })
    }
}

#[async_trait]
impl VectorStore for AudioVectorStore {
    type Item = EnhancedAudioAnalysis;
    type Config = StorageConfig;
    type Query = AudioQuery;
    type Result = AudioSearchResult;

    async fn add(&mut self, item: Self::Item) -> Result<(), StoreError> {
        let vector = self.create_vector(&item)?;
        let metadata = self.create_metadata(&item)?;
        self.index.add(vector, metadata)
    }

    async fn search(&self, query: Self::Query) -> Result<Vec<Self::Result>, StoreError> {
        let query_vector = match (&query.audio_features, &query.semantic_description) {
            (Some(features), _) => features.clone(),
            (None, Some(description)) => self.create_semantic_query_vector(description)?,
            (None, None) => return Err(StoreError::InvalidQuery("No query provided".into())),
        };

        // Perform initial similarity search
        let results = self.index.search(&query_vector, self.config.max_results)?;

        // Apply temporal analysis if needed
        let temporal_results = if query.filters.duration_range.is_some() {
            self.analyze_temporal_matches(results, &query.filters)?
        } else {
            results
        };

        // Apply content filters
        let filtered_results = self.apply_audio_filters(temporal_results, &query.filters)?;

        // Build detailed search results
        let search_results = self.build_audio_search_results(filtered_results)?;

        Ok(search_results)
    }

    async fn save(&self, path: PathBuf) -> Result<(), StoreError> {
        let store_data = AudioStoreData {
            vectors: self.index.vectors.clone(),
            metadata: self.index.metadata.clone(),
            config: self.config.clone(),
            temporal_index: self.create_temporal_index()?,
            spectral_features: self.extract_spectral_features()?,
        };

        // Create directory if it doesn't exist
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }

        // Save main data
        let file = File::create(&path)?;
        let buf_writer = BufWriter::new(file);
        bincode::serialize_into(buf_writer, &store_data)
            .map_err(|e| StoreError::Storage(e.to_string()))?;

        Ok(())
    }

    async fn load(&mut self, path: PathBuf) -> Result<(), StoreError> {
        let file = File::open(path)?;
        let buf_reader = BufReader::new(file);
        let store_data: AudioStoreData = bincode::deserialize_from(buf_reader)
            .map_err(|e| StoreError::Storage(e.to_string()))?;

        self.index.vectors = store_data.vectors;
        self.index.metadata = store_data.metadata;
        self.config = store_data.config;
        self.rebuild_temporal_index(&store_data.temporal_index)?;
        self.load_spectral_features(store_data.spectral_features)?;

        Ok(())
    }
}
