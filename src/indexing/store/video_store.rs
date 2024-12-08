use super::{StorageConfig, StoreError, VectorIndex, VectorStore};
use crate::indexing::llm::EnhancedVideoAnalysis;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug)]
pub struct VideoVectorStore {
    index: VectorIndex,
    frame_index: VectorIndex, // Additional index for frame-level search
    config: StorageConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoQuery {
    pub visual_features: Option<Vec<f32>>,
    pub motion_features: Option<Vec<f32>>,
    pub semantic_description: Option<String>,
    pub filters: VideoQueryFilters,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoQueryFilters {
    pub scene_types: Option<Vec<SceneType>>,
    pub duration_range: Option<(f32, f32)>,
    pub motion_intensity: Option<MotionIntensity>,
    pub temporal_constraints: Option<TemporalConstraints>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SceneType {
    Action,
    Dialogue,
    Establishing,
    Transition,
    Montage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MotionIntensity {
    Low,
    Medium,
    High,
    Dynamic,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalConstraints {
    pub min_duration: Option<f32>,
    pub max_duration: Option<f32>,
    pub required_sequence: Option<Vec<SceneType>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoSearchResult {
    pub analysis: EnhancedVideoAnalysis,
    pub similarity: f32,
    pub path: PathBuf,
    pub segment_info: Option<VideoSegmentInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoSegmentInfo {
    pub start_time: f32,
    pub end_time: f32,
    pub scene_type: SceneType,
    pub confidence: f32,
    pub context: VideoContext,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoContext {
    pub narrative_elements: Vec<NarrativeElement>,
    pub temporal_context: TemporalContext,
    pub motion_context: MotionContext,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NarrativeElement {
    pub element_type: String,
    pub significance: f32,
    pub relationships: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalContext {
    pub position_in_sequence: f32,
    pub relative_duration: f32,
    pub pacing: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotionContext {
    pub motion_type: String,
    pub intensity: f32,
    pub camera_movement: Option<String>,
}

impl VideoVectorStore {
    pub fn new(config: StorageConfig) -> Self {
        Self {
            index: VectorIndex::new(IndexConfig {
                num_trees: 15, // Optimized for video feature space
                max_items_per_node: 200,
                search_k: 150,
            }),
            frame_index: VectorIndex::new(IndexConfig {
                num_trees: 10,
                max_items_per_node: 100,
                search_k: 50,
            }),
            config,
        }
    }

    fn create_vector(&self, analysis: &EnhancedVideoAnalysis) -> Result<Vec<f32>, StoreError> {
        let mut vector = Vec::with_capacity(self.config.vector_dimension);

        // 1. Visual features (30% of vector)
        let visual_features = self.encode_visual_features(&analysis.base.frames)?;
        vector.extend(visual_features);

        // 2. Motion features (25% of vector)
        let motion_features = self.encode_motion_features(&analysis.motion_interpretation)?;
        vector.extend(motion_features);

        // 3. Narrative features (25% of vector)
        let narrative_features =
            self.encode_narrative_features(&analysis.narrative_understanding)?;
        vector.extend(narrative_features);

        // 4. Temporal features (20% of vector)
        let temporal_features = self.encode_temporal_features(&analysis.temporal_analysis)?;
        vector.extend(temporal_features);

        // Normalize the vector
        self.normalize_vector(&mut vector)?;

        Ok(vector)
    }

    fn create_frame_vector(&self, frame: &FrameAnalysis) -> Result<Vec<f32>, StoreError> {
        let mut vector = Vec::with_capacity(self.config.vector_dimension);

        // Encode frame-specific features
        let frame_features = self.encode_frame_features(frame)?;
        vector.extend(frame_features);

        // Encode motion vectors
        let motion_features = self.encode_frame_motion(&frame.motion_vectors)?;
        vector.extend(motion_features);

        // Normalize
        self.normalize_vector(&mut vector)?;

        Ok(vector)
    }

    fn encode_visual_features(&self, frames: &[FrameAnalysis]) -> Result<Vec<f32>, StoreError> {
        let mut features = Vec::new();

        // Encode key frame features
        let keyframe_features = self.encode_keyframes(frames)?;
        features.extend(keyframe_features);

        // Encode scene transition features
        let transition_features = self.encode_scene_transitions(frames)?;
        features.extend(transition_features);

        Ok(features)
    }

    fn encode_motion_features(
        &self,
        motion: &MotionInterpretation,
    ) -> Result<Vec<f32>, StoreError> {
        let mut features = Vec::new();

        // Encode motion patterns
        for pattern in &motion.motion_patterns {
            let pattern_features = self.encode_motion_pattern(pattern)?;
            features.extend(pattern_features);
        }

        // Encode camera movement
        let camera_features = self.encode_camera_movement(&motion.camera_movement)?;
        features.extend(camera_features);

        Ok(features)
    }

    fn merge_frame_results(
        &self,
        frame_results: Vec<(usize, f32)>,
    ) -> Result<Vec<VideoMatch>, StoreError> {
        let mut video_matches = HashMap::new();

        // Group frame matches by video
        for (idx, similarity) in frame_results {
            let frame_metadata = self
                .frame_index
                .metadata
                .get(&idx)
                .ok_or(StoreError::Storage("Missing frame metadata".into()))?;

            let video_path = self.get_video_path(frame_metadata)?;
            video_matches
                .entry(video_path)
                .or_insert_with(Vec::new)
                .push((idx, similarity));
        }

        // Convert to video matches
        let mut results = Vec::new();
        for (video_path, frame_matches) in video_matches {
            results.push(self.create_video_match(video_path, frame_matches)?);
        }

        Ok(results)
    }

    fn analyze_motion_matches(
        &self,
        matches: Vec<VideoMatch>,
        motion_features: &[f32],
    ) -> Result<Vec<MotionAnalyzedMatch>, StoreError> {
        let mut analyzed = Vec::new();

        for mat in matches {
            let motion_analysis = self.analyze_motion_similarity(&mat, motion_features)?;

            if motion_analysis.similarity >= self.config.motion_similarity_threshold {
                analyzed.push(MotionAnalyzedMatch {
                    video_match: mat,
                    motion_analysis,
                });
            }
        }

        Ok(analyzed)
    }

    fn apply_video_filters(
        &self,
        matches: Vec<MotionAnalyzedMatch>,
        filters: &VideoQueryFilters,
    ) -> Result<Vec<FilteredVideoResult>, StoreError> {
        let mut filtered = Vec::new();

        for mat in matches {
            if self.matches_video_filters(&mat, filters)? {
                let segments = self.find_matching_segments(&mat, filters)?;

                if !segments.is_empty() {
                    filtered.push(FilteredVideoResult {
                        motion_match: mat,
                        segments,
                    });
                }
            }
        }

        Ok(filtered)
    }

    fn build_video_search_results(
        &self,
        filtered: Vec<FilteredVideoResult>,
    ) -> Result<Vec<VideoSearchResult>, StoreError> {
        let mut results = Vec::new();

        for result in filtered {
            let analysis = self.load_video_analysis(&result.motion_match.video_match.path)?;

            results.push(VideoSearchResult {
                analysis,
                similarity: result.motion_match.video_match.similarity,
                path: result.motion_match.video_match.path.clone(),
                segment_info: Some(VideoSegmentInfo {
                    segments: result.segments,
                    context: self.build_video_context(&result)?,
                }),
            });
        }

        Ok(results)
    }

    fn build_video_context(
        &self,
        result: &FilteredVideoResult,
    ) -> Result<VideoContext, StoreError> {
        // Build video context including:
        // - Narrative flow
        // - Motion patterns
        // - Temporal relationships
        // - Scene composition
        todo!()
    }

    fn create_metadata(
        &self,
        analysis: &EnhancedVideoAnalysis,
    ) -> Result<IndexMetadata, StoreError> {
        let mut attributes = HashMap::new();
        attributes.insert(
            "duration".to_string(),
            analysis.temporal_analysis.total_duration.to_string(),
        );
        attributes.insert(
            "scene_count".to_string(),
            analysis
                .narrative_understanding
                .story_elements
                .len()
                .to_string(),
        );
        attributes.insert(
            "motion_complexity".to_string(),
            analysis.motion_interpretation.complexity_score.to_string(),
        );

        Ok(IndexMetadata {
            id: 0, // Will be set by index
            path: analysis.base.path.to_string_lossy().into_owned(),
            modality: "video".to_string(),
            attributes,
        })
    }

    async fn index_frames(&mut self, analysis: &EnhancedVideoAnalysis) -> Result<(), StoreError> {
        for frame in &analysis.base.frames {
            let frame_vector = self.create_frame_vector(frame)?;
            let frame_metadata = self.create_frame_metadata(frame, &analysis.base.path)?;
            self.frame_index.add(frame_vector, frame_metadata)?;
        }
        Ok(())
    }
}

#[async_trait]
impl VectorStore for VideoVectorStore {
    type Item = EnhancedVideoAnalysis;
    type Config = StorageConfig;
    type Query = VideoQuery;
    type Result = VideoSearchResult;

    async fn add(&mut self, item: Self::Item) -> Result<(), StoreError> {
        // Index both the full video and individual frames
        let vector = self.create_vector(&item)?;
        let metadata = self.create_metadata(&item)?;
        self.index.add(vector, metadata)?;

        // Index frames separately
        self.index_frames(&item).await?;

        Ok(())
    }

    async fn search(&self, query: Self::Query) -> Result<Vec<Self::Result>, StoreError> {
        let mut final_results = Vec::new();

        // 1. Search full video index
        let video_vector = self.create_query_vector(&query)?;
        let video_results = self.index.search(&video_vector, self.config.max_results)?;

        // 2. Search frame index if needed
        if let Some(visual_features) = query.visual_features {
            let frame_results = self
                .frame_index
                .search(&visual_features, self.config.max_results)?;
            final_results.extend(self.merge_frame_results(frame_results)?);
        }

        // 3. Apply motion analysis
        if let Some(motion_features) = query.motion_features {
            final_results = self.analyze_motion_matches(final_results, &motion_features)?;
        }

        // 4. Apply filters
        let filtered_results = self.apply_video_filters(final_results, &query.filters)?;

        // 5. Build final search results with segments
        let search_results = self.build_video_search_results(filtered_results)?;

        Ok(search_results)
    }

    async fn save(&self, path: PathBuf) -> Result<(), StoreError> {
        let store_data = VideoStoreData {
            main_index: self.index.clone(),
            frame_index: self.frame_index.clone(),
            config: self.config.clone(),
            temporal_metadata: self.extract_temporal_metadata()?,
            motion_features: self.extract_motion_features()?,
        };

        // Save in chunks due to potentially large size
        let chunks = self.split_store_data(&store_data)?;

        for (i, chunk) in chunks.iter().enumerate() {
            let chunk_path = path.with_extension(format!("part{}", i));
            let file = File::create(chunk_path)?;
            let compressed = self.compress_chunk(chunk)?;
            fs::write(file, compressed)?;
        }

        // Save metadata
        let metadata_path = path.with_extension("meta");
        let metadata = VideoStoreMetadata {
            num_chunks: chunks.len(),
            total_frames: store_data.frame_index.len(),
            config: self.config.clone(),
        };

        let meta_file = File::create(metadata_path)?;
        serde_json::to_writer(meta_file, &metadata)
            .map_err(|e| StoreError::Storage(e.to_string()))?;

        Ok(())
    }

    async fn load(&mut self, path: PathBuf) -> Result<(), StoreError> {
        // Load metadata first
        let metadata_path = path.with_extension("meta");
        let metadata: VideoStoreMetadata = serde_json::from_reader(File::open(metadata_path)?)
            .map_err(|e| StoreError::Storage(e.to_string()))?;

        // Load chunks
        let mut store_data = VideoStoreData::default();
        for i in 0..metadata.num_chunks {
            let chunk_path = path.with_extension(format!("part{}", i));
            let compressed = fs::read(chunk_path)?;
            let chunk = self.decompress_chunk(&compressed)?;
            self.merge_chunk_into_store_data(&mut store_data, chunk)?;
        }

        // Rebuild indices
        self.index = store_data.main_index;
        self.frame_index = store_data.frame_index;
        self.config = store_data.config;
        self.rebuild_indices(&store_data)?;

        Ok(())
    }
}
