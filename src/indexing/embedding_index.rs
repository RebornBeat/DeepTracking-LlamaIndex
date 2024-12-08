use super::embeddings::{EmbeddingVector, TextChunk};
use std::collections::HashMap;
use std::sync::Arc;

pub struct EmbeddingIndex {
    vectors: Vec<(EmbeddingVector, TextChunk)>,
    dimension_indices: Vec<DimensionIndex>,
    config: IndexConfig,
}

struct DimensionIndex {
    dimension: usize,
    values: Vec<(f32, usize)>, // (value, vector_index)
}

#[derive(Clone)]
struct IndexConfig {
    num_dimensions_to_index: usize,
    similarity_threshold: f32,
}

impl EmbeddingIndex {
    pub fn new(config: IndexConfig) -> Self {
        Self {
            vectors: Vec::new(),
            dimension_indices: Vec::new(),
            config,
        }
    }

    pub fn add(&mut self, vector: EmbeddingVector, chunk: TextChunk) {
        let vector_idx = self.vectors.len();
        self.vectors.push((vector.clone(), chunk));

        // Update dimension indices
        for (dim_idx, &value) in vector.0.iter().enumerate() {
            if dim_idx >= self.config.num_dimensions_to_index {
                break;
            }

            while dim_idx >= self.dimension_indices.len() {
                self.dimension_indices.push(DimensionIndex {
                    dimension: dim_idx,
                    values: Vec::new(),
                });
            }

            self.dimension_indices[dim_idx]
                .values
                .push((value, vector_idx));
            self.dimension_indices[dim_idx]
                .values
                .sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        }
    }

    pub fn search(&self, query_vector: &EmbeddingVector, top_k: usize) -> Vec<(TextChunk, f32)> {
        // Find candidate vectors using dimension indices
        let candidates = self.find_candidates(query_vector);

        // Calculate exact similarities for candidates
        let mut results: Vec<_> = candidates
            .into_iter()
            .map(|idx| {
                let (vector, chunk) = &self.vectors[idx];
                let similarity = vector.cosine_similarity(query_vector);
                (chunk.clone(), similarity)
            })
            .filter(|(_, similarity)| *similarity >= self.config.similarity_threshold)
            .collect();

        // Sort by similarity and take top k
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);

        results
    }

    fn find_candidates(&self, query_vector: &EmbeddingVector) -> Vec<usize> {
        let mut candidate_scores: HashMap<usize, u32> = HashMap::new();

        // Check each indexed dimension
        for (dim_idx, value) in query_vector.0.iter().enumerate() {
            if dim_idx >= self.config.num_dimensions_to_index {
                break;
            }

            let dimension_index = &self.dimension_indices[dim_idx];
            let similar_vectors = self.find_similar_in_dimension(dimension_index, *value);

            for vector_idx in similar_vectors {
                *candidate_scores.entry(vector_idx).or_default() += 1;
            }
        }

        // Return candidates that appear in multiple dimensions
        candidate_scores
            .into_iter()
            .filter(|(_, score)| *score > self.config.num_dimensions_to_index / 3)
            .map(|(idx, _)| idx)
            .collect()
    }

    fn find_similar_in_dimension(&self, index: &DimensionIndex, value: f32) -> Vec<usize> {
        let mut similar = Vec::new();
        let threshold = 0.1; // Adjustable similarity threshold

        // Binary search for closest value
        let pos = index.values.binary_search_by(|probe| {
            probe
                .0
                .partial_cmp(&value)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let start_idx = match pos {
            Ok(idx) => idx,
            Err(idx) => idx,
        };

        // Look in both directions for similar values
        let mut left = start_idx;
        let mut right = start_idx;

        while left > 0 && (index.values[left - 1].0 - value).abs() <= threshold {
            left -= 1;
            similar.push(index.values[left].1);
        }

        while right < index.values.len() && (index.values[right].0 - value).abs() <= threshold {
            similar.push(index.values[right].1);
            right += 1;
        }

        similar
    }
}
