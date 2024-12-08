use crate::indexing::VectorStore;
use crate::llm::{Model, ModelResponse};
use crate::relationships::RelationshipContext;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;

pub struct QueryEngine {
    vector_store: Arc<RwLock<VectorStore>>,
    model: Arc<dyn Model>,
    settings: QuerySettings,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct QuerySettings {
    pub max_results: usize,
    pub similarity_threshold: f32,
    pub include_relationships: bool,
    pub context_window: usize,
}

#[derive(Clone)]
pub struct QueryContext {
    pub query: String,
    pub relationship_context: Option<RelationshipContext>,
    pub file_context: Option<String>,
    pub settings: QuerySettings,
}

impl QueryEngine {
    pub fn new(
        vector_store: Arc<RwLock<VectorStore>>,
        model: Arc<dyn Model>,
        settings: QuerySettings,
    ) -> Self {
        Self {
            vector_store,
            model,
            settings,
        }
    }

    pub fn default(vector_store: Arc<RwLock<VectorStore>>, model: Arc<dyn Model>) -> Self {
        Self::new(
            vector_store,
            model,
            QuerySettings {
                max_results: 5,
                similarity_threshold: 0.7,
                include_relationships: true,
                context_window: 3,
            },
        )
    }

    pub async fn query(&self, query: &str) -> Result<QueryResponse, String> {
        // Create query context
        let context = self.build_query_context(query).await?;

        // Search for relevant code snippets
        let search_results = self.search_relevant_code(&context).await?;

        // Analyze relationships in results
        let enhanced_results = self.analyze_relationships(search_results).await?;

        // Generate response using LLM
        let response = self.generate_response(&context, &enhanced_results).await?;

        Ok(self.build_query_response(query, response, enhanced_results))
    }

    async fn build_query_context(&self, query: &str) -> Result<QueryContext, String> {
        // Analyze query intent
        let query_analysis = self.analyze_query_intent(query).await?;

        // Build relationship context if needed
        let relationship_context = if self.settings.include_relationships {
            Some(
                self.build_relationship_context(query, &query_analysis)
                    .await?,
            )
        } else {
            None
        };

        Ok(QueryContext {
            query: query.to_string(),
            relationship_context,
            file_context: None,
            settings: self.settings.clone(),
        })
    }

    async fn analyze_query_intent(&self, query: &str) -> Result<ModelResponse, String> {
        let prompt = format!(
            "Analyze the following code query intent:\n\n{}\n\nIdentify:\n1. Type of information needed\n2. Relevant code elements\n3. Relationship requirements",
            query
        );

        self.model.generate(&prompt).await
    }

    async fn search_relevant_code(
        &self,
        context: &QueryContext,
    ) -> Result<Vec<SearchResult>, String> {
        let store = self.vector_store.read().await;
        store
            .search(&context.query, self.settings.max_results)
            .await
    }

    async fn analyze_relationships(
        &self,
        results: Vec<SearchResult>,
    ) -> Result<Vec<EnhancedSearchResult>, String> {
        let mut enhanced_results = Vec::new();

        for result in results {
            let relationships = self.extract_relationships(&result).await?;
            enhanced_results.push(EnhancedSearchResult {
                result,
                relationships,
            });
        }

        Ok(enhanced_results)
    }

    async fn generate_response(
        &self,
        context: &QueryContext,
        results: &[EnhancedSearchResult],
    ) -> Result<String, String> {
        let prompt = self.build_response_prompt(context, results);
        let response = self.model.generate(&prompt).await?;
        Ok(response.text)
    }

    fn build_response_prompt(
        &self,
        context: &QueryContext,
        results: &[EnhancedSearchResult],
    ) -> String {
        let mut prompt = format!(
            "Answer the following code query:\n\n{}\n\nBased on these relevant code sections:\n\n",
            context.query
        );

        for result in results {
            prompt.push_str(&format!(
                "File: {}\n```\n{}\n```\n\nRelationships:\n{}\n\n",
                result.result.key,
                result
                    .result
                    .metadata
                    .as_ref()
                    .map(|m| m.content_hash.clone())
                    .unwrap_or_default(),
                self.format_relationships(&result.relationships)
            ));
        }

        prompt.push_str("\nProvide a detailed response that:\n");
        prompt.push_str("1. Directly answers the query\n");
        prompt.push_str("2. References specific code sections\n");
        prompt.push_str("3. Explains relevant relationships\n");
        prompt.push_str("4. Provides any necessary context\n");

        prompt
    }

    fn build_query_response(
        &self,
        query: &str,
        response: String,
        results: Vec<EnhancedSearchResult>,
    ) -> QueryResponse {
        QueryResponse {
            query: query.to_string(),
            response,
            code_contexts: results.into_iter().map(|r| CodeContext::from(r)).collect(),
            metadata: self.build_response_metadata(),
        }
    }
}

#[derive(Clone)]
struct SearchResult {
    key: String,
    similarity: f32,
    metadata: Option<NodeMetadata>,
}

#[derive(Clone)]
struct EnhancedSearchResult {
    result: SearchResult,
    relationships: HashMap<String, Vec<String>>,
}
