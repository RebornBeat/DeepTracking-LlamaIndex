use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

#[derive(Clone, Serialize, Deserialize)]
pub struct QueryResponse {
    pub query: String,
    pub response: String,
    pub code_contexts: Vec<CodeContext>,
    pub metadata: ResponseMetadata,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct CodeContext {
    pub file_path: PathBuf,
    pub content: String,
    pub language: Option<String>,
    pub relationships: HashMap<String, Vec<String>>,
    pub similarity_score: f32,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct ResponseMetadata {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub execution_time_ms: u64,
    pub num_results: usize,
    pub relationship_depth: usize,
}

impl From<EnhancedSearchResult> for CodeContext {
    fn from(result: EnhancedSearchResult) -> Self {
        CodeContext {
            file_path: PathBuf::from(&result.result.key),
            content: result
                .result
                .metadata
                .as_ref()
                .map(|m| m.content_hash.clone())
                .unwrap_or_default(),
            language: result
                .result
                .metadata
                .as_ref()
                .and_then(|m| m.language.clone()),
            relationships: result.relationships,
            similarity_score: result.result.similarity,
        }
    }
}

impl QueryResponse {
    pub fn new(
        query: String,
        response: String,
        code_contexts: Vec<CodeContext>,
        execution_time_ms: u64,
    ) -> Self {
        Self {
            query,
            response,
            code_contexts,
            metadata: ResponseMetadata {
                timestamp: chrono::Utc::now(),
                execution_time_ms,
                num_results: code_contexts.len(),
                relationship_depth: 2, // Default value, could be configurable
            },
        }
    }

    pub fn format_markdown(&self) -> String {
        let mut markdown = String::new();

        // Add query and response
        markdown.push_str(&format!("# Query\n\n{}\n\n", self.query));
        markdown.push_str(&format!("# Response\n\n{}\n\n", self.response));

        // Add code contexts
        markdown.push_str("# Relevant Code Sections\n\n");
        for context in &self.code_contexts {
            markdown.push_str(&format!("## {}\n\n", context.file_path.display()));
            markdown.push_str("```");
            if let Some(lang) = &context.language {
                markdown.push_str(lang);
            }
            markdown.push_str("\n");
            markdown.push_str(&context.content);
            markdown.push_str("\n```\n\n");

            // Add relationships
            if !context.relationships.is_empty() {
                markdown.push_str("### Relationships\n\n");
                for (rel_type, targets) in &context.relationships {
                    markdown.push_str(&format!("- {}: {}\n", rel_type, targets.join(", ")));
                }
                markdown.push_str("\n");
            }
        }

        // Add metadata
        markdown.push_str("# Metadata\n\n");
        markdown.push_str(&format!("- Timestamp: {}\n", self.metadata.timestamp));
        markdown.push_str(&format!(
            "- Execution Time: {}ms\n",
            self.metadata.execution_time_ms
        ));
        markdown.push_str(&format!("- Results Found: {}\n", self.metadata.num_results));

        markdown
    }
}
