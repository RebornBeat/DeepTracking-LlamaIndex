use crate::analyzers::manager::{AnalyzerManager, FileEntry, ProjectStructure};
use crate::indexing::Indexer;
use crate::query::{QueryEngine, QueryResponse, QueryResult};
use clap::{arg, ArgMatches, Command};
use colored::*;
use indicatif::{ProgressBar, ProgressStyle};
use rustyline::history::DefaultHistory;
use rustyline::Editor;
use std::fs;
use std::path::{Path, PathBuf};
use tokio;

pub struct CLI {
    analyzer: AnalyzerManager,
    project_root: PathBuf,
    indexer: Option<Indexer>,
    query_engine: Option<QueryEngine>,
}

impl CLI {
    pub fn new() -> Result<Self, String> {
        let project_root = std::env::current_dir()
            .map_err(|e| format!("Failed to get current directory: {}", e))?;

        Ok(CLI {
            analyzer: AnalyzerManager::new(&project_root)?,
            project_root,
            indexer: None,
            query_engine: None,
        })
    }

    pub fn run() -> Result<(), String> {
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| format!("Failed to create Tokio runtime: {}", e))?;

        rt.block_on(async {
            let matches = Command::new("deeptracking-llamaindex")
                .version("1.0")
                .about("Deep code analysis and semantic search tool")
                .subcommand(
                    Command::new("run")
                        .about("Run analysis and query interface")
                        .arg(arg!(-f --force "Force reanalysis of all files"))
                        .arg(arg!(-m --model <PATH> "Path to LLM model"))
                        .arg(arg!(-i --index <PATH> "Path to existing index")),
                )
                .get_matches();

            match matches.subcommand() {
                Some(("run", sub_matches)) => {
                    let mut cli = CLI::new()?;
                    cli.handle_run(sub_matches).await
                }
                _ => {
                    println!(
                        "Use 'deeptracking-llamaindex run' to start the analysis and query interface"
                    );
                    Ok(())
                }
            }
        })
    }

    async fn handle_run(&mut self, matches: &ArgMatches) -> Result<(), String> {
        let force_analysis = matches.get_flag("force");
        let model_path = matches.get_one::<String>("model");
        let index_path = matches.get_one::<String>("index");

        let spinner = ProgressBar::new_spinner();
        spinner.set_style(
            ProgressStyle::default_spinner()
                .template("{spinner:.green} {msg}")
                .unwrap(),
        );

        // Initialize indexer
        spinner.set_message("Initializing indexer...");
        self.indexer = Some(Indexer::new()?);

        // Load existing index or perform new analysis
        if let Some(index_path) = index_path {
            spinner.set_message("Loading existing index...");
            if let Some(indexer) = &mut self.indexer {
                indexer.load(PathBuf::from(index_path)).await?;
            }
        } else {
            // Perform new analysis
            spinner.set_message("Analyzing project structure and dependencies...");
            let analysis_result = self.analyzer.analyze_project(&self.project_root)?;

            // Index all files
            spinner.set_message("Indexing files...");
            if let Some(indexer) = &mut self.indexer {
                for file_entry in &analysis_result.project_structure.files {
                    indexer.index_file(file_entry).await?;
                }
            }
        }

        // Initialize query engine
        spinner.set_message("Initializing query engine...");
        if let Some(indexer) = &self.indexer {
            self.query_engine = Some(QueryEngine::new(
                indexer.clone(),
                model_path.map(PathBuf::from),
            )?);
        }

        spinner.finish_with_message("Analysis complete! Starting query interface...");

        // Start query interface
        self.run_query_interface().await
    }

    async fn run_query_interface(&mut self) -> Result<(), String> {
        let mut rl = Editor::<(), DefaultHistory>::new()
            .map_err(|e| format!("Failed to create line editor: {}", e))?;

        println!("\n{}", "DeepTracking Code Query Interface".green().bold());
        println!("{}", "Enter your query or 'exit' to quit".cyan());
        println!("Commands:".cyan());
        println!("  :save <path> - Save current index");
        println!("  :load <path> - Load index from file");
        println!("  :help       - Show this help");
        println!(
            "{}",
            "------------------------------------------------".cyan()
        );

        loop {
            let readline = rl.readline("query> ");
            match readline {
                Ok(line) => {
                    let input = line.trim();
                    if input.eq_ignore_ascii_case("exit") {
                        break;
                    }

                    if !input.is_empty() {
                        rl.add_history_entry(input);

                        // Handle special commands
                        if input.starts_with(':') {
                            self.handle_command(input).await?;
                            continue;
                        }

                        // Execute query
                        match self.execute_query(input).await {
                            Ok(result) => self.generate_query_report(&result)?,
                            Err(e) => println!("{}: {}", "Error".red().bold(), e),
                        }
                    }
                }
                Err(err) => {
                    println!("Error: {}", err);
                    break;
                }
            }
        }

        Ok(())
    }

    async fn handle_command(&mut self, command: &str) -> Result<(), String> {
        let parts: Vec<&str> = command.splitn(2, ' ').collect();
        match parts[0] {
            ":save" => {
                if parts.len() != 2 {
                    println!("Usage: :save <path>");
                    return Ok(());
                }
                if let Some(indexer) = &self.indexer {
                    indexer.save(PathBuf::from(parts[1])).await?;
                    println!("Index saved to: {}", parts[1]);
                }
            }
            ":load" => {
                if parts.len() != 2 {
                    println!("Usage: :load <path>");
                    return Ok(());
                }
                if let Some(indexer) = &mut self.indexer {
                    indexer.load(PathBuf::from(parts[1])).await?;
                    println!("Index loaded from: {}", parts[1]);
                }
            }
            ":help" => {
                println!("Available commands:");
                println!("  :save <path> - Save current index");
                println!("  :load <path> - Load index from file");
                println!("  :help       - Show this help");
                println!("  exit        - Exit the program");
            }
            _ => println!("Unknown command: {}", command),
        }
        Ok(())
    }

    async fn execute_query(&self, query: &str) -> Result<QueryResult, String> {
        if let Some(engine) = &self.query_engine {
            engine.query(query).await
        } else {
            Err("Query engine not initialized".to_string())
        }
    }

    fn generate_query_report(&self, result: &QueryResult) -> Result<(), String> {
        let report_number = self.get_next_report_number()?;
        let report_path = self
            .project_root
            .join(format!(".deeptracking-query-report-{}.md", report_number));

        let mut content = String::new();

        // Add project structure
        content.push_str("# Project Structure\n\n");
        if let Some(structure) = &result.project_structure {
            self.format_project_structure(structure, &mut content);
        }
        content.push_str("\n");

        // Add query and response
        content.push_str("# Query Report\n\n");
        content.push_str(&format!("## Query\n\n{}\n\n", result.query));
        content.push_str(&format!("## Response\n\n{}\n\n", result.response));

        // Add code contexts with relationships
        content.push_str("## Relevant Code Contexts\n\n");
        for context in &result.code_contexts {
            content.push_str(&format!("### {}\n\n", context.file_path.display()));
            content.push_str("```");
            if let Some(lang) = &context.language {
                content.push_str(lang);
            }
            content.push_str("\n");
            content.push_str(&context.content);
            content.push_str("\n```\n\n");

            // Add relationship context
            if !context.relationships.is_empty() {
                content.push_str("#### Relationships\n\n");
                for (rel_type, targets) in &context.relationships {
                    content.push_str(&format!("- {}: {}\n", rel_type, targets.join(", ")));
                }
                content.push_str("\n");
            }

            // Add dependency paths if available
            if let Some(deps) = &context.dependency_paths {
                content.push_str("#### Dependency Paths\n\n");
                for dep in deps {
                    content.push_str(&format!(
                        "- {} → {}: {}\n",
                        dep.source,
                        dep.target,
                        dep.path.join(" → ")
                    ));
                }
                content.push_str("\n");
            }
        }

        // Add project context section
        if let Some(project_context) = &result.project_context {
            content.push_str("## Project Context\n\n");
            content.push_str(&format!(
                "Analyzed Files: {}\n\n",
                project_context.analyzed_files
            ));

            // Add file relationships
            if !project_context.file_relationships.is_empty() {
                content.push_str("### File Relationships\n\n");
                for (file, relationships) in &project_context.file_relationships {
                    content.push_str(&format!("#### {}\n", file));
                    for rel in relationships {
                        content.push_str(&format!("- {}\n", rel));
                    }
                    content.push_str("\n");
                }
            }
        }

        fs::write(&report_path, content).map_err(|e| format!("Failed to write report: {}", e))?;

        println!("\n{}", "Report generated:".green().bold());
        println!("→ {}", report_path.display());

        Ok(())
    }

    fn get_next_report_number(&self) -> Result<u32, String> {
        let mut highest = 0;

        for entry in fs::read_dir(&self.project_root)
            .map_err(|e| format!("Failed to read directory: {}", e))?
        {
            let entry = entry.map_err(|e| format!("Failed to read entry: {}", e))?;
            let filename = entry.file_name();
            let filename_str = filename.to_string_lossy();

            if filename_str.starts_with(".deeptracking-query-report-")
                && filename_str.ends_with(".md")
            {
                if let Some(num_str) = filename_str
                    .strip_prefix(".deeptracking-query-report-")
                    .and_then(|s| s.strip_suffix(".md"))
                {
                    if let Ok(num) = num_str.parse::<u32>() {
                        highest = highest.max(num);
                    }
                }
            }
        }

        Ok(highest + 1)
    }
}

fn format_project_structure(&self, structure: &ProjectStructure, content: &mut String) {
    for entry in &structure.files {
        self.format_entry(entry, 0, content);
    }
}

fn format_entry(&self, entry: &FileEntry, depth: usize, content: &mut String) {
    let indent = "  ".repeat(depth);
    if entry.file_type == "directory" {
        content.push_str(&format!("{}📁 {}/\n", indent, entry.path.display()));
        for child in &entry.children {
            self.format_entry(child, depth + 1, content);
        }
    } else {
        let icon = match entry.file_type.as_str() {
            "rs" => "🦀",
            "py" => "🐍",
            _ => "📄",
        };

        content.push_str(&format!("{}{} {}", indent, icon, entry.path.display()));

        // Add file metadata if available
        if let Some(metadata) = &entry.metadata {
            if let Some(lang) = &metadata.language {
                content.push_str(&format!(" ({})", lang));
            }
            if !metadata.dependencies.is_empty() {
                content.push_str("\n");
                content.push_str(&format!(
                    "{}  Dependencies: {}\n",
                    indent,
                    metadata.dependencies.join(", ")
                ));
            }
        }
        content.push_str("\n");
    }
}
