use crate::analysis::{AnalyzerManager, CodeTracker};
use crate::bridge::LlamaIndexBridge;
use crate::report::ReportGenerator;
use clap::{arg, ArgMatches, Command};
use colored::*;
use indicatif::{ProgressBar, ProgressStyle};
use rustyline::Editor;
use serde_json::Value;
use std::fs;
use std::path::PathBuf;

pub struct CLI {
    analyzer: AnalyzerManager,
    bridge: LlamaIndexBridge,
    report_generator: ReportGenerator,
    project_root: PathBuf,
}

impl CLI {
    pub fn new() -> Result<Self, String> {
        let project_root = std::env::current_dir()
            .map_err(|e| format!("Failed to get current directory: {}", e))?;

        Ok(CLI {
            analyzer: AnalyzerManager::new(&project_root)?,
            bridge: LlamaIndexBridge::new(project_root.clone())?,
            report_generator: ReportGenerator::new(project_root.clone()),
            project_root,
        })
    }

    pub fn run() -> Result<(), String> {
        let matches = Command::new("deeptracking-llamaindex")
            .version("1.0")
            .about("Deep code analysis and semantic search tool")
            .subcommand(
                Command::new("run")
                    .about("Run analysis and query interface")
                    .arg(arg!(-f --force "Force reanalysis of all files")),
            )
            .get_matches();

        match matches.subcommand() {
            Some(("run", sub_matches)) => {
                let mut cli = CLI::new()?;
                cli.handle_run(sub_matches)?;
            }
            _ => {
                println!(
                    "Use 'deeptracking-llamaindex run' to start the analysis and query interface"
                );
            }
        }

        Ok(())
    }

    fn handle_run(&mut self, matches: &ArgMatches) -> Result<(), String> {
        let force_analysis = matches.get_flag("force");

        let spinner = ProgressBar::new_spinner();
        spinner.set_style(
            ProgressStyle::default_spinner()
                .template("{spinner:.green} {msg}")
                .unwrap(),
        );

        // Step 1: Analyze project
        spinner.set_message("Analyzing project structure and dependencies...");
        let dependencies = self.analyzer.analyze_project(&self.project_root)?;

        // Step 2: Initialize LlamaIndex with dependencies
        spinner.set_message("Initializing LlamaIndex...");
        pyo3::Python::with_gil(|py| {
            self.bridge
                .initialize(py)
                .map_err(|e| format!("Failed to initialize LlamaIndex: {}", e))
        })?;

        spinner.finish_with_message("Analysis complete! Starting query interface...");

        // Step 3: Start query interface
        self.run_query_interface()
    }

    fn run_query_interface(&mut self) -> Result<(), String> {
        let mut rl =
            Editor::<()>::new().map_err(|e| format!("Failed to create line editor: {}", e))?;

        println!(
            "\n{}",
            "DeepTracking LlamaIndex Query Interface".green().bold()
        );
        println!("{}", "Enter your query or 'exit' to quit".cyan());
        println!(
            "{}",
            "------------------------------------------------".cyan()
        );

        loop {
            let readline = rl.readline("query> ");
            match readline {
                Ok(line) => {
                    let query = line.trim();
                    if query.eq_ignore_ascii_case("exit") {
                        break;
                    }

                    if !query.is_empty() {
                        rl.add_history_entry(query);
                        match self.execute_query(query) {
                            Ok(report) => self.display_report(&report)?,
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

    fn execute_query(&self, query: &str) -> Result<QueryReport, String> {
        // Execute query through LlamaIndex
        let llama_response = pyo3::Python::with_gil(|py| {
            self.bridge
                .query(py, query)
                .map_err(|e| format!("Query failed: {}", e))
        })?;

        // Generate combined report
        let report = self
            .report_generator
            .generate_report(llama_response, &self.analyzer.get_project_structure()?)?;

        Ok(report)
    }

    fn display_report(&self, report: &QueryReport) -> Result<(), String> {
        // Save report to file
        let report_path = self.project_root.join(".deeptracking-report.md");
        let mut content = String::new();

        // Add project structure
        content.push_str("# Project Structure\n\n");
        self.format_project_structure(&report.project_structure, &mut content);

        // Add query response
        content.push_str("\n# Query Response\n\n");
        content.push_str(&report.llama_index_response.response);

        // Add relevant code sections
        content.push_str("\n# Relevant Code Sections\n\n");
        for node in &report.llama_index_response.source_nodes {
            content.push_str(&format!(
                "\n## File: {}\n\n```{}\n{}\n```\n",
                node.file_path.display(),
                node.metadata.language.as_deref().unwrap_or(""),
                node.content
            ));
        }

        fs::write(&report_path, content).map_err(|e| format!("Failed to write report: {}", e))?;

        println!("\n{}", "Report generated:".green().bold());
        println!("→ {}", report_path.display());

        Ok(())
    }

    fn format_project_structure(&self, structure: &ProjectStructure, content: &mut String) {
        for entry in &structure.entries {
            self.format_entry(entry, 0, content);
        }
    }

    fn format_entry(&self, entry: &FileEntry, depth: usize, content: &mut String) {
        let indent = "  ".repeat(depth);
        match &entry.entry_type {
            EntryType::Directory => {
                content.push_str(&format!("{}📁 {}/\n", indent, entry.path.display()));
                for child in &entry.children {
                    self.format_entry(child, depth + 1, content);
                }
            }
            EntryType::File { extension } => {
                content.push_str(&format!("{}📄 {}\n", indent, entry.path.display()));
            }
        }
    }
}
