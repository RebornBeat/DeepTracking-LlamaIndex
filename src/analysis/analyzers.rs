use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dependency {
    pub source: PathBuf,
    pub target: PathBuf,
    pub dependency_type: DependencyType,
    pub metadata: DependencyMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DependencyType {
    Import,
    FunctionCall,
    Inheritance,
    Usage,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DependencyMetadata {
    pub line_number: Option<usize>,
    pub description: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ProjectState {
    pub last_analysis: DateTime<Utc>,
    pub analyzed_files: HashMap<PathBuf, FileState>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct FileState {
    pub last_modified: DateTime<Utc>,
    pub dependencies: Vec<Dependency>,
    pub hash: String, // For detecting content changes
}

pub struct AnalyzerManager {
    analyzers: Vec<Box<dyn CodeAnalyzer>>,
    project_state: ProjectState,
    state_file: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisReport {
    pub project_structure: ProjectStructure,
    pub relevant_code: Vec<CodeSection>,
    pub dependencies: Vec<Dependency>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectStructure {
    pub root: String,
    pub files: Vec<FileEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileEntry {
    pub path: PathBuf,
    pub file_type: String,
    pub children: Vec<FileEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeSection {
    pub file: PathBuf,
    pub content: String,
    pub relevance_score: f32,
}

impl AnalyzerManager {
    pub fn new(project_root: &Path) -> Result<Self, String> {
        let state_file = project_root.join(".deeptracking-state.json");
        let project_state = if state_file.exists() {
            serde_json::from_reader(fs::File::open(&state_file).map_err(|e| e.to_string())?)
                .unwrap_or_default()
        } else {
            ProjectState {
                last_analysis: Utc::now(),
                analyzed_files: HashMap::new(),
            }
        };

        Ok(Self {
            analyzers: vec![
                Box::new(RustAnalyzer::new()),
                Box::new(PythonAnalyzer::new()),
            ],
            project_state,
            state_file,
        })
    }

    pub fn analyze_project(&mut self, root_path: &Path) -> Result<Vec<Dependency>, String> {
        let mut all_dependencies = Vec::new();
        let mut current_files = HashSet::new();

        // Walk through project directory
        for entry in WalkDir::new(root_path)
            .into_iter()
            .filter_entry(|e| !self.is_ignored(e.path()))
        {
            let entry = entry.map_err(|e| e.to_string())?;
            let path = entry.path();

            if !path.is_file() {
                continue;
            }

            current_files.insert(path.to_path_buf());

            // Check if file needs analysis
            if let Some(analyzer) = self.get_analyzer_for_file(path) {
                if self.needs_analysis(path)? {
                    let deps = analyzer.analyze(path)?;
                    self.update_file_state(path, &deps)?;
                    all_dependencies.extend(deps);
                } else {
                    if let Some(state) = self.project_state.analyzed_files.get(path) {
                        all_dependencies.extend(state.dependencies.clone());
                    }
                }
            }
        }

        // Remove state for deleted files
        self.project_state
            .analyzed_files
            .retain(|path, _| current_files.contains(path));
        self.save_state()?;

        Ok(all_dependencies)
    }

    fn needs_analysis(&self, path: &Path) -> Result<bool, String> {
        let metadata = fs::metadata(path).map_err(|e| e.to_string())?;
        let modified = metadata.modified().map_err(|e| e.to_string())?;

        if let Some(state) = self.project_state.analyzed_files.get(path) {
            let file_hash = self.calculate_file_hash(path)?;
            Ok(DateTime::from(modified) > state.last_modified || file_hash != state.hash)
        } else {
            Ok(true)
        }
    }

    fn update_file_state(&mut self, path: &Path, deps: &[Dependency]) -> Result<(), String> {
        let metadata = fs::metadata(path).map_err(|e| e.to_string())?;
        let modified = metadata.modified().map_err(|e| e.to_string())?;

        self.project_state.analyzed_files.insert(
            path.to_path_buf(),
            FileState {
                last_modified: DateTime::from(modified),
                dependencies: deps.to_vec(),
                hash: self.calculate_file_hash(path)?,
            },
        );
        Ok(())
    }

    fn calculate_file_hash(&self, path: &Path) -> Result<String, String> {
        let content = fs::read(path).map_err(|e| e.to_string())?;
        Ok(format!("{:x}", md5::compute(&content)))
    }

    fn get_analyzer_for_file(&self, path: &Path) -> Option<&dyn CodeAnalyzer> {
        let extension = path.extension()?.to_str()?;
        self.analyzers
            .iter()
            .find(|analyzer| analyzer.supported_extensions().contains(&extension))
            .map(|analyzer| analyzer.as_ref())
    }

    fn is_ignored(&self, path: &Path) -> bool {
        let ignored_patterns = [
            "target",
            "node_modules",
            ".git",
            "__pycache__",
            ".deeptracking-state.json",
        ];

        path.components().any(|c| {
            if let Some(s) = c.as_os_str().to_str() {
                ignored_patterns.contains(&s)
            } else {
                false
            }
        })
    }

    fn save_state(&self) -> Result<(), String> {
        self.project_state.last_analysis = Utc::now();
        serde_json::to_writer_pretty(
            fs::File::create(&self.state_file).map_err(|e| e.to_string())?,
            &self.project_state,
        )
        .map_err(|e| e.to_string())
    }
}

pub trait CodeAnalyzer {
    fn analyze(&self, path: &Path) -> Result<Vec<Dependency>, String>;
    fn supported_extensions(&self) -> Vec<&'static str>;
}

impl CodeAnalyzer for RustAnalyzer {
    fn analyze(&self, path: &Path) -> Result<Vec<Dependency>, String> {
        let content = fs::read_to_string(path).map_err(|e| e.to_string())?;
        let mut dependencies = Vec::new();

        dependencies.extend(self.analyze_imports(&content, path)?);
        dependencies.extend(self.analyze_function_calls(&content, path)?);

        Ok(dependencies)
    }

    fn supported_extensions(&self) -> Vec<&'static str> {
        vec!["rs"]
    }
}

pub struct RustAnalyzer;
pub struct PythonAnalyzer;

impl RustAnalyzer {
    pub fn new() -> Self {
        RustAnalyzer
    }

    fn analyze_imports(&self, content: &str, path: &Path) -> Result<Vec<Dependency>, String> {
        let mut dependencies = Vec::new();
        for line in content.lines() {
            let line = line.trim();
            if line.starts_with("use ") || line.starts_with("mod ") {
                let import_parts: Vec<&str> = line.split_whitespace().collect();
                if import_parts.len() >= 2 {
                    let target = import_parts[1].trim_end_matches(';');
                    dependencies.push(Dependency {
                        source: path.to_path_buf(),
                        target: PathBuf::from(target),
                        dependency_type: DependencyType::Import,
                        metadata: DependencyMetadata {
                            line_number: Some(content.lines().position(|l| l == line).unwrap_or(0)),
                            description: Some(format!("Import: {}", target)),
                        },
                    });
                }
            }
        }
        Ok(dependencies)
    }

    fn analyze_function_calls(
        &self,
        content: &str,
        path: &Path,
    ) -> Result<Vec<Dependency>, String> {
        let mut dependencies = Vec::new();
        // Basic function call analysis - in practice, you'd want more sophisticated parsing
        for (line_num, line) in content.lines().enumerate() {
            if let Some(func_call) = self.extract_function_call(line) {
                dependencies.push(Dependency {
                    source: path.to_path_buf(),
                    target: PathBuf::from(&func_call),
                    dependency_type: DependencyType::FunctionCall,
                    metadata: DependencyMetadata {
                        line_number: Some(line_num),
                        description: Some(format!("Function call: {}", func_call)),
                    },
                });
            }
        }
        Ok(dependencies)
    }

    fn extract_function_call(&self, line: &str) -> Option<String> {
        // Simple function call detection - would need more sophisticated parsing in practice
        let line = line.trim();
        if line.contains("(") && line.contains(")") {
            let function_name = line.split("(").next()?.trim().to_string();
            if !function_name.is_empty() {
                return Some(function_name);
            }
        }
        None
    }
}

impl PythonAnalyzer {
    pub fn new() -> Self {
        PythonAnalyzer
    }
}
