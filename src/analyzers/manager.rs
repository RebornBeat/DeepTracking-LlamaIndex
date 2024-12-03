use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

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
