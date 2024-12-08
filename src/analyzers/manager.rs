use crate::analyzers::{CodeAnalyzer, Dependency, PythonAnalyzer, RustAnalyzer};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

#[derive(Debug, Serialize, Deserialize, Default)]
pub struct ProjectState {
    #[serde(default)]
    pub last_analysis: DateTime<Utc>,
    #[serde(default)]
    pub analyzed_files: HashMap<PathBuf, FileState>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct FileState {
    pub last_modified: DateTime<Utc>,
    pub dependencies: Vec<Dependency>,
    pub hash: String,
}

#[derive(Debug)]
pub struct AnalysisResult {
    pub dependencies: Vec<Dependency>,
    pub project_structure: ProjectStructure,
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
    pub metadata: Option<FileMetadata>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileMetadata {
    pub last_modified: DateTime<Utc>,
    pub language: Option<String>,
    pub dependencies: Vec<String>,
    pub size: u64,
}

#[derive(Debug)]
pub struct AnalyzerManager {
    analyzers: Vec<Box<dyn CodeAnalyzer>>,
    project_state: ProjectState,
    state_file: PathBuf,
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

    pub fn root_path(&self) -> PathBuf {
        self.state_file
            .parent()
            .unwrap_or(Path::new("."))
            .to_path_buf()
    }

    pub fn analyze_project(&mut self, root_path: &Path) -> Result<AnalysisResult, String> {
        let mut all_dependencies = Vec::new();
        let mut current_files = HashSet::new();
        let mut entries_by_path: HashMap<PathBuf, FileEntry> = HashMap::new();

        // First collect all files that need analysis
        let files_to_analyze: Vec<_> = WalkDir::new(root_path)
            .into_iter()
            .filter_entry(|e| !self.is_ignored(e.path()))
            .filter_map(|e| e.ok())
            .filter(|entry| entry.path().is_file())
            .map(|entry| entry.path().to_path_buf())
            .collect();

        // Now process each file
        for path in files_to_analyze {
            current_files.insert(path.clone());
            let relative_path = path
                .strip_prefix(root_path)
                .map_err(|e| e.to_string())?
                .to_path_buf();

            if let Some(analyzer) = self.get_analyzer_for_file(&path) {
                // Analyze file if needed
                if self.needs_analysis(&path)? {
                    let deps = analyzer.analyze(&path)?;
                    self.update_file_state(&path, &deps)?;
                    all_dependencies.extend(deps.clone());

                    // Create file entry with metadata
                    if let Ok(metadata) = fs::metadata(&path) {
                        entries_by_path.insert(
                            relative_path.clone(),
                            FileEntry {
                                path: relative_path,
                                file_type: path
                                    .extension()
                                    .and_then(|ext| ext.to_str())
                                    .map(|ext| ext.to_string())
                                    .unwrap_or_else(|| "unknown".to_string()),
                                children: Vec::new(),
                                metadata: Some(FileMetadata {
                                    last_modified: DateTime::from(metadata.modified().unwrap()),
                                    language: Some(self.determine_language(&path)),
                                    dependencies: deps
                                        .iter()
                                        .map(|d| d.target.to_string_lossy().to_string())
                                        .collect(),
                                    size: metadata.len(),
                                }),
                            },
                        );
                    }
                }
            }
        }

        // Update state and clean up deleted files
        self.project_state
            .analyzed_files
            .retain(|path, _| current_files.contains(path));

        Ok(AnalysisResult {
            dependencies: all_dependencies,
            project_structure: ProjectStructure {
                root: root_path.to_string_lossy().into_owned(),
                files: self.build_directory_tree(PathBuf::new(), &entries_by_path)?,
            },
        })
    }

    fn build_directory_tree(
        &self,
        current_path: PathBuf,
        entries: &HashMap<PathBuf, FileEntry>,
    ) -> Result<Vec<FileEntry>, String> {
        let mut current_entries = Vec::new();

        for (path, entry) in entries {
            if let Some(parent) = path.parent() {
                if parent == &current_path {
                    let mut entry_clone = entry.clone();
                    if entry.file_type == "directory" {
                        entry_clone.children = self.build_directory_tree(path.clone(), entries)?;
                    }
                    current_entries.push(entry_clone);
                }
            }
        }

        // Sort entries: directories first, then files alphabetically
        current_entries.sort_by(|a, b| {
            match (a.file_type == "directory", b.file_type == "directory") {
                (true, false) => std::cmp::Ordering::Less,
                (false, true) => std::cmp::Ordering::Greater,
                _ => a.path.cmp(&b.path),
            }
        });

        Ok(current_entries)
    }

    fn needs_analysis(&self, path: &Path) -> Result<bool, String> {
        let metadata = fs::metadata(path).map_err(|e| e.to_string())?;
        let modified = metadata.modified().map_err(|e| e.to_string())?;
        let modified: DateTime<Utc> = DateTime::from(modified);

        if let Some(state) = self.project_state.analyzed_files.get(path) {
            let file_hash = self.calculate_file_hash(path)?;
            Ok(modified > state.last_modified || file_hash != state.hash)
        } else {
            Ok(true)
        }
    }

    fn update_file_state(&mut self, path: &Path, deps: &[Dependency]) -> Result<(), String> {
        let metadata = fs::metadata(path).map_err(|e| e.to_string())?;
        let modified = metadata.modified().map_err(|e| e.to_string())?;
        let modified: DateTime<Utc> = DateTime::from(modified);

        self.project_state.analyzed_files.insert(
            path.to_path_buf(),
            FileState {
                last_modified: modified,
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

    fn determine_language(&self, path: &Path) -> String {
        path.extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| match ext {
                "rs" => "Rust",
                "py" => "Python",
                _ => "Unknown",
            })
            .unwrap_or("Unknown")
            .to_string()
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

    fn save_state(&mut self) -> Result<(), String> {
        self.project_state.last_analysis = Utc::now();
        serde_json::to_writer_pretty(
            fs::File::create(&self.state_file).map_err(|e| e.to_string())?,
            &self.project_state,
        )
        .map_err(|e| e.to_string())
    }
}
