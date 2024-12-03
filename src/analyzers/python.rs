use super::{CodeAnalyzer, Dependency, DependencyMetadata, DependencyType};
use lazy_static::lazy_static;
use regex::Regex;
use std::fs;
use std::path::{Path, PathBuf};

pub struct PythonAnalyzer {
    class_pattern: Regex,
    function_pattern: Regex,
    import_pattern: Regex,
    from_import_pattern: Regex,
}

impl PythonAnalyzer {
    pub fn new() -> Self {
        lazy_static! {
            static ref CLASS_RE: Regex =
                Regex::new(r"^class\s+(\w+)(?:\s*\([^)]*\))?\s*:").unwrap();
            static ref FUNC_RE: Regex =
                Regex::new(r"^def\s+(\w+)\s*\([^)]*\)\s*(?:->.*)?:").unwrap();
            static ref IMPORT_RE: Regex = Regex::new(r"^import\s+([\w,\s]+)").unwrap();
            static ref FROM_IMPORT_RE: Regex =
                Regex::new(r"^from\s+([\w.]+)\s+import\s+([\w,\s]+)").unwrap();
        }

        PythonAnalyzer {
            class_pattern: CLASS_RE.clone(),
            function_pattern: FUNC_RE.clone(),
            import_pattern: IMPORT_RE.clone(),
            from_import_pattern: FROM_IMPORT_RE.clone(),
        }
    }

    fn analyze_imports(&self, content: &str, path: &Path) -> Vec<Dependency> {
        let mut dependencies = Vec::new();

        for (line_num, line) in content.lines().enumerate() {
            let line = line.trim();

            // Handle standard imports
            if let Some(cap) = self.import_pattern.captures(line) {
                let imports = cap[1].split(',').map(str::trim);
                for import in imports {
                    dependencies.push(Dependency {
                        source: path.to_path_buf(),
                        target: PathBuf::from(import),
                        dependency_type: DependencyType::Import,
                        metadata: DependencyMetadata {
                            line_number: Some(line_num),
                            description: Some(format!("Import: {}", import)),
                        },
                    });
                }
            }

            // Handle from ... import ...
            if let Some(cap) = self.from_import_pattern.captures(line) {
                let module = &cap[1];
                let imports = cap[2].split(',').map(str::trim);
                for import in imports {
                    dependencies.push(Dependency {
                        source: path.to_path_buf(),
                        target: PathBuf::from(format!("{}.{}", module, import)),
                        dependency_type: DependencyType::Import,
                        metadata: DependencyMetadata {
                            line_number: Some(line_num),
                            description: Some(format!("From {} import {}", module, import)),
                        },
                    });
                }
            }
        }

        dependencies
    }

    fn analyze_classes_and_functions(&self, content: &str, path: &Path) -> Vec<Dependency> {
        let mut dependencies = Vec::new();
        let mut current_class: Option<String> = None;
        let mut indent_level = 0;

        for (line_num, line) in content.lines().enumerate() {
            let spaces = line.chars().take_while(|c| c.is_whitespace()).count();
            let line = line.trim();

            // Track indentation
            if spaces == 0 {
                current_class = None;
                indent_level = 0;
            }

            // Class definition
            if let Some(cap) = self.class_pattern.captures(line) {
                let class_name = &cap[1];
                current_class = Some(class_name.to_string());
                indent_level = spaces;

                dependencies.push(Dependency {
                    source: path.to_path_buf(),
                    target: PathBuf::from(format!("class:{}", class_name)),
                    dependency_type: DependencyType::Inheritance,
                    metadata: DependencyMetadata {
                        line_number: Some(line_num),
                        description: Some(format!("Class definition: {}", class_name)),
                    },
                });
            }

            // Function definition
            if let Some(cap) = self.function_pattern.captures(line) {
                let func_name = &cap[1];
                let qualified_name = if let Some(ref class_name) = current_class {
                    format!("{}::{}", class_name, func_name)
                } else {
                    func_name.to_string()
                };

                dependencies.push(Dependency {
                    source: path.to_path_buf(),
                    target: PathBuf::from(format!("function:{}", qualified_name)),
                    dependency_type: DependencyType::FunctionCall,
                    metadata: DependencyMetadata {
                        line_number: Some(line_num),
                        description: Some(format!("Function definition: {}", qualified_name)),
                    },
                });
            }
        }

        dependencies
    }
}

impl CodeAnalyzer for PythonAnalyzer {
    fn analyze(&self, path: &Path) -> Result<Vec<Dependency>, String> {
        let content = fs::read_to_string(path).map_err(|e| e.to_string())?;
        let mut dependencies = Vec::new();

        dependencies.extend(self.analyze_imports(&content, path));
        dependencies.extend(self.analyze_classes_and_functions(&content, path));

        Ok(dependencies)
    }

    fn supported_extensions(&self) -> Vec<&'static str> {
        vec!["py"]
    }
}
