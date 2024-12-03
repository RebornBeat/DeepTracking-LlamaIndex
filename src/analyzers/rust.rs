pub struct RustAnalyzer;

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
