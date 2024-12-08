use crate::analyzers::{CodeAnalyzer, Dependency, DependencyMetadata, DependencyType};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug)]
pub struct RustAnalyzer {
    // Track function definitions and their relationships
    functions: HashMap<String, FunctionInfo>,
    // Track modules and their contents
    modules: HashMap<String, ModuleInfo>,
    // Track current analysis state
    current_scope: Vec<String>,
}

#[derive(Debug, Clone)]
struct FunctionInfo {
    name: String,
    module_path: Vec<String>,
    line_number: usize,
    parameters: Vec<ParameterInfo>,
    return_type: Option<String>,
    calls: HashSet<String>,
    is_public: bool,
    block_start: usize,
    block_end: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ParameterInfo {
    name: String,
    param_type: Option<String>,
    is_mutable: bool,
}

#[derive(Debug, Clone)]
struct ModuleInfo {
    name: String,
    path: PathBuf,
    functions: HashSet<String>,
    submodules: HashSet<String>,
    imports: HashSet<String>,
}

impl RustAnalyzer {
    pub fn new() -> Self {
        Self {
            functions: HashMap::new(),
            modules: HashMap::new(),
            current_scope: Vec::new(),
        }
    }

    fn analyze_file(&mut self, content: &str, path: &Path) -> Result<Vec<Dependency>, String> {
        let mut dependencies = Vec::new();
        let mut current_block_depth = 0;
        let mut in_function = false;
        let mut current_function: Option<String> = None;

        let lines: Vec<_> = content.lines().collect();
        let total_lines = lines.len();

        for (line_num, line) in lines.iter().enumerate() {
            let line = line.trim();

            // Track block depth
            current_block_depth += line.matches('{').count();
            current_block_depth -= line.matches('}').count();

            // Track function endings
            if current_block_depth == 0 && in_function {
                if let Some(func_name) = current_function.take() {
                    if let Some(func_info) = self.functions.get_mut(&func_name) {
                        func_info.block_end = Some(line_num);
                    }
                }
                in_function = false;
            }

            // Analyze different Rust constructs
            if let Some(dep) = self.analyze_imports(line, line_num, path)? {
                dependencies.push(dep);
            }

            if let Some(mut deps) = self.analyze_function_definition(line, line_num, path)? {
                if !in_function {
                    in_function = true;
                    current_function = deps.first().map(|d| {
                        d.metadata
                            .description
                            .as_ref()
                            .map(|desc| desc.replace("Function definition: ", ""))
                            .unwrap_or_default()
                    });
                }
                dependencies.append(&mut deps);
            }

            if let Some(mut deps) =
                self.analyze_function_calls(line, line_num, current_function.as_ref(), path)?
            {
                dependencies.append(&mut deps);
            }

            // Handle closing blocks and scope
            if line.contains('}') && current_block_depth == 0 {
                if !self.current_scope.is_empty() {
                    self.current_scope.pop();
                }
            }
        }

        Ok(dependencies)
    }

    fn analyze_imports(
        &self,
        line: &str,
        line_num: usize,
        path: &Path,
    ) -> Result<Option<Dependency>, String> {
        if line.starts_with("use ") || line.starts_with("mod ") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 2 {
                let import_path = parts[1].trim_end_matches(';');
                return Ok(Some(Dependency {
                    source: path.to_path_buf(),
                    target: PathBuf::from(import_path),
                    dependency_type: DependencyType::Import,
                    metadata: DependencyMetadata {
                        line_number: Some(line_num),
                        description: Some(format!("Import: {}", import_path)),
                        context: Some(json!({
                            "type": if line.starts_with("use ") { "use" } else { "mod" },
                            "scope": self.current_scope.join("::"),
                            "is_public": line.starts_with("pub "),
                        })),
                        relationships: Some(vec![format!(
                            "Imported at scope: {}",
                            self.current_scope.join("::")
                        )]),
                    },
                }));
            }
        }
        Ok(None)
    }

    fn analyze_function_definition(
        &mut self,
        line: &str,
        line_num: usize,
        path: &Path,
    ) -> Result<Option<Vec<Dependency>>, String> {
        if let Some(func_info) = self.extract_function_info(line, line_num)? {
            let func_name = func_info.name.clone();
            let full_path = self
                .current_scope
                .iter()
                .chain(std::iter::once(&func_name))
                .cloned()
                .collect::<Vec<_>>()
                .join("::");

            self.functions.insert(full_path.clone(), func_info.clone());

            let mut dependencies = vec![Dependency {
                source: path.to_path_buf(),
                target: PathBuf::from(&full_path),
                dependency_type: DependencyType::FunctionDefinition,
                metadata: DependencyMetadata {
                    line_number: Some(line_num),
                    description: Some(format!("Function definition: {}", full_path)),
                    context: Some(json!({
                        "is_public": func_info.is_public,
                        "parameters": func_info.parameters,
                        "return_type": func_info.return_type,
                        "scope": self.current_scope.join("::"),
                    })),
                    relationships: Some(vec![
                        format!("Defined in scope: {}", self.current_scope.join("::")),
                        format!("Parameters: {}", func_info.parameters.len()),
                    ]),
                },
            }];

            // Add parameter type dependencies if they exist
            for param in &func_info.parameters {
                if let Some(param_type) = &param.param_type {
                    dependencies.push(Dependency {
                        source: PathBuf::from(&full_path),
                        target: PathBuf::from(param_type),
                        dependency_type: DependencyType::TypeUsage,
                        metadata: DependencyMetadata {
                            line_number: Some(line_num),
                            description: Some(format!(
                                "Parameter {} uses type {}",
                                param.name, param_type
                            )),
                            context: Some(json!({
                                "parameter": param,
                                "function": full_path,
                            })),
                            relationships: None,
                        },
                    });
                }
            }

            return Ok(Some(dependencies));
        }
        Ok(None)
    }

    fn analyze_function_calls(
        &self,
        line: &str,
        line_num: usize,
        current_function: Option<&String>,
        path: &Path,
    ) -> Result<Option<Vec<Dependency>>, String> {
        if let Some(calls) = self.extract_function_calls(line)? {
            let mut dependencies = Vec::new();

            for call in calls {
                dependencies.push(Dependency {
                    source: path.to_path_buf(),
                    target: PathBuf::from(&call),
                    dependency_type: DependencyType::FunctionCall,
                    metadata: DependencyMetadata {
                        line_number: Some(line_num),
                        description: Some(format!("Function call: {}", call)),
                        context: Some(json!({
                            "caller": current_function,
                            "scope": self.current_scope.join("::"),
                            "line_content": line.trim(),
                        })),
                        relationships: current_function
                            .as_ref()
                            .map(|caller| vec![format!("Called by function: {}", caller)]),
                    },
                });
            }

            return Ok(Some(dependencies));
        }
        Ok(None)
    }

    fn extract_function_info(
        &self,
        line: &str,
        line_num: usize,
    ) -> Result<Option<FunctionInfo>, String> {
        if !line.contains("fn ") {
            return Ok(None);
        }

        let is_public = line.starts_with("pub ");
        let fn_parts: Vec<&str> = line
            .split("fn ")
            .nth(1)
            .ok_or("Invalid function definition")?
            .split('(')
            .collect();

        if fn_parts.is_empty() {
            return Ok(None);
        }

        let name = fn_parts[0].trim().to_string();
        let params = if fn_parts.len() > 1 {
            self.extract_parameters(fn_parts[1])?
        } else {
            Vec::new()
        };

        let return_type = if line.contains("->") {
            line.split("->")
                .nth(1)
                .map(|rt| rt.trim().trim_end_matches('{').trim().to_string())
        } else {
            None
        };

        Ok(Some(FunctionInfo {
            name,
            module_path: self.current_scope.clone(),
            line_number: line_num,
            parameters: params,
            return_type,
            calls: HashSet::new(),
            is_public,
            block_start: line_num,
            block_end: None,
        }))
    }

    fn extract_parameters(&self, params_str: &str) -> Result<Vec<ParameterInfo>, String> {
        let params_end = params_str.find(')').ok_or("Invalid parameter list")?;
        let params = params_str[..params_end].split(',');

        let mut result = Vec::new();
        for param in params {
            let param = param.trim();
            if param.is_empty() {
                continue;
            }

            let parts: Vec<&str> = param.split(':').collect();
            if parts.len() >= 2 {
                let name = parts[0].trim().to_string();
                let mut param_type = parts[1].trim().to_string();
                let is_mutable = name.starts_with("mut ");

                // Clean up type if it's a reference or mutable
                if param_type.starts_with('&') {
                    param_type = param_type[1..].trim().to_string();
                }
                if param_type.starts_with("mut ") {
                    param_type = param_type[4..].trim().to_string();
                }

                result.push(ParameterInfo {
                    name: name.replace("mut ", ""),
                    param_type: Some(param_type),
                    is_mutable,
                });
            }
        }
        Ok(result)
    }

    fn extract_function_calls(&self, line: &str) -> Result<Option<Vec<String>>, String> {
        let mut calls = Vec::new();
        let mut current_pos = 0;

        while let Some(pos) = line[current_pos..].find('(') {
            let start_pos = current_pos + pos;
            let before_paren = line[..start_pos].trim_end();

            if let Some(last_word) = before_paren.split_whitespace().last() {
                // Avoid matching keywords and special cases
                if !["if", "while", "for", "match", "fn"].contains(&last_word)
                    && !last_word.contains("->")
                {
                    calls.push(last_word.to_string());
                }
            }

            current_pos = start_pos + 1;
        }

        if calls.is_empty() {
            Ok(None)
        } else {
            Ok(Some(calls))
        }
    }
}

impl CodeAnalyzer for RustAnalyzer {
    fn analyze(&self, path: &Path) -> Result<Vec<Dependency>, String> {
        let content = fs::read_to_string(path).map_err(|e| e.to_string())?;
        let mut analyzer = RustAnalyzer::new();
        analyzer.analyze_file(&content, path)
    }

    fn supported_extensions(&self) -> Vec<&'static str> {
        vec!["rs"]
    }
}
