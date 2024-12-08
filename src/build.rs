use std::env;

fn main() {
    // Get Python configuration using pyo3-build-config
    let python_config = pyo3_build_config::get().expect("Failed to get Python config");

    // Add Python library search paths
    for path in python_config.lib_dirs {
        println!("cargo:rustc-link-search=native={}", path.display());
    }

    // Link against Python library
    let python_lib = if cfg!(target_os = "windows") {
        python_config
            .lib_name
            .expect("Failed to get Python lib name")
    } else {
        format!("python{}", python_config.version.major)
    };

    println!("cargo:rustc-link-lib=dylib={}", python_lib);

    // Allow the Python symbols to be included in the final binary
    println!("cargo:rustc-cdylib-link-arg=-Wl,--no-as-needed");
}
