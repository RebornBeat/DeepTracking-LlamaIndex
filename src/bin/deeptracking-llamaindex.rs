use deeptracking_llamaindex::cli::CLI;
use std::process;

fn main() {
    pyo3::prepare_freethreaded_python();

    if let Err(e) = CLI::run() {
        eprintln!("Error: {}", e);
        process::exit(1);
    }
}
