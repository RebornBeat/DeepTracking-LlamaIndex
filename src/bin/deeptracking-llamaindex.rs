use deeptracking_llamaindex::cli::CLI;
use std::process;

fn main() {
    if let Err(e) = CLI::run() {
        eprintln!("Error: {}", e);
        process::exit(1);
    }
}
