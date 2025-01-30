mod gguf;
mod tensor;

use std::env;
use std::fs::File;

fn main() {
    let args: Vec<String> = env::args().collect();

    // Ensure a file path is provided as the first argument
    if args.len() < 2 {
        eprintln!("Usage: {} <file_path>", args[0]);
        std::process::exit(1);
    }
    let file_path = &args[1];

    let mut file = File::open(file_path).unwrap();
    let parsed_file = gguf::parse_gguf(&mut file).unwrap();
    let _ = tensor::load_tensor(&mut file, &parsed_file.tensor_infos[0]);
}
