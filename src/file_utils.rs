use std::io::{Read, Write};

/// Writes a slice of f32 floats to disk as raw bytes (little-endian).
pub fn write_floats_to_disk(path: &str, floats: &Vec<f32>) {
  let path = std::path::Path::new(path);

  // Create parent directories if they don't exist
  if let Some(parent) = path.parent() {
    std::fs::create_dir_all(parent)
      .unwrap_or_else(|e| panic!("Failed to create directories for '{}': {}", &path.display(), e));
  }

  let mut file = std::fs::File::create(&path)
    .unwrap_or_else(|e| panic!("Failed to create file '{}': {}", &path.display(), e));

  let bytes = unsafe {
    std::slice::from_raw_parts(
      floats.as_ptr() as *const u8,
      floats.len() * std::mem::size_of::<f32>(),
    )
  };

  file.write_all(bytes)
    .unwrap_or_else(|e| panic!("Failed to write to file '{}': {}", &path.display(), e));
}

/// Reads a file of raw f32 values (little-endian) and returns a Vec<f32>.
pub fn read_floats_from_disk(path: &str) -> Vec<f32> {
  let mut file = std::fs::File::open(path)
    .unwrap_or_else(|e| panic!("Failed to open file '{}': {}", path, e));

  let mut buffer = Vec::new();
  file.read_to_end(&mut buffer)
    .unwrap_or_else(|e| panic!("Failed to read file '{}': {}", path, e));

  assert!(buffer.len() % 4 == 0, "File size is not a multiple of 4 (f32 size)");

  let num_floats = buffer.len() / 4;

let floats: Vec<f32> = buffer
    .chunks_exact(4)
    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    .collect();

  floats
}