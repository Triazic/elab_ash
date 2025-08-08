fn main() {
  let shader_file_path = "shaders/theshader.comp";
  let shader_file_path_spv = "shaders/theshader.spv";
  let glslang_validator_installed = get_if_glslang_validator_exists();
  if !glslang_validator_installed {
    println!("warn: no glslangValidator found. shaders will not be recompiled");
    return;
  }
  // invoke glsllangValidator
  std::process::Command::new("glslangValidator")
    .arg("-V")
    .arg("--target-env")
    .arg("vulkan1.1")
    .arg("-o")
    .arg(shader_file_path_spv)
    .arg(shader_file_path)
    .output()
    .expect("failed to execute glslangValidator");
}


fn get_if_glslang_validator_exists() -> bool {
  std::process::Command::new("glslangValidator")
    .arg("--version")
    .output()
    .map(|output| output.status.success())
    .unwrap_or(false)
}