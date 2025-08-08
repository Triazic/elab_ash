use std::ffi::{CStr, CString};

pub fn int_vec_to_f32_vec(ints: &[i32]) -> Vec<f32> {
  ints.iter().map(|x| *x as f32).collect()
}

pub fn cstr(str: &str) -> CString {
  CString::new(str).expect(format!("Could not create CString from {}", str).as_str())
}

pub fn ptr_to_str(ptr: &*const i8) -> &str {
  unsafe {
    CStr::from_ptr(*ptr)
      .to_str()
      .expect("Invalid UTF-8 string")
  }
}

pub fn split_bits(n: u32) -> Vec<u32> {
  (0..32).rev()
    .map(|i| {
      let mask = 1 << i;
      let matches = n & mask == mask;
      if matches {
          1
      } else {
          0
      }
    })
    .collect()
}

/// On a little-endian system (like x86), the least significant byte comes first.
/// u32::MAX_VALUE - 1 would be 254, 255, 255, 255 in little-endian.
pub fn print_endianness() {
  let n: u32 = 0x01020304;
  let bytes = n.to_ne_bytes(); // native endian bytes

  match bytes {
      [0x01, 0x02, 0x03, 0x04] => println!("Big-endian"),
      [0x04, 0x03, 0x02, 0x01] => println!("Little-endian"),
      _ => println!("Unknown endianness"),
  }

  // Optionally: print raw bytes
  println!("Native endian bytes: {:02x?}", bytes);
}