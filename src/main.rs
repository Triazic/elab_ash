#![allow(dead_code, unused_variables, unused_imports, redundant_semicolons, unused_macros)]

use std::{ffi::CString, io::Read, panic, str::FromStr};
pub mod utils;
pub mod file_utils;
pub mod constants;
pub mod vk_context;
pub mod create_vk_context;
#[macro_use]
pub mod macros;
pub mod memory;
extern crate itertools;
extern crate strum;
use ash::vk::PipelineLayout;
use itertools::Itertools;
use utils::{cstr};
use crate::{memory::{print_flags, split_flags, split_flags_u32}, utils::{int_vec_to_f32_vec, print_endianness}};
use vk_context::*;

macro_rules! time_it {
  ($context:literal, $s:stmt) => {
      let timer = std::time::Instant::now();
      $s
      println!("{}: {:?}", $context, timer.elapsed());
  };
}

fn main() {
  let n_sets = 24 * 4; // 24 channels, WXYZ
  let vk_context = create_vk_context::create_vk_context();
  // unpack context
  let entry = vk_context.entry();
  let instance = vk_context.instance();
  let physical_device = vk_context.physical_device();
  let device = vk_context.device();
  let command_pool = vk_context.command_pool();
  let main_queue = vk_context.main_queue();
  let main_queue_family_index = vk_context.main_queue_family_index();
  let pipeline_cache = vk_context.pipeline_cache();

  // let kernel = int_vec_to_f32_vec(&[1, 2, 1, 1]);
  // let signal = int_vec_to_f32_vec(&[3, 9, 1, 2, 3, 4, 5, 6, 3, 4, 5, 6, 7, 8, 9, 8, 7, 5]);
  // let expected_1 = int_vec_to_f32_vec(&[3, 15, 22, 16, 17, 13, 18, 23, 24, 21, 22, 23, 28, 33, 38, 41, 40, 36, 25, 12, 5, 0, 0, 0]);
  // let expected_2 = int_vec_to_f32_vec(&[9, 19, 13, 17, 13, 18, 23, 24, 21, 22, 23, 28, 33, 38, 41, 40, 36, 25, 12, 5, 0, 0, 0 ]);
  let kernel = file_utils::read_floats_from_disk("assets/test_1/test_kernel.bin");
  let signal = file_utils::read_floats_from_disk("assets/test_1/test_signal.bin");
  let expected_1 = file_utils::read_floats_from_disk("assets/test_1/test_result_0.bin"); // sample index 0
  let expected_2 = file_utils::read_floats_from_disk("assets/test_1/test_result_1.bin"); // sample index 1

  // allocate buffers
  let buffer_collections = (0..n_sets).map(|i| {
    let block_samples = 1024;
    let memory_kind_flags = memory::get_memory_flags_raw(&memory::get_memory_flags_from_kind(constants::MemoryKind::BlockBufferCPU));
    let block_cpu = create_and_prepare_cpu_buffer(device, instance, physical_device, memory_kind_flags, block_samples, format!("block_cpu_{}", i).as_str());
    let memory_kind_flags = memory::get_memory_flags_raw(&memory::get_memory_flags_from_kind(constants::MemoryKind::BlockBufferCPU));
    let block_status_cpu = create_and_prepare_cpu_buffer(device, instance, physical_device, memory_kind_flags, block_samples, format!("block_status_cpu_{}", i).as_str());
    let memory_kind_flags = memory::get_memory_flags_raw(&memory::get_memory_flags_from_kind(constants::MemoryKind::BlockBufferGPU));
    let block_gpu = create_and_prepare_gpu_buffer(device, instance, physical_device, memory_kind_flags, block_samples, format!("block_gpu_{}", i).as_str());

    let signal_samples = std::cmp::max(block_samples, signal.len()); // effectively, the signal block will be padded with zeroes
    let memory_kind_flags = memory::get_memory_flags_raw(&memory::get_memory_flags_from_kind(constants::MemoryKind::SignalBufferCPU));
    let signal_cpu = create_and_prepare_cpu_buffer(device, instance, physical_device, memory_kind_flags, signal_samples, format!("signal_cpu_{}", i).as_str());
    let memory_kind_flags = memory::get_memory_flags_raw(&memory::get_memory_flags_from_kind(constants::MemoryKind::SignalBufferGPU));
    let signal_gpu = create_and_prepare_gpu_buffer(device, instance, physical_device, memory_kind_flags, signal_samples, format!("signal_gpu_{}", i).as_str());
  
    let kernel_samples = 48000 * 2; // 2 second delay
    let memory_kind_flags = memory::get_memory_flags_raw(&memory::get_memory_flags_from_kind(constants::MemoryKind::KernelBufferCPU));
    let kernel_cpu = create_and_prepare_cpu_buffer(device, instance, physical_device, memory_kind_flags, kernel_samples, format!("kernel_cpu_{}", i).as_str());
    let memory_kind_flags = memory::get_memory_flags_raw(&memory::get_memory_flags_from_kind(constants::MemoryKind::KernelBufferGPU));
    let kernel_gpu = create_and_prepare_gpu_buffer(device, instance, physical_device, memory_kind_flags, kernel_samples, format!("kernel_gpu_{}", i).as_str());
  
    // populate the signal buffer
    write_floats(signal_cpu.mapped_memory, &signal);
  
    // populate the kernel buffer
    write_floats(kernel_cpu.mapped_memory, &kernel);

    // populate the block status buffer with some non-zero value
    let mut seed_status = vec![0.0; block_status_cpu.samples as usize];
    seed_status.fill(-1.0);
    write_floats(block_status_cpu.mapped_memory, &seed_status);

    PreparedBuffers {
      signal_cpu,
      kernel_cpu,
      block_cpu,
      block_status_cpu,
      signal_gpu,
      kernel_gpu,
      block_gpu,
    }
  }).collect_vec();

  // copy the cpu buffers to the gpu buffers
  time_it!("copy gpu buffers", {
    let commands = buffer_collections.iter().enumerate().flat_map(|(i, buffers)| {
      let a = make_cpu_gpu_copy_command(device, main_queue, command_pool, &buffers.signal_cpu, &buffers.signal_gpu);
      let b = make_cpu_gpu_copy_command(device, main_queue, command_pool, &buffers.kernel_cpu, &buffers.kernel_gpu);
      [a, b]
    }).collect_vec();
    let fence = submit(device, main_queue, &commands);
    unsafe { device.wait_for_fences(&[fence], true, std::u64::MAX).expect("Failed to wait for fence"); }
    unsafe { device.destroy_fence(fence, None); }
  });

  // make compute pipeline
  let spirv_code = create_spirv_code();
  let shader_module = create_shader_module(device, &spirv_code);
  let descriptor_set_layout = create_descriptor_set_layout(device);
  let pipeline_layout = create_pipeline_layout(device, &[descriptor_set_layout]);
  let descriptor_pool = create_descriptor_pool(device, n_sets);
  let descriptor_sets = (0..n_sets).map(|_| {
    allocate_descriptors(device, &descriptor_pool, &[descriptor_set_layout])
  }).collect_vec();
  descriptor_sets.iter().enumerate().for_each(|(i, descriptor_set)| {
    let buffers = &buffer_collections[i];
    update_descriptor_sets_cpu(device, descriptor_set, &buffers.signal_cpu, &buffers.kernel_cpu, &buffers.block_cpu, &buffers.block_status_cpu);
    // update_descriptor_sets_gpu(device, descriptor_set, &buffers.signal_gpu, &buffers.kernel_gpu, &buffers.block_gpu);
    // update_descriptor_sets_gpu_and_cpu(device, descriptor_set, &buffers.signal_gpu, &buffers.kernel_gpu, &buffers.block_cpu);
  });
  let pipelines = (0..n_sets).map(|_| {
    create_pipeline(device, command_pool, pipeline_cache, &pipeline_layout, &shader_module)
  }).collect_vec();

  // make command buffers
  let command_buffers = (0..n_sets).map(|_| {
    create_command_buffer(device, command_pool)
  }).collect_vec();

  let n_repetitions = 5;
  for rep in 0..n_repetitions {
    let start_time = std::time::Instant::now();

    let is_even = rep % 2 == 0;
    let expected = if is_even { &expected_1 } else { &expected_2 };
    let signal_index = if is_even { 0 } else { 1 };

    // time_it!("record command buffers", 
      command_buffers.iter().enumerate().for_each(|(i, command_buffer)| {
        let buffers = &buffer_collections[i];
        record_command_buffer(
          device, command_buffer, *main_queue_family_index,
          &buffers.signal_cpu, &buffers.kernel_cpu, &buffers.block_cpu, 
          &buffers.signal_gpu, &buffers.kernel_gpu, &buffers.block_gpu, 
          &pipelines[i], &pipeline_layout, &[descriptor_sets[i]], signal_index
        );
      })
    // )
    ;
  
    // submit and wait for its completion
    // time_it!("submit", 
      let fence = submit(device, main_queue, &command_buffers)
    // )
    ;

    // time_it!("wait for fences", 
      unsafe { device.wait_for_fences(&[fence], true, std::u64::MAX).expect("Failed to wait for fence"); }
    // )
    ;
    unsafe { device.destroy_fence(fence, None); }

    println!("computation complete: {:?}", std::time::Instant::now() - start_time);


    buffer_collections.iter().enumerate().for_each(|(i, buffers)| {
      // assert that the STATUS buffer is what we expect (ie. did the shader actualy succeed)
      let mut expected_status = vec![0.0; buffers.block_status_cpu.samples as usize];
      expected_status.fill(signal_index as f32);
      assert_floats(buffers.block_status_cpu.mapped_memory, &expected_status, "STATUS ERROR. SHADER EXECUTION FAILED");

      // assert that the actual result is what we expect
      assert_floats(buffers.block_cpu.mapped_memory, &expected, "RESULT ERROR. SHADER EXECUTION PRODUCED WRONG VALUE");
    });
  }
 
  unsafe { device.device_wait_idle().expect("Failed to wait for device to become idle"); }
  unsafe { device.destroy_command_pool(*command_pool, None); }
  buffer_collections.iter().for_each(|buffers| {
    unsafe { device.free_memory(buffers.signal_cpu.memory_allocation, None); }
    unsafe { device.destroy_buffer(buffers.signal_cpu.buffer, None); }
    unsafe { device.free_memory(buffers.kernel_cpu.memory_allocation, None); }
    unsafe { device.destroy_buffer(buffers.kernel_cpu.buffer, None); }
    unsafe { device.free_memory(buffers.block_cpu.memory_allocation, None); }
    unsafe { device.destroy_buffer(buffers.block_cpu.buffer, None); }
    unsafe { device.free_memory(buffers.block_status_cpu.memory_allocation, None); }
    unsafe { device.destroy_buffer(buffers.block_status_cpu.buffer, None); }
    unsafe { device.free_memory(buffers.signal_gpu.memory_allocation, None); }
    unsafe { device.destroy_buffer(buffers.signal_gpu.buffer, None); }
    unsafe { device.free_memory(buffers.kernel_gpu.memory_allocation, None); }
    unsafe { device.destroy_buffer(buffers.kernel_gpu.buffer, None); }
    unsafe { device.free_memory(buffers.block_gpu.memory_allocation, None); }
    unsafe { device.destroy_buffer(buffers.block_gpu.buffer, None); }
  });
  unsafe { device.destroy_shader_module(shader_module, None); }
  unsafe { device.destroy_descriptor_set_layout(descriptor_set_layout, None); }
  unsafe { device.destroy_pipeline_cache(*pipeline_cache, None); }
  unsafe { device.destroy_pipeline_layout(pipeline_layout, None); }
  unsafe { device.destroy_descriptor_pool(descriptor_pool, None); }
  pipelines.iter().for_each(|pipeline| {
    unsafe { device.destroy_pipeline(*pipeline, None); }
  });
  unsafe { device.destroy_device(None); }
  unsafe { instance.destroy_instance(None); }
  println!("Finished");
}

struct PreparedBuffers {
  signal_cpu: PreparedCPUBuffer,
  kernel_cpu: PreparedCPUBuffer,
  block_cpu: PreparedCPUBuffer,
  block_status_cpu: PreparedCPUBuffer,
  signal_gpu: PreparedGPUBuffer,
  kernel_gpu: PreparedGPUBuffer,
  block_gpu: PreparedGPUBuffer,
}
struct PreparedCPUBuffer {
  buffer: ash::vk::Buffer,
  memory_allocation: ash::vk::DeviceMemory,
  mapped_memory: *mut std::ffi::c_void,
  len: u64,
  samples: u64,
}
fn create_and_prepare_cpu_buffer(device: &ash::Device, instance: &ash::Instance, physical_device: &ash::vk::PhysicalDevice, memory_kind_flags: u32, n_floats: usize, name: &str) -> PreparedCPUBuffer {
  let buffer_samples = n_floats;
  let buffer_len = (buffer_samples * std::mem::size_of::<f32>()) as ash::vk::DeviceSize;
  let usage = 
    ash::vk::BufferUsageFlags::TRANSFER_SRC | 
    ash::vk::BufferUsageFlags::TRANSFER_DST | 
    ash::vk::BufferUsageFlags::STORAGE_BUFFER
  ;
  let buffer = create_buffer(device, buffer_len, usage);
  // set_object_name(instance, device, buffer, name);
  let requirements = get_buffer_memory_requirements(device, &buffer);
  let memory_type_bits = requirements.memory_type_bits;
  let memory_type_index = 
    memory::get_memory_type_index_raw(instance, physical_device, memory_kind_flags, memory_type_bits)
    .expect("no suitable memory type index found");
  let memory_allocation = allocate_memory(device, memory_type_index, requirements.size);
  let offset = 0;
  bind_buffer_memory(device, &buffer, &memory_allocation, offset);

  // map the memory so the CPU can consume it
  let mapped_memory = map_memory(device, &memory_allocation);

  PreparedCPUBuffer {
    buffer,
    memory_allocation,
    mapped_memory,    
    len: buffer_len,
    samples: buffer_samples as u64,
  }
}

struct PreparedGPUBuffer {
  buffer: ash::vk::Buffer,
  memory_allocation: ash::vk::DeviceMemory,
  len: u64,
  samples: u64,
}
fn create_and_prepare_gpu_buffer(device: &ash::Device, instance: &ash::Instance, physical_device: &ash::vk::PhysicalDevice, memory_kind_flags: u32, n_floats: usize, name: &str) -> PreparedGPUBuffer {
  let buffer_samples = n_floats;
  let buffer_len = (buffer_samples * std::mem::size_of::<f32>()) as ash::vk::DeviceSize;
  let usage = 
    ash::vk::BufferUsageFlags::TRANSFER_SRC | 
    ash::vk::BufferUsageFlags::TRANSFER_DST | 
    ash::vk::BufferUsageFlags::STORAGE_BUFFER
  ;
  let buffer = create_buffer(device, buffer_len, usage);
  // set_object_name(instance, device, buffer, name);
  let requirements = get_buffer_memory_requirements(device, &buffer);
  let memory_type_bits = requirements.memory_type_bits;
  let memory_type_index = 
    memory::get_memory_type_index_raw(instance, physical_device, memory_kind_flags, memory_type_bits)
    .expect("no suitable memory type index found");
  let memory_allocation = allocate_memory(device, memory_type_index, requirements.size);
  let offset = 0;
  bind_buffer_memory(device, &buffer, &memory_allocation, offset);

  PreparedGPUBuffer {
    buffer,
    memory_allocation,
    len: buffer_len,
    samples: buffer_samples as u64,
  }
}

/// just a handle. not backed with memory
fn create_buffer(device: &ash::Device, buffer_size: u64, usage: ash::vk::BufferUsageFlags) -> ash::vk::Buffer {
  let flags = ash::vk::BufferCreateFlags::empty();
  let sharing_mode = ash::vk::SharingMode::EXCLUSIVE; // used in one queue
  let create_info = ash::vk::BufferCreateInfo::default()
    .flags(flags) 
    .size(buffer_size)
    .usage(usage)
    .sharing_mode(sharing_mode);

  let buffer = unsafe { device.create_buffer(&create_info, None).expect("Could not create Vulkan buffer") };
  buffer
}

fn allocate_memory(device: &ash::Device, memory_type_index: u32, size: u64) -> ash::vk::DeviceMemory {
  let info = ash::vk::MemoryAllocateInfo::default()
    .allocation_size(size)
    .memory_type_index(memory_type_index);

  let memory = unsafe { device.allocate_memory(&info, None).expect("Could not allocate Vulkan memory") };
  memory
}

fn get_heap_usage(instance: &ash::Instance, physical_device: &ash::vk::PhysicalDevice) -> [u64; ash::vk::MAX_MEMORY_HEAPS] {
  let mut memory_budget_props = ash::vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
  
  let mut memory_props2 = ash::vk::PhysicalDeviceMemoryProperties2::default()
    .push_next(&mut memory_budget_props);
  
  unsafe { instance.get_physical_device_memory_properties2(*physical_device, &mut memory_props2); }
  
  let heap_usage = memory_budget_props.heap_usage;
  return heap_usage;
}

fn print_heap_usage(instance: &ash::Instance, physical_device: &ash::vk::PhysicalDevice) -> () {
  let heap_usage = get_heap_usage(instance, physical_device);
  dbg!(heap_usage);
}

fn get_buffer_memory_requirements(device: &ash::Device, buffer: &ash::vk::Buffer) -> ash::vk::MemoryRequirements {
  let reqs = unsafe { device.get_buffer_memory_requirements(*buffer) };
  return reqs;
}

fn bind_buffer_memory(device: &ash::Device, buffer: &ash::vk::Buffer, memory_allocation: &ash::vk::DeviceMemory, offset: u64) -> () {
  unsafe { device.bind_buffer_memory(*buffer, *memory_allocation, offset).expect("failed to bind buffer memory") }
}

fn create_command_buffer(device: &ash::Device, command_pool: &ash::vk::CommandPool) -> ash::vk::CommandBuffer {
  let command_buffer_allocate_info = ash::vk::CommandBufferAllocateInfo::default()
    .command_buffer_count(1)
    .command_pool(*command_pool)
    .level(ash::vk::CommandBufferLevel::PRIMARY);
  let command_buffers = unsafe { device.allocate_command_buffers(&command_buffer_allocate_info).expect("failed to allocate command buffer") };
  return *command_buffers.get(0).expect("no command buffers created?");
}

fn create_spirv_code() -> Vec<u32> {
  let path = std::path::Path::new("shaders/theshader.spv");
  if !path.exists() {
    panic!("shader file {:?} does not exist", path); 
  }
  let spirv_bytes = std::fs::read(path).expect("failed to read shader file");

  // Ensure alignment: SPIR-V should be a multiple of 4 bytes
  if spirv_bytes.len() % 4 != 0 {
    panic!("SPIR-V binary size is not a multiple of 4");
  }

  // SAFELY reinterpret bytes as u32
  let spirv_u32: Vec<u32> = spirv_bytes
    .chunks_exact(4)
    .map(|chunk| u32::from_le_bytes(chunk.try_into().expect("failed try_into on bytes shader chunk")))
    .collect();

  spirv_u32
}

fn create_shader_module(device: &ash::Device, spirv_code: &[u32]) -> ash::vk::ShaderModule {
  let shader_module_info = 
    ash::vk::ShaderModuleCreateInfo::default()
    .code(spirv_code);
  let shader_module = unsafe { device.create_shader_module(&shader_module_info, None).expect("failed to create shader module") };
  shader_module
}

fn create_descriptor_pool(device: &ash::Device, max_sets: u32) -> ash::vk::DescriptorPool {
  let descriptor_pool_size = ash::vk::DescriptorPoolSize::default()
    .descriptor_count(max_sets * 4)
    .ty(ash::vk::DescriptorType::STORAGE_BUFFER);
  let descriptor_pool_sizes = [descriptor_pool_size];
  let descriptor_pool_info = ash::vk::DescriptorPoolCreateInfo::default()
    .pool_sizes(&descriptor_pool_sizes)
    .max_sets(max_sets);
  let descriptor_pool = unsafe { device.create_descriptor_pool(&descriptor_pool_info, None).expect("failed to create descriptor pool") };
  descriptor_pool
}

fn allocate_descriptors(device: &ash::Device, descriptor_pool: &ash::vk::DescriptorPool, descriptor_set_layouts: &[ash::vk::DescriptorSetLayout]) -> ash::vk::DescriptorSet {
  let descriptor_set_allocate_info = ash::vk::DescriptorSetAllocateInfo::default()
    .descriptor_pool(*descriptor_pool)
    .set_layouts(descriptor_set_layouts);
  let descriptor_set = unsafe { device.allocate_descriptor_sets(&descriptor_set_allocate_info).expect("failed to allocate descriptor set") };
  descriptor_set[0]
}

fn create_descriptor_set_layout(device: &ash::Device) -> ash::vk::DescriptorSetLayout {
  let signal_binding = 
    ash::vk::DescriptorSetLayoutBinding::default()
    .binding(0)
    .descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER)
    .descriptor_count(1)
    .stage_flags(ash::vk::ShaderStageFlags::COMPUTE);
  let kernel_binding = 
    ash::vk::DescriptorSetLayoutBinding::default()
    .binding(1)
    .descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER)
    .descriptor_count(1)
    .stage_flags(ash::vk::ShaderStageFlags::COMPUTE);
  let block_binding = 
    ash::vk::DescriptorSetLayoutBinding::default()
    .binding(2)
    .descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER)
    .descriptor_count(1)
    .stage_flags(ash::vk::ShaderStageFlags::COMPUTE);
  let block_status_binding = 
    ash::vk::DescriptorSetLayoutBinding::default()
    .binding(3)
    .descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER)
    .descriptor_count(1)
    .stage_flags(ash::vk::ShaderStageFlags::COMPUTE);
  let bindings = [signal_binding, kernel_binding, block_binding, block_status_binding];
  let create_info = 
    ash::vk::DescriptorSetLayoutCreateInfo::default()
    .bindings(&bindings);
  let descriptor_set_layout = unsafe { device.create_descriptor_set_layout(&create_info, None).expect("failed to create descriptor set layout") };
  descriptor_set_layout
}

fn create_pipeline_layout(device: &ash::Device, descriptor_set_layouts: &[ash::vk::DescriptorSetLayout]) -> ash::vk::PipelineLayout {
  let push_constant_range = ash::vk::PushConstantRange::default()
    .stage_flags(ash::vk::ShaderStageFlags::COMPUTE)
    .offset(0)
    .size(std::mem::size_of::<PushConstants>() as u32)
  ;
  let push_constant_ranges = [push_constant_range];
  let create_info = 
    ash::vk::PipelineLayoutCreateInfo::default()
    .set_layouts(descriptor_set_layouts)
    .push_constant_ranges(&push_constant_ranges)
    .flags(ash::vk::PipelineLayoutCreateFlags::default())
    ;
  let pipeline_layout = unsafe { device.create_pipeline_layout(&create_info, None).expect("failed to create pipeline layout") };
  pipeline_layout
}

fn update_descriptor_sets_cpu(
  device: &ash::Device, descriptor_set: &ash::vk::DescriptorSet, 
  signal_buffer: &PreparedCPUBuffer, kernel_buffer: &PreparedCPUBuffer, block_buffer: &PreparedCPUBuffer, block_buffer_status: &PreparedCPUBuffer
) {
  let signal_buffer_info = ash::vk::DescriptorBufferInfo::default()
    .buffer(signal_buffer.buffer)
    .offset(0)
    .range(signal_buffer.len);
  let signal_buffer_infos = [signal_buffer_info];
  let signal_buffer_descriptor_write = ash::vk::WriteDescriptorSet::default()
    .dst_set(*descriptor_set)
    .dst_binding(0)
    .dst_array_element(0)
    .descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER)
    .buffer_info(&signal_buffer_infos)
    ;
  let kernel_buffer_info = ash::vk::DescriptorBufferInfo::default()
    .buffer(kernel_buffer.buffer)
    .offset(0)
    .range(kernel_buffer.len);
  let kernel_buffer_infos = [kernel_buffer_info];
  let kernel_buffer_descriptor_write = ash::vk::WriteDescriptorSet::default()
    .dst_set(*descriptor_set)
    .dst_binding(1)
    .dst_array_element(0)
    .descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER)
    .buffer_info(&kernel_buffer_infos)
    ;
  let block_buffer_info = ash::vk::DescriptorBufferInfo::default()
    .buffer(block_buffer.buffer)
    .offset(0)
    .range(block_buffer.len);
  let block_buffer_infos = [block_buffer_info];
  let block_buffer_descriptor_write = ash::vk::WriteDescriptorSet::default()
    .dst_set(*descriptor_set)
    .dst_binding(2)
    .dst_array_element(0)
    .descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER)
    .buffer_info(&block_buffer_infos)
    ;
  let block_buffer_status_info = ash::vk::DescriptorBufferInfo::default()
    .buffer(block_buffer_status.buffer)
    .offset(0)
    .range(block_buffer_status.len);
  let block_buffer_status_infos = [block_buffer_status_info];
  let block_buffer_status_descriptor_write = ash::vk::WriteDescriptorSet::default()
    .dst_set(*descriptor_set)
    .dst_binding(3)
    .dst_array_element(0)
    .descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER)
    .buffer_info(&block_buffer_status_infos)
    ;
  let descriptor_writes = [signal_buffer_descriptor_write, kernel_buffer_descriptor_write, block_buffer_descriptor_write, block_buffer_status_descriptor_write];
  unsafe { device.update_descriptor_sets(&descriptor_writes, &[]) };
}

fn update_descriptor_sets_gpu(
  device: &ash::Device, descriptor_set: &ash::vk::DescriptorSet, 
  signal_buffer: &PreparedGPUBuffer, kernel_buffer: &PreparedGPUBuffer, block_buffer: &PreparedGPUBuffer
) {
  let signal_buffer_info = ash::vk::DescriptorBufferInfo::default()
    .buffer(signal_buffer.buffer)
    .offset(0)
    .range(signal_buffer.len);
  let signal_buffer_infos = [signal_buffer_info];
  let signal_buffer_descriptor_write = ash::vk::WriteDescriptorSet::default()
    .dst_set(*descriptor_set)
    .dst_binding(0)
    .dst_array_element(0)
    .descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER)
    .buffer_info(&signal_buffer_infos)
    ;
  let kernel_buffer_info = ash::vk::DescriptorBufferInfo::default()
    .buffer(kernel_buffer.buffer)
    .offset(0)
    .range(kernel_buffer.len);
  let kernel_buffer_infos = [kernel_buffer_info];
  let kernel_buffer_descriptor_write = ash::vk::WriteDescriptorSet::default()
    .dst_set(*descriptor_set)
    .dst_binding(1)
    .dst_array_element(0)
    .descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER)
    .buffer_info(&kernel_buffer_infos)
    ;
  let block_buffer_info = ash::vk::DescriptorBufferInfo::default()
    .buffer(block_buffer.buffer)
    .offset(0)
    .range(block_buffer.len);
  let block_buffer_infos = [block_buffer_info];
  let block_buffer_descriptor_write = ash::vk::WriteDescriptorSet::default()
    .dst_set(*descriptor_set)
    .dst_binding(2)
    .dst_array_element(0)
    .descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER)
    .buffer_info(&block_buffer_infos)
    ;
  let descriptor_writes = [signal_buffer_descriptor_write, kernel_buffer_descriptor_write, block_buffer_descriptor_write];
  unsafe { device.update_descriptor_sets(&descriptor_writes, &[]) };
}

fn update_descriptor_sets_gpu_and_cpu(
  device: &ash::Device, descriptor_set: &ash::vk::DescriptorSet, 
  signal_buffer: &PreparedGPUBuffer, kernel_buffer: &PreparedGPUBuffer, block_buffer: &PreparedCPUBuffer
) {
  let signal_buffer_info = ash::vk::DescriptorBufferInfo::default()
    .buffer(signal_buffer.buffer)
    .offset(0)
    .range(signal_buffer.len);
  let signal_buffer_infos = [signal_buffer_info];
  let signal_buffer_descriptor_write = ash::vk::WriteDescriptorSet::default()
    .dst_set(*descriptor_set)
    .dst_binding(0)
    .dst_array_element(0)
    .descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER)
    .buffer_info(&signal_buffer_infos)
    ;
  let kernel_buffer_info = ash::vk::DescriptorBufferInfo::default()
    .buffer(kernel_buffer.buffer)
    .offset(0)
    .range(kernel_buffer.len);
  let kernel_buffer_infos = [kernel_buffer_info];
  let kernel_buffer_descriptor_write = ash::vk::WriteDescriptorSet::default()
    .dst_set(*descriptor_set)
    .dst_binding(1)
    .dst_array_element(0)
    .descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER)
    .buffer_info(&kernel_buffer_infos)
    ;
  let block_buffer_info = ash::vk::DescriptorBufferInfo::default()
    .buffer(block_buffer.buffer)
    .offset(0)
    .range(block_buffer.len);
  let block_buffer_infos = [block_buffer_info];
  let block_buffer_descriptor_write = ash::vk::WriteDescriptorSet::default()
    .dst_set(*descriptor_set)
    .dst_binding(2)
    .dst_array_element(0)
    .descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER)
    .buffer_info(&block_buffer_infos)
    ;
  let descriptor_writes = [signal_buffer_descriptor_write, kernel_buffer_descriptor_write, block_buffer_descriptor_write];
  unsafe { device.update_descriptor_sets(&descriptor_writes, &[]) };
}

fn create_pipeline(device: &ash::Device, command_pool: &ash::vk::CommandPool, pipeline_cache: &ash::vk::PipelineCache, pipeline_layout: &ash::vk::PipelineLayout, shader_module: &ash::vk::ShaderModule) -> ash::vk::Pipeline {
  let entry_point = cstr("main");
  let stage_create_info = 
    ash::vk::PipelineShaderStageCreateInfo::default()
    .stage(ash::vk::ShaderStageFlags::COMPUTE)
    .name(entry_point.as_c_str())
    .module(*shader_module)
    ;
  let create_info = 
    ash::vk::ComputePipelineCreateInfo::default()
    .stage(stage_create_info)
    .layout(*pipeline_layout)
    ;
  let create_infos = [create_info];
  let pipelines = unsafe {
    device.create_compute_pipelines(*pipeline_cache, &create_infos, None).expect("failed to create compute pipeline")
  };
  pipelines[0]
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
struct PushConstants {
  signal_len: u32,
  kernel_len: u32,
  block_len: u32,
  signal_index: u32,
}

fn record_command_buffer(
  device: &ash::Device, command_buffer: &ash::vk::CommandBuffer, queue_family_index: u32,
  signal_buffer_cpu: &PreparedCPUBuffer, kernel_buffer_cpu: &PreparedCPUBuffer, block_buffer_cpu: &PreparedCPUBuffer,
  signal_buffer_gpu: &PreparedGPUBuffer, kernel_buffer_gpu: &PreparedGPUBuffer, block_buffer_gpu: &PreparedGPUBuffer,
  pipeline: &ash::vk::Pipeline, pipeline_layout: &ash::vk::PipelineLayout, descriptor_sets: &[ash::vk::DescriptorSet],
  signal_index: u32,
) -> () {
  let begin_flags = ash::vk::CommandBufferUsageFlags::default();
  let begin_create_info = ash::vk::CommandBufferBeginInfo::default()
    .flags(begin_flags);
  unsafe { 
    device
    .begin_command_buffer(*command_buffer, &begin_create_info)
    .expect("failed to begin command buffer");

    device.cmd_bind_pipeline(
      *command_buffer, 
      ash::vk::PipelineBindPoint::COMPUTE, 
      *pipeline
    );

    device.cmd_bind_descriptor_sets(
      *command_buffer, 
      ash::vk::PipelineBindPoint::COMPUTE, 
      *pipeline_layout, 
      0,  // first set
      descriptor_sets, 
      &[] // dynamic offsets
    );

    let constants = PushConstants {
      signal_len: signal_buffer_gpu.samples as u32,
      kernel_len: kernel_buffer_gpu.samples as u32,
      block_len: block_buffer_gpu.samples as u32,
      signal_index,
    };

    // copy is immediate
    device.cmd_push_constants(
      *command_buffer, 
      *pipeline_layout, 
      ash::vk::ShaderStageFlags::COMPUTE, 
      0, 
      std::slice::from_raw_parts(
        (&constants as *const PushConstants) as *const u8,
        std::mem::size_of::<PushConstants>(),
    ),
    );

    // dispatch enough workgroups to cover the block buffer
    let n_floats = block_buffer_cpu.samples;
    let workgroup_size = 64;
    let num_workgroups = (n_floats + workgroup_size - 1) / workgroup_size;

    // execute work on the GPU buffers
    device.cmd_dispatch(*command_buffer, num_workgroups as u32, 1, 1);

    // wait for pipeline completion
    if false {
      let barrier = ash::vk::BufferMemoryBarrier::default()
      .src_access_mask(ash::vk::AccessFlags::SHADER_WRITE)
      .dst_access_mask(ash::vk::AccessFlags::HOST_READ)
      .src_queue_family_index(queue_family_index)
      .dst_queue_family_index(queue_family_index)
      .buffer(block_buffer_cpu.buffer)
      .offset(0)
      .size(ash::vk::WHOLE_SIZE);

      device.cmd_pipeline_barrier(
        *command_buffer,
        ash::vk::PipelineStageFlags::COMPUTE_SHADER,
        ash::vk::PipelineStageFlags::HOST,
        ash::vk::DependencyFlags::empty(),
        &[],
        &[barrier],
        &[],
      );
    }

    // copy GPU block back to CPU block
    if false {
      device.cmd_copy_buffer(
        *command_buffer, 
        block_buffer_gpu.buffer, 
        block_buffer_cpu.buffer, 
        &[
          ash::vk::BufferCopy::default()
            .dst_offset(0)
            .src_offset(0)
            .size(block_buffer_gpu.len)
        ]
      );
    }

    device
    .end_command_buffer(*command_buffer)
    .expect("failed to end command buffer");
  };
}

fn make_cpu_gpu_copy_command(device: &ash::Device, queue: &ash::vk::Queue, command_pool: &ash::vk::CommandPool, src: &PreparedCPUBuffer, dst: &PreparedGPUBuffer) -> ash::vk::CommandBuffer {
  // make a command buffer
  let command_buffer = create_command_buffer(device, command_pool);
  let begin_flags = ash::vk::CommandBufferUsageFlags::default();
  let begin_create_info = ash::vk::CommandBufferBeginInfo::default()
    .flags(begin_flags);

  // record the command buffers
  unsafe { 
    device
    .begin_command_buffer(command_buffer, &begin_create_info)
    .expect("failed to begin command buffer");

    device.cmd_copy_buffer(
      command_buffer, 
      src.buffer, 
      dst.buffer, 
      &[
        ash::vk::BufferCopy::default()
          .dst_offset(0)
          .src_offset(0)
          .size(src.len)
      ]
    );

    device
    .end_command_buffer(command_buffer)
    .expect("failed to end command buffer");
  };

  command_buffer
}

fn submit(device: &ash::Device, queue: &ash::vk::Queue, command_buffers: &[ash::vk::CommandBuffer]) -> ash::vk::Fence {
  let submit_info = ash::vk::SubmitInfo::default()
    .command_buffers(&command_buffers);

  let fence = unsafe { device.create_fence(&ash::vk::FenceCreateInfo::default(), None).expect("failed to create fence") };

  unsafe { device.queue_submit(*queue, &[submit_info], fence).expect("failed to submit to queue"); }

  fence
}

fn map_memory(device: &ash::Device, memory_allocation: &ash::vk::DeviceMemory) -> *mut std::ffi::c_void {
  let flags = ash::vk::MemoryMapFlags::default();
  let pointer = unsafe { device.map_memory(*memory_allocation, 0, ash::vk::WHOLE_SIZE, flags).expect("failed to map memory") };
  pointer
}

fn print_buffer(mapped_memory: *mut std::ffi::c_void, buffer_size: u64) -> () {
  unsafe {
    // Cast the void pointer to a u8 pointer
    let byte_ptr = mapped_memory as *mut u8;

    // Create a slice from the raw pointer
    let slice = std::slice::from_raw_parts(byte_ptr, buffer_size as usize);

    // Now you can use the slice safely
    dbg!(&slice, slice.len());
  }
}

fn write_floats(mapped_memory: *mut std::ffi::c_void, floats: &[f32]) {
  unsafe {
    let byte_ptr = mapped_memory as *mut f32; // underlying memory is [u8] but cosplays here as [f32]
    let slice = std::slice::from_raw_parts_mut(byte_ptr, floats.len());
    slice.copy_from_slice(floats);
  }
}

fn read_floats(mapped_memory: *mut std::ffi::c_void, n_floats: u64) -> Vec<f32> {
  unsafe {
    // Cast the void pointer to a u8 pointer
    let byte_ptr = mapped_memory as *mut f32;

    // Create a slice from the raw pointer
    let slice = std::slice::from_raw_parts(byte_ptr, n_floats as usize);

    let mut acc = Vec::new();
    for i in 0..slice.len() {
      acc.push(slice[i] as f32);
    }
    acc
  }
}

fn assert_floats(mapped_memory: *mut std::ffi::c_void, expected: &[f32], ident: &str) -> () {
  unsafe {
    // Cast the void pointer to a u8 pointer
    let byte_ptr = mapped_memory as *mut f32;

    // Create a slice from the raw pointer
    let slice = std::slice::from_raw_parts(byte_ptr, expected.len());

    for i in 0..slice.len() {
      let expected = expected[i];
      let actual = slice[i];
      assert_equal_ident(expected, actual, 0.001, ident);
    }
  }
}

fn assert_equal(a: f32, b: f32, tolerance: f32) -> () {
  let equal = (a - b).abs() < tolerance;
  if !equal {
    panic!("{} and {} are not equal", a, b);
  }
}

fn assert_equal_ident(a: f32, b: f32, tolerance: f32, ident: &str) -> () {
  let equal = (a - b).abs() < tolerance;
  if !equal {
    panic!("{}. {} and {} are not equal", ident, a, b);
  }
}

fn set_object_name<H: ash::vk::Handle>(
  instance: &ash::Instance,
  device: &ash::Device,
  object_handle: H,
  name: &str,
) -> () {
  use ash::vk;
  use std::ffi::CString;

  let debug_utils_loader = ash::ext::debug_utils::Device::new(&instance, &device);
  let name_cstr = CString::new(name).unwrap();
  let name_info = ash::vk::DebugUtilsObjectNameInfoEXT::default()
    .object_handle(object_handle)
    .object_name(&name_cstr)
    ;
  unsafe {
    debug_utils_loader
      .set_debug_utils_object_name(&name_info)
      .expect("Failed to set Vulkan object name");
  }
}