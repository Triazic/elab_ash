use proc_macros::{Getters};

#[derive(Getters)]
/// collection of vulkan stuff with an effectively 'static' lifetime
pub struct VKContext {
  // required for all contexts
  pub entry: ash::Entry,
  pub instance: ash::Instance,
  pub physical_device: ash::vk::PhysicalDevice,
  pub device: ash::Device,
  pub command_pool: ash::vk::CommandPool,
  pub main_queue_family_index: u32,
  pub main_queue: ash::vk::Queue,
  pub pipeline_cache: ash::vk::PipelineCache,
}