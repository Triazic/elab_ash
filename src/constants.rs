pub static API_VERSION: u32 = ash::vk::API_VERSION_1_1;

// pub static REQUIRED_INSTANCE_LAYERS: [&str; 1] = ["VK_LAYER_KHRONOS_validation"];
pub static REQUIRED_INSTANCE_LAYERS: [&str; 0] = [];

// pub static REQUIRED_DEVICE_LAYERS: [&str; 1] = ["VK_LAYER_KHRONOS_validation"];
pub static REQUIRED_DEVICE_LAYERS: [&str; 0] = [];

// pub static REQUIRED_DEVICE_EXTENSIONS: [&str; 1] = ["VK_EXT_memory_budget"];
pub static REQUIRED_DEVICE_EXTENSIONS: [&str; 0] = [];

// pub static REQUIRED_INSTANCE_EXTENSIONS: [&str; 1] = ["VK_EXT_debug_utils"];
pub static REQUIRED_INSTANCE_EXTENSIONS: [&str; 0] = [];

#[derive(Debug, strum_macros::EnumIter)]
pub enum MemoryKind {
    SignalBufferCPU,
    KernelBufferCPU,
    BlockBufferCPU,
    SignalBufferGPU,
    KernelBufferGPU,
    BlockBufferGPU,
}
