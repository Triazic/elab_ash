use crate::{constants, memory, utils, vk_context::{VKContext}};
use std::{ffi::{CStr, CString}, io::Read, str::FromStr};
extern crate itertools;
extern crate strum;
use itertools::Itertools;
use crate::utils::{cstr};
use crate::{memory::{print_flags, split_flags, split_flags_u32}, utils::print_endianness};

pub fn create_vk_context() -> VKContext {
  // make entry, instance, device
  let entry = create_entry();
  let instance = create_instance(&entry);
  let (physical_device, device, main_queue_family_index) = create_device(&entry, &instance);
  let queue_index = 0; // only one queue for now

  // queue
  let main_queue = get_queue(&device, main_queue_family_index, queue_index);

  // command pool
  let command_pool = create_command_pool(&device, main_queue_family_index);

  // pipeline cache
  let pipeline_cache = unsafe { device.create_pipeline_cache(&ash::vk::PipelineCacheCreateInfo::default(), None).expect("failed to create pipeline cache") };

  println!("chosen device: {:?}", get_physical_device_name(&instance, physical_device));
  println!("chosen queue_family_index: {:?}", main_queue_family_index);
  println!("chosen queue_index: {:?}", queue_index);

  let vk_context = VKContext {
    entry, 
    instance, 
    physical_device, 
    device, 
    command_pool, 
    main_queue_family_index, 
    main_queue,
    pipeline_cache
  };

  vk_context
}

fn get_physical_device_name(instance: &ash::Instance, physical_device: ash::vk::PhysicalDevice) -> String {
  let properties = unsafe { instance.get_physical_device_properties(physical_device) };
  let name = properties.device_name.clone();
  let str =     
    unsafe {
    // Interpret the raw i8 array as a CStr
    CStr::from_ptr(name.as_ptr())
      .to_string_lossy()
      .into_owned()
  };
  str 
}
fn create_entry() -> ash::Entry {
  let entry = unsafe { ash::Entry::load().expect("Could not load Vulkan") };
  entry
}

fn create_instance(entry: &ash::Entry) -> ash::Instance {
  // application info
  let application_name = cstr("ELABS_AUDIO_APPLICATION");
  let application_version = 1;
  let engine_name = cstr("ELABS_AUDIO_ENGINE");
  let engine_version = 1;
  
  let application_info = ash::vk::ApplicationInfo::default()
    .application_name(&application_name)
    .application_version(application_version)
    .engine_name(&engine_name)
    .engine_version(engine_version)
    .api_version(constants::API_VERSION);
  
  // check that required layers are supported
  let layers = unsafe { entry.enumerate_instance_layer_properties().expect("failed to enumerate instance layer properties") };
  let assert_layer_supported = |layer_name: &str| {
    let has = layers.iter().any(|layer| {
      let name = layer.layer_name_as_c_str().expect("could not get layer name").to_str().expect("could not convert layer name to &str");
      name == layer_name
    });
    assert!(has, "instance layer {} is not supported", layer_name);
  };
  constants::REQUIRED_INSTANCE_LAYERS.iter().for_each(|layer| assert_layer_supported(layer));

  // check that required extensions are supported
  let extensions = unsafe { entry.enumerate_instance_extension_properties(None).expect("failed to enumerate instance extension properties") };
  // extensions.iter().for_each(|extension| {
  //   let name = extension.extension_name_as_c_str().expect("could not get extension name").to_str().expect("could not convert extension name to &str");
  //   println!("instance extension {}", name);
  // });
  let assert_extension_supported = |extension_name: &str| {
    let has = extensions.iter().any(|extension| {
      let name = extension.extension_name_as_c_str().expect("could not get extension name").to_str().expect("could not convert extension name to &str");
      name == extension_name
    });
    assert!(has, "instance extension {} is not supported", extension_name);
  };
  constants::REQUIRED_INSTANCE_EXTENSIONS.iter().for_each(|extension| assert_extension_supported(extension));

  // instance create info
  let flags = ash::vk::InstanceCreateFlags::empty();
  let layer_cstrs = constants::REQUIRED_INSTANCE_LAYERS.iter().map(|str| cstr(str)).collect_vec();
  let layer_ptrs: Vec<*const i8> = layer_cstrs.iter().map(|s| s.as_ptr()).collect();
  let extension_strs = 
    constants::REQUIRED_INSTANCE_EXTENSIONS.iter()
    .unique()
    .collect_vec();
    ;
  let extension_cstrs = extension_strs.iter().map(|str| cstr(str)).collect_vec();
  let extension_ptrs: Vec<*const i8> = extension_cstrs.iter().map(|s| s.as_ptr()).collect();
  let instance_create_info = ash::vk::InstanceCreateInfo::default()
    .flags(flags)
    .application_info(&application_info)
    .enabled_layer_names(&layer_ptrs)
    .enabled_extension_names(&extension_ptrs);

  // create instance
  let instance = unsafe { entry.create_instance(&instance_create_info, None).expect("Could not create Vulkan instance") };
  instance
}

fn create_device(entry: &ash::Entry, instance: &ash::Instance) -> (ash::vk::PhysicalDevice, ash::Device, u32) {
  // physical device
  let physical_devices = unsafe { instance.enumerate_physical_devices().expect("failed to enumerate physical devices") };
  // assert that there is at least one physical device
  assert!(physical_devices.len() > 0, "no physical devices found");
  // find the first suitable device
  let physical_device = *physical_devices.iter().find(|&physical_device| {
    let properties = unsafe { instance.get_physical_device_properties(*physical_device) };

    // number of memory allocations
    if properties.limits.max_memory_allocation_count < 1 { return false; }

    // memory flags
    if !memory::get_if_physical_device_supports_all_memory_requirements(instance, physical_device) { return false; }

    // supported image formats
    let req_formats = [];
    if !req_formats.iter().all(|&req_format|
      {
        let props = unsafe {
          instance.get_physical_device_format_properties(*physical_device, req_format)
        };

        let flags = ash::vk::FormatFeatureFlags::SAMPLED_IMAGE;

        let pass_linear = props.linear_tiling_features & flags == flags;
        let pass_optimal = props.optimal_tiling_features & flags == flags;
  
        pass_linear && pass_optimal
      }
    ) { return false; }

    // check vulkan version
    let required_vulkan_version = constants::API_VERSION;
    let supported_version = properties.api_version;
    if supported_version < required_vulkan_version { return false; }

    let limits = properties.limits;
    let max_image_size = limits.max_image_dimension2_d;
    // dbg!(limits.buffer_image_granularity);
    // dbg!(limits.non_coherent_atom_size);

    let sparse_properties = properties.sparse_properties;

    let features = unsafe { instance.get_physical_device_features(*physical_device) };

    let device_memory_properties = unsafe { instance.get_physical_device_memory_properties(*physical_device) };

    let memory_types = device_memory_properties.memory_types_as_slice();

    // let physical_device_format_properties = unsafe { instance.get_physical_device_format_properties(*physical_device, format) };

    // let physical_device_image_properties = unsafe { instance.get_physical_device_image_format_properties(*physical_device, format) };

    // check for required device layers
    let layers = unsafe { instance.enumerate_device_layer_properties(*physical_device).expect("failed to enumerate device layer properties") };
    let layer_supported = |layer_name: &str| {
      layers.iter().any(|layer| {
        let name = layer.layer_name_as_c_str().expect("could not get layer name").to_str().expect("could not convert layer name to &str");
        name == layer_name
      })
    };
    let all_layers_supported = constants::REQUIRED_DEVICE_LAYERS.iter().all(|layer| layer_supported(layer));
    if !all_layers_supported { return false; }

    // check for required device extensions
    let extensions = unsafe { instance.enumerate_device_extension_properties(*physical_device).expect("failed to enumerate device extension properties") };
    let extension_supported = |extension_name: &str| {
      extensions.iter().any(|extension| {
        let name = extension.extension_name_as_c_str().expect("could not get extension name").to_str().expect("could not convert extension name to &str");
        name == extension_name
      })
    };
    let all_extensions_supported = constants::REQUIRED_DEVICE_EXTENSIONS.iter().all(|extension| extension_supported(extension));
    if !all_extensions_supported { return false; }

    true // device is adequate
  }).expect("no physical device satisifies the requirements of this application");

  let queue_family_properties = unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
  // find index of first suitable queue family
  assert!(queue_family_properties.len() > 0, "no queue families found");
  let queue_family_index = queue_family_properties.iter().enumerate().position(|(i, &properties)| {
    if properties.queue_count < 1 { return false; }
    let req_flags = [
      ash::vk::QueueFlags::COMPUTE, 
      ash::vk::QueueFlags::TRANSFER,
    ];
    for req_flag in req_flags.iter() {
      if !properties.queue_flags.contains(*req_flag) { return false; }
    }
    let exc_flags = [
      // ash::vk::QueueFlags::GRAPHICS, // don't want to choose a graphics family
    ];
    for exc_flag in exc_flags.iter() {
      if properties.queue_flags.contains(*exc_flag) { return false; }
    }

    true // queue family is adequate
  }).expect("no queue family satisifies the requirements of this application");

  // queue create info
  let queue_priorities = [1.0];
  let main_queue =  ash::vk::DeviceQueueCreateInfo::default()
    .queue_family_index(queue_family_index as u32)
    .queue_priorities(&queue_priorities);

  let queue_create_infos = vec![main_queue];

  // device create info
  let extension_strs: Vec<&str> = constants::REQUIRED_DEVICE_EXTENSIONS.into_iter().collect_vec();
  let extension_cstrs = extension_strs.iter().map(|str| cstr(str)).collect_vec();
  let extension_ptrs: Vec<*const i8> = extension_cstrs.iter().map(|s| s.as_ptr()).collect();
  let device_features = ash::vk::PhysicalDeviceFeatures::default();
  let device_create_info = ash::vk::DeviceCreateInfo::default()
    .queue_create_infos(&queue_create_infos)
    .enabled_extension_names(&extension_ptrs)
    .enabled_features(&device_features);

  // create device
  let device = unsafe { instance.create_device(physical_device, &device_create_info, None).expect("Could not create Vulkan device") };
  (physical_device, device, queue_family_index as u32)
}

fn get_queue(device: &ash::Device, queue_family_index: u32, queue_index: u32) -> ash::vk::Queue { 
  let queue = unsafe { device.get_device_queue(queue_family_index, queue_index) };
  return queue;
}

fn create_command_pool(device: &ash::Device, queue_family_index: u32) -> ash::vk::CommandPool {
  let flags = ash::vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER;
  let create_info = ash::vk::CommandPoolCreateInfo::default()
    .flags(flags)
    .queue_family_index(queue_family_index)
    .flags(flags);
  let command_pool = unsafe { device.create_command_pool(&create_info, None).expect("failed to create command pool") };
  return command_pool;
}