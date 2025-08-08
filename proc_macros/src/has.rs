extern crate proc_macro;
use syn::Type;

use crate::{utils};

pub fn derive_has(item: proc_macro::TokenStream) -> proc_macro::TokenStream {
  let empty_return = "".parse().unwrap();
  let input = syn::parse_macro_input!(item as syn::DeriveInput);

  // check it's a struct...
  let syn::Data::Struct(data) = &input.data else { return empty_return; };

  // struct name
  let struct_name = input.ident;

  // get fields (other than skipped ones)
  let fields = crate::utils::get_idents_and_types_for_struct(&data, Some("Has_Skip"));

  let has_traits: proc_macro2::TokenStream = fields.iter().map(|(field_name, field_type)| {
    get_has_for_field(field_name, field_type)
  }).into_iter().collect();

  let has_impls: proc_macro2::TokenStream = fields.iter().map(|(field_name, field_type)| {
    get_impl_for_field(&struct_name, field_name, field_type)
  }).into_iter().collect();

  quote::quote! {
    #has_traits
    #has_impls
  }.into()
}

/// GPT
fn snake_to_pascal(s: &str) -> String {
  s.split('_')
    .filter(|w| !w.is_empty())
    .map(|w| {
      let mut c = w.chars();
      match c.next() {
        Some(f) => f.to_ascii_uppercase().to_string() + c.as_str(),
        None => String::new(),
      }
    })
    .collect()
}

fn get_has_for_field(field_name: &utils::FieldName, field_type: &utils::FieldType) -> proc_macro2::TokenStream {
  // need to convert field name from snake_case to PascalCase
  let field_name_pascal = snake_to_pascal(field_name);
  let trait_name_ident = quote::format_ident!("Has{}", field_name_pascal);
  let fn_name = quote::format_ident!("{}", field_name);

  // slow
  let no_ref_types = ["bool", "i8", "u8", "i16", "u16", "i32", "u32", "i64", "u64", "f32", "f64"];
  let is_no_ref_type = {
    match &field_type {
      Type::Path(path) => {
        let last_segment = path.path.segments.iter().last().expect("failed to get last segment?");
        no_ref_types.contains(&last_segment.ident.to_string().as_str())
      },
      _ => false
    }
  };
  if is_no_ref_type {
    return quote::quote!{
      pub trait #trait_name_ident {
        fn #fn_name(&self) -> #field_type;
      }
    }.into()
  }

  let is_string = {
    match &field_type {
      Type::Path(path) => {
        let last_segment = path.path.segments.iter().last().expect("failed to get last segment?");
        last_segment.ident == "String"
      },
      _ => false
    }
  };
  if is_string {
    return quote::quote!{
      pub trait #trait_name_ident {
        fn #fn_name(&self) -> &str;
      }
    }.into()
  }

  quote::quote! {
    pub trait #trait_name_ident {
      fn #fn_name(&self) -> &#field_type;
    }
  }
}

fn get_impl_for_field(struct_name: &syn::Ident, field_name: &utils::FieldName, field_type: &utils::FieldType) -> proc_macro2::TokenStream {
  // need to convert field name from snake_case to PascalCase
  let field_name_pascal = snake_to_pascal(field_name);
  let trait_name_ident = quote::format_ident!("Has{}", field_name_pascal);
  let field_name_ident = quote::format_ident!("{}", field_name);
  let fn_name = quote::format_ident!("{}", field_name);

  // slow
  let no_ref_types = ["bool", "i8", "u8", "i16", "u16", "i32", "u32", "i64", "u64", "f32", "f64"];
  let is_no_ref_type = {
    match &field_type {
      Type::Path(path) => {
        let last_segment = path.path.segments.iter().last().expect("failed to get last segment?");
        no_ref_types.contains(&last_segment.ident.to_string().as_str())
      },
      _ => false
    }
  };
  if is_no_ref_type {
    return quote::quote!{
      impl #trait_name_ident for #struct_name {
        fn #fn_name(&self) -> #field_type {
          self.#field_name_ident
        }
      }
    }.into()
  }

  let is_string = {
    match &field_type {
      Type::Path(path) => {
        let last_segment = path.path.segments.iter().last().expect("failed to get last segment?");
        last_segment.ident == "String"
      },
      _ => false
    }
  };
  if is_string {
    return quote::quote!{
      impl #trait_name_ident for #struct_name {
        fn #fn_name(&self) -> &str {
          &self.#field_name_ident
        }
      }
    }.into()
  }

  quote::quote! {
    impl #trait_name_ident for #struct_name {
      fn #fn_name(&self) -> &#field_type {
        &self.#field_name_ident
      }
    }
  }
}