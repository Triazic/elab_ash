use syn::Type;
extern crate proc_macro;
use crate::utils;

#[allow(unused)]
pub fn derive_getters(item: proc_macro::TokenStream) -> proc_macro::TokenStream {
  let empty_return = "".parse().unwrap();
  let input = syn::parse_macro_input!(item as syn::DeriveInput);

  // check it's a struct...
  let syn::Data::Struct(data) = &input.data else { return empty_return; };

  // struct name
  let struct_name = input.ident;

  // get fields (other than skipped ones)
  let fields = crate::utils::get_idents_and_types_for_struct(&data, Some("Getters_Skip"));

  // make getters for fields and concat them
  let getters: proc_macro2::TokenStream = fields.iter().map(|(field_name, field_type)| {
    get_getter_for_field(field_name, field_type)
  }).into_iter().collect();

  // prepare final output
  quote::quote! {
    impl #struct_name {
      #getters
    }
  }.into()
}

fn get_getter_for_field(field_name: &utils::FieldName, field_type: &utils::FieldType) -> proc_macro2::TokenStream {
  let getter_name = quote::format_ident!("{}", field_name);
  let field_ident = quote::format_ident!("{}", field_name);

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
      pub fn #getter_name(&self) -> #field_type {
        self.#field_ident
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
      pub fn #getter_name(&self) -> &str {
        &self.#field_ident
      }
    }.into()
  }
  
  quote::quote!{
    pub fn #getter_name(&self) -> &#field_type {
      &self.#field_ident
    }
  }.into()
}