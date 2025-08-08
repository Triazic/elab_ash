#![feature(proc_macro_quote)]

mod getters;
mod has;
mod utils;

#[proc_macro_derive(Getters, attributes(Getters_Skip))]
pub fn derive_getters(item: proc_macro::TokenStream) -> proc_macro::TokenStream {
  getters::derive_getters(item)
}

#[proc_macro_derive(Has, attributes(Has_Skip))]
pub fn derive_has(item: proc_macro::TokenStream) -> proc_macro::TokenStream {
  has::derive_has(item)
}