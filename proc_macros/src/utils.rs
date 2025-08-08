use syn::Type;

pub type FieldName = String;
pub type FieldType = Type;

pub fn get_idents_and_types_for_struct(struc: &syn::DataStruct, skip: Option<&str>) -> Vec<(FieldName, FieldType)> {
  struc.fields.iter()
  .filter(|field| {
    match skip {
      None => true,
      Some(skip) => {
        !field.attrs.iter().any(|attr| {
          attr.path().is_ident(skip)
        })
      }
    }
  })
  .map(|field| {
    get_ident_and_type_for_field(field)
  }).collect()
}

pub fn get_ident_and_type_for_field(field: &syn::Field) -> (FieldName, FieldType) {
  let field_name = field.ident.as_ref().expect("failed to get field ident").to_string();
  let field_type = field.ty.clone();
  (field_name, field_type)
}