#[macro_export]
macro_rules! unpack {
  ($x:ident, $($name:ident),+ $(,)?) => {
      $(let $name = $x.$name();)+
  };
}