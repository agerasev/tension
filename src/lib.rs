mod num;
mod error;
mod buffer;
mod tensor;

pub use num::*;
pub use error::*;
pub use buffer::*;
pub use tensor::*;


#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
