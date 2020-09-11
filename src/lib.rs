mod num;
pub use num::*;

mod buffer;
pub use buffer::*;

mod tensor;
pub use tensor::*;

pub mod host {
    use super::*;
    pub use buffer::host::*;
    pub use tensor::host::*;
}

#[cfg(feature = "device")]
pub mod device {
    use super::*;
    pub use buffer::device::*;
    pub use tensor::device::*;
}

pub mod prelude {
    pub use super::CommonTensor;
}
