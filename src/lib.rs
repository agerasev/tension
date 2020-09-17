mod num;
pub use num::*;

mod buffer;
pub use buffer::*;

mod tensor;
pub use tensor::*;

pub mod prelude {
    pub use crate::{TensorTrait as _};
}

/// Host-only profile.
/// Import all from it if you are planning to use only host-side tensors.
pub mod host_only {
    pub use super::{
        *,
        HostBuffer as Buffer,
        HostTensor as Tensor,
    };
}

#[cfg(feature = "device")]
/// Host and device profile.
/// Import all from it if you are planning to use both host-side and device-side tensors.
pub mod host_device {
    pub use super::{
        *,
        TensorTrait as Tensor,
    };
}
