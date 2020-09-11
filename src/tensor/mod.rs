pub mod common;
pub use common::{Tensor as CommonTensor};

pub mod host;
pub use host::{Tensor as HostTensor};

#[cfg(feature = "device")]
pub mod device;
#[cfg(feature = "device")]
pub use device::{Tensor as DeviceTensor};

#[cfg(test)]
mod test;
