pub mod host;
pub use host::{
    Buffer as HostBuffer,
};

#[cfg(feature = "device")]
pub mod device;
#[cfg(feature = "device")]
pub use device::{
    Location as DeviceLocation,
    Buffer as DeviceBuffer,
};
