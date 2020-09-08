use crate::Prm;
#[cfg(feature = "device")]
use ocl::Buffer as OclBuffer;
#[cfg(feature = "device")]
use std::ffi::c_void;

pub struct HostBuffer<T: Prm> {
    pub vec: Vec<T>,
}

#[cfg(feature = "device")]
pub struct DeviceBuffer<T: Prm> {
    pub mem: OclBuffer<T::Dev>,
}

pub enum Buffer<T: Prm> {
    Host(HostBuffer<T>),
    #[cfg(feature = "device")]
    Device(DeviceBuffer<T>),
}

#[cfg(feature = "device")]
pub type OclQueueId = *const c_void;
#[derive(Debug, PartialEq)]
pub enum Location {
    Host,
    #[cfg(feature = "device")]
    Device(OclQueueId),
}

impl<T: Prm> Buffer<T> {
    pub fn location(&self) -> Location {
        match self {
            Self::Host(_) => Location::Host,
            #[cfg(feature = "device")]
            Self::Device(buf) => Location::Device(buf.mem.default_queue().unwrap().as_ptr()),
        }
    }
}
