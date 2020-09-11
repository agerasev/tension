use crate::{Prm, Error};

#[cfg(feature = "device")]
use ocl::{Buffer as OclBuffer, Queue, MemFlags};


/// Memory location. Can be either host or some of devices.
#[derive(Clone, Debug)]
pub enum Location {
    Host,
    #[cfg(feature = "device")]
    Device(Queue),
}

impl Location {
    pub fn eq_queue(aq: &Queue, bq: &Queue) -> bool {
        aq.as_ptr() == bq.as_ptr()
    }
}

impl PartialEq for Location {
    fn eq(&self, other: &Self) -> bool {
        match self {
            Self::Host => match other {
                Self::Host => true,
                #[cfg(feature = "device")]
                _ => false,
            },

            #[cfg(feature = "device")]
            Self::Device(sq) => match other {
                Self::Device(oq) => Location::eq_queue(sq, oq),
                _ => false,
            }
        }
    }
}

/// Buffer that stores data on the host. Simply a wrapper around `Vec`.
#[derive(Clone)]
pub struct HostBuffer<T: Prm> {
    vec: Vec<T>,
}
impl<T: Prm> HostBuffer<T> {
    pub unsafe fn new_uninit(len: usize) -> Result<Self, Error> {
        let mut vec = Vec::<T>::with_capacity(len);
        vec.set_len(len);
        Ok(Self { vec })
    }
    pub fn new_filled(len: usize, value: T) -> Result<Self, Error> {
        let mut vec = Vec::<T>::new();
        vec.resize(len, value);
        Ok(Self { vec })
    }
    pub fn len(&self) -> usize {
        self.vec.len()
    }
    pub fn as_slice(&self) -> &[T] {
        self.vec.as_slice()
    }
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.vec.as_mut_slice()
    }
    pub fn load(&self, dst: &mut [T]) {
        dst.copy_from_slice(self.as_slice());
    }
    pub fn store(&mut self, src: &[T]) {
        self.as_mut_slice().copy_from_slice(src);
    }
}

/// Buffer that stores data on device. Wrapper over OpenCL buffer.
#[cfg(feature = "device")]
pub struct DeviceBuffer<T: Prm> {
    mem: OclBuffer<T::Dev>,
}
#[cfg(feature = "device")]
impl<T: Prm> DeviceBuffer<T> {
    pub unsafe fn new_uninit(queue: &Queue, len: usize) -> Result<Self, Error> {
        OclBuffer::builder()
        .queue(queue.clone())
        .flags(MemFlags::READ_WRITE)
        .len(len)
        .build()
        .map(|mem| DeviceBuffer { mem })
        .map_err(|e| Error::Ocl(e))
    }
    pub fn new_filled(queue: &Queue, len: usize, value: T) -> Result<Self, Error> {
        OclBuffer::builder()
        .queue(queue.clone())
        .flags(MemFlags::READ_WRITE)
        .len(len)
        .fill_val(value.to_dev())
        .build()
        .map(|mem| DeviceBuffer { mem })
        .map_err(|e| Error::Ocl(e))
    }
    pub fn len(&self) -> usize {
        self.mem.len()
    }
    pub fn queue(&self) -> &Queue {
        self.mem.default_queue().unwrap()
    }
    pub fn load(&self, dst: &mut [T]) -> Result<(), Error> {
        T::load_from_buffer(dst, &self.mem).map_err(|e| Error::Ocl(e))
    }
    pub fn store(&mut self, src: &[T]) -> Result<(), Error> {
        T::store_to_buffer(&mut self.mem, src).map_err(|e| Error::Ocl(e))
    }
    pub fn copy_from(&mut self, src: &Self) -> Result<(), Error> {
        assert_eq!(self.len(), src.len());
        if Location::eq_queue(self.queue(), src.queue()) {
            src.mem.copy(&mut self.mem, None, None).enq().map_err(|e| Error::Ocl(e))
        } else {
            let mut tmp = Vec::<T::Dev>::new();
            src.mem.read(&mut tmp).enq()
            .and_then(|_| self.mem.write(tmp.as_slice()).enq())
            .map_err(|e| Error::Ocl(e))
        }
    }
}

/// Generic buffer that can be either host-side and device-side.
pub enum Buffer<T: Prm> {
    Host(HostBuffer<T>),
    #[cfg(feature = "device")]
    Device(DeviceBuffer<T>),
}

impl<T: Prm> Buffer<T> {
    /// Create uninitialzed buffer.
    /// This is unsafe method, but it is helpful for allocation of storage for some subsequent operation.
    pub unsafe fn new_uninit(location: &Location, len: usize) -> Result<Self, Error> {
        match location {
            Location::Host => {
                HostBuffer::new_uninit(len)
            }.map(|hbuf| Self::Host(hbuf)),
            #[cfg(feature = "device")]
            Location::Device(queue) => {
                DeviceBuffer::new_uninit(queue, len)
            }.map(|dbuf| Self::Device(dbuf)),
        }
    }
    /// Create buffer filled with a single value.
    pub fn new_filled(location: &Location, len: usize, value: T) -> Result<Self, Error> {
        match location {
            Location::Host => {
                HostBuffer::new_filled(len, value)
            }.map(|hbuf| Self::Host(hbuf)),
            #[cfg(feature = "device")]
            Location::Device(queue) => {
                DeviceBuffer::new_filled(queue, len, value)
            }.map(|dbuf| Self::Device(dbuf)),
        }
    }

    /// Returns location of the buffer.
    pub fn location(&self) -> Location {
        match self {
            Self::Host(_) => Location::Host,
            #[cfg(feature = "device")]
            Self::Device(buf) => Location::Device(buf.queue().clone()),
        }
    }
    /// Returns the length of the buffer.
    pub fn len(&self) -> usize {
        match self {
            Self::Host(hbuf) => hbuf.len(),
            #[cfg(feature = "device")]
            Self::Device(dbuf) => dbuf.len(),
        }
    }
    /// Loads data from buffer to slice.
    pub fn load(&self, dst: &mut [T]) -> Result<(), Error> {
        if self.len() == dst.len() {
            match self {
                Self::Host(hbuf) => { hbuf.load(dst); Ok(()) },
                #[cfg(feature = "device")]
                Self::Device(dbuf) => dbuf.load(dst),
            }
        } else {
            Err(Error::LengthMismatch(dst.len(), self.len()))
        }
    }
    /// Stores data from slice to buffer.
    pub fn store(&mut self, src: &[T]) -> Result<(), Error> {
        if self.len() == src.len() {
            match self {
                Self::Host(hbuf) => { hbuf.store(src); Ok(()) },
                #[cfg(feature = "device")]
                Self::Device(dbuf) => dbuf.store(src),
            }
        } else {
            Err(Error::LengthMismatch(self.len(), src.len()))
        }
    }
    /// Copies content to `self` from another buffer.
    pub fn copy_from(&mut self, src: &Self) -> Result<(), Error> {
        if self.len() == src.len() {
            match (self, src) {
                (Self::Host(hdst), Self::Host(hsrc)) => {
                    hdst.as_mut_slice().copy_from_slice(hsrc.as_slice());
                    Ok(())
                },
                #[cfg(feature = "device")]
                (Self::Device(ddst), Self::Device(dsrc)) => {
                    ddst.copy_from(dsrc)
                },
                #[cfg(feature = "device")]
                (Self::Host(hdst), Self::Device(dsrc)) => {
                    dsrc.load(hdst.as_mut_slice())
                },
                #[cfg(feature = "device")]
                (Self::Device(ddst), Self::Host(hsrc)) => {
                    ddst.store(hsrc.as_slice())
                },
            }
        } else {
            Err(Error::LengthMismatch(self.len(), src.len()))
        }
    }
    /// Copies content from `self` to another buffer.
    pub fn copy_to(&self, dst: &mut Self) -> Result<(), Error> {
        dst.copy_from(self)
    }

    /// Creates a new buffer in a specified location and copies the content to it.
    pub fn clone_to(&self, location: &Location) -> Result<Self, Error> {
        unsafe { Self::new_uninit(location, self.len()) }
        .and_then(|mut dst| dst.copy_from(self).map(|_| dst))
    }
    /// Creates a new buffer in the same location and copies the content to it.
    /// It is not an implementation of the `Clone` trait because there may be an error.
    pub fn clone(&self) -> Result<Self, Error> {
        self.clone_to(&self.location())
    }
}
