use crate::{Prm};

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
    pub unsafe fn new_uninit(len: usize) -> Self {
        let mut vec = Vec::<T>::with_capacity(len);
        vec.set_len(len);
        Self { vec }
    }
    pub fn new_filled(len: usize, value: T) -> Self {
        let mut vec = Vec::<T>::new();
        vec.resize(len, value);
        Self { vec }
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
    pub unsafe fn new_uninit(queue: &Queue, len: usize) -> Self {
        OclBuffer::builder()
        .queue(queue.clone())
        .flags(MemFlags::READ_WRITE)
        .len(len)
        .build()
        .map(|mem| DeviceBuffer { mem })
        .unwrap()
    }
    pub fn new_filled(queue: &Queue, len: usize, value: T) -> Self {
        OclBuffer::builder()
        .queue(queue.clone())
        .flags(MemFlags::READ_WRITE)
        .len(len)
        .fill_val(value.to_dev())
        .build()
        .map(|mem| DeviceBuffer { mem })
        .unwrap()
    }
    pub fn len(&self) -> usize {
        self.mem.len()
    }
    pub fn queue(&self) -> &Queue {
        self.mem.default_queue().unwrap()
    }
    pub fn load(&self, dst: &mut [T]) {
        T::load_from_buffer(dst, &self.mem);
    }
    pub fn store(&mut self, src: &[T]) {
        T::store_to_buffer(&mut self.mem, src);
    }
    pub fn copy_from(&mut self, src: &Self) {
        assert_eq!(self.len(), src.len());
        if Location::eq_queue(self.queue(), src.queue()) {
            src.mem.copy(&mut self.mem, None, None).enq().unwrap();
        } else {
            let mut tmp = Vec::<T::Dev>::new();
            src.mem.read(&mut tmp).enq()
            .and_then(|_| self.mem.write(tmp.as_slice()).enq())
            .unwrap();
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
    pub unsafe fn new_uninit(location: &Location, len: usize) -> Self {
        match location {
            Location::Host => Self::Host({
                HostBuffer::new_uninit(len)
            }),
            #[cfg(feature = "device")]
            Location::Device(queue) => Self::Device({
                DeviceBuffer::new_uninit(queue, len)
            }),
        }
    }
    /// Create buffer filled with a single value.
    pub fn new_filled(location: &Location, len: usize, value: T) -> Self {
        match location {
            Location::Host => Self::Host({
                HostBuffer::new_filled(len, value)
            }),
            #[cfg(feature = "device")]
            Location::Device(queue) => Self::Device({
                DeviceBuffer::new_filled(queue, len, value)
            }),
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
    pub fn load(&self, dst: &mut [T]) {
        match self {
            Self::Host(hbuf) => hbuf.load(dst),
            #[cfg(feature = "device")]
            Self::Device(dbuf) => dbuf.load(dst),
        }
    }
    /// Stores data from slice to buffer.
    pub fn store(&mut self, src: &[T]) {
        match self {
            Self::Host(hbuf) => hbuf.store(src),
            #[cfg(feature = "device")]
            Self::Device(dbuf) => dbuf.store(src),
        }
    }
    /// Copies content to `self` from another buffer.
    pub fn copy_from(&mut self, src: &Self) {
        match (self, src) {
            (Self::Host(hdst), Self::Host(hsrc)) => {
                hdst.as_mut_slice().copy_from_slice(hsrc.as_slice());
            },
            #[cfg(feature = "device")]
            (Self::Device(ddst), Self::Device(dsrc)) => {
                ddst.copy_from(dsrc);
            },
            #[cfg(feature = "device")]
            (Self::Host(hdst), Self::Device(dsrc)) => {
                dsrc.load(hdst.as_mut_slice());
            },
            #[cfg(feature = "device")]
            (Self::Device(ddst), Self::Host(hsrc)) => {
                ddst.store(hsrc.as_slice());
            },
        }
    }
    /// Copies content from `self` to another buffer.
    pub fn copy_to(&self, dst: &mut Self) {
        dst.copy_from(self);
    }

    /// Creates a new buffer in a specified location and copies the content to it.
    pub fn clone_to(&self, location: &Location) -> Self {
        let mut dst = unsafe { Self::new_uninit(location, self.len()) };
        dst.copy_from(self);
        dst
    }
}

impl<T: Prm> Clone for Buffer<T> {
    fn clone(&self) -> Self {
        self.clone_to(&self.location())
    }
}
