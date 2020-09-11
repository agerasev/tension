use crate::{Prm};
use super::host::*;

use ocl::{Buffer as OclBuffer, Queue, MemFlags};


/// Buffer location.
/// For now it's simply the wrapper around `ocl::Queue`, but it could be changed in future.
#[derive(Clone, Debug)]
pub struct Location(Queue);

impl Location {
    pub fn eq_queue(aq: &Queue, bq: &Queue) -> bool {
        aq.as_ptr() == bq.as_ptr()
    }
}

impl PartialEq for Location {
    fn eq(&self, other: &Self) -> bool {
        Location::eq_queue(&self.0, &other.0)
    }
}

/// Buffer that stores data on device. Wrapper over OpenCL buffer.
pub struct DeviceBuffer<T: Prm> {
    mem: OclBuffer<T::Dev>,
}
impl<T: Prm> DeviceBuffer<T> {
    /// Create uninitialzed buffer.
    /// This is unsafe method, but it is helpful for allocation of storage for some subsequent operation.
    pub unsafe fn new_uninit(location: &Location, len: usize) -> Self {
        OclBuffer::builder()
        .queue(location.0.clone())
        .flags(MemFlags::READ_WRITE)
        .len(len)
        .build()
        .map(|mem| DeviceBuffer { mem })
        .unwrap()
    }
    /// Create buffer filled with a single value.
    pub fn new_filled(location: &Location, len: usize, value: T) -> Self {
        OclBuffer::builder()
        .queue(location.0.clone())
        .flags(MemFlags::READ_WRITE)
        .len(len)
        .fill_val(value.to_dev())
        .build()
        .map(|mem| DeviceBuffer { mem })
        .unwrap()
    }
    /// Length of the buffer.
    pub fn len(&self) -> usize {
        self.mem.len()
    }
    /// Location of the buffer.
    pub fn location(&self) -> Location {
        Location(self.mem.default_queue().unwrap().clone())
    }
    /// Default command queue for buffer.
    pub fn queue(&self) -> &Queue {
        self.mem.default_queue().unwrap()
    }
    /// Loads data from buffer to slice.
    pub fn load(&self, dst: &mut [T]) {
        T::load_from_buffer(dst, &self.mem);
    }
    /// Stores data from slice to buffer.
    pub fn store(&mut self, src: &[T]) {
        T::store_to_buffer(&mut self.mem, src);
    }
    /// Copies content to `self` from another buffer.
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
    /// Copies content from `self` to another buffer.
    pub fn copy_to(&self, dst: &mut Self) {
        dst.copy_from(self);
    }
    /// Copies content to `self` from host buffer.
    pub fn copy_from_host(&mut self, src: &HostBuffer<T>) {
        assert_eq!(self.len(), src.len());
        self.store(src.as_slice());
    }
    /// Copies content from `self` to host buffer.
    pub fn copy_to_host(&self, dst: &mut HostBuffer<T>) {
        assert_eq!(self.len(), dst.len());
        self.load(dst.as_mut_slice());
    }

    /// Creates a new buffer in a specified location and copies the content to it.
    pub fn clone_to(&self, location: &Location) -> Self {
        let mut dst = unsafe { Self::new_uninit(location, self.len()) };
        dst.copy_from(self);
        dst
    }
}

impl<T: Prm> Clone for DeviceBuffer<T> {
    fn clone(&self) -> Self {
        self.clone_to(&self.location())
    }
}
