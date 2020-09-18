use crate::{
    Prm, Interop,
    Buffer, HostBuffer,
};

use ocl::{Buffer as OclBuffer, Queue, MemFlags};


/// Buffer location.
/// For now it's simply the wrapper around `ocl::Queue`, but it could be changed in future.
#[derive(Clone, Debug)]
pub struct DeviceLocation(Queue);

impl DeviceLocation {
    pub fn eq_queue(aq: &Queue, bq: &Queue) -> bool {
        aq.as_ptr() == bq.as_ptr()
    }
}

impl PartialEq for DeviceLocation {
    fn eq(&self, other: &Self) -> bool {
        DeviceLocation::eq_queue(&self.0, &other.0)
    }
}

/// Buffer that stores data on device. Wrapper over OpenCL buffer.
pub struct DeviceBuffer<T: Prm + Interop> {
    mem: OclBuffer<T::Dev>,
}

impl<T: Prm + Interop> Buffer<T> for DeviceBuffer<T> {
    type Context = DeviceLocation;

    unsafe fn new_uninit_in(location: &DeviceLocation, len: usize) -> Self {
        OclBuffer::builder()
        .queue(location.0.clone())
        .flags(MemFlags::READ_WRITE)
        .len(len)
        .build()
        .map(|mem| DeviceBuffer { mem })
        .unwrap()
    }
    fn new_filled_in(location: &DeviceLocation, len: usize, value: T) -> Self {
        OclBuffer::builder()
        .queue(location.0.clone())
        .flags(MemFlags::READ_WRITE)
        .len(len)
        .fill_val(value.to_dev())
        .build()
        .map(|mem| DeviceBuffer { mem })
        .unwrap()
    }
    fn len(&self) -> usize {
        self.mem.len()
    }
    fn load(&self, dst: &mut [T]) {
        T::load_from_buffer(dst, &self.mem);
    }
    fn store(&mut self, src: &[T]) {
        T::store_to_buffer(&mut self.mem, src);
    }
    fn copy_from(&mut self, src: &Self) {
        assert_eq!(self.len(), src.len());
        if DeviceLocation::eq_queue(self.queue(), src.queue()) {
            src.mem.copy(&mut self.mem, None, None).enq().unwrap();
        } else {
            let mut tmp = Vec::<T::Dev>::new();
            src.mem.read(&mut tmp).enq()
            .and_then(|_| self.mem.write(tmp.as_slice()).enq())
            .unwrap();
        }
    }
    fn copy_to(&self, dst: &mut Self) {
        dst.copy_from(self);
    }
}

impl<T: Prm + Interop> DeviceBuffer<T> {
    /// DeviceLocation of the buffer.
    pub fn location(&self) -> DeviceLocation {
        DeviceLocation(self.mem.default_queue().unwrap().clone())
    }
    /// Default command queue for buffer.
    pub fn queue(&self) -> &Queue {
        self.mem.default_queue().unwrap()
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
    pub fn clone_to(&self, location: &DeviceLocation) -> Self {
        let mut dst = unsafe { Self::new_uninit_in(location, self.len()) };
        dst.copy_from(self);
        dst
    }
}

impl<T: Prm + Interop> Clone for DeviceBuffer<T> {
    fn clone(&self) -> Self {
        self.clone_to(&self.location())
    }
}
