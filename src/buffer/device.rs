use crate::{
    Prm, Interop,
    Buffer, HostBuffer,
};

use ocl::{Buffer as OclBuffer, Queue, MemFlags};


/// Buffer context.
/// Determines buffer context.
#[derive(Clone, Debug)]
pub struct DeviceContext {
    queue: Queue,
}
impl DeviceContext {
    pub fn new(queue: Queue) -> Self {
        Self { queue }
    }
    pub fn queue(&self) -> &Queue {
        return &self.queue
    }
}
impl PartialEq for DeviceContext {
    fn eq(&self, other: &Self) -> bool {
        self.queue.as_ptr() == other.queue.as_ptr()
    }
}

/// Buffer that stores data on device. Wrapper over OpenCL buffer.
pub struct DeviceBuffer<T: Prm + Interop> {
    mem: OclBuffer<T::Dev>,
    ctx: DeviceContext,
}

impl<T: Prm + Interop> Buffer<T> for DeviceBuffer<T> {
    type Context = DeviceContext;

    unsafe fn new_uninit_in(context: &DeviceContext, len: usize) -> Self {
        OclBuffer::builder()
        .queue(context.queue().clone())
        .flags(MemFlags::READ_WRITE)
        .len(len)
        .build()
        .map(|mem| DeviceBuffer { mem, ctx: context.clone() })
        .unwrap()
    }

    fn new_filled_in(context: &DeviceContext, len: usize, value: T) -> Self {
        OclBuffer::builder()
        .queue(context.queue().clone())
        .flags(MemFlags::READ_WRITE)
        .len(len)
        .fill_val(value.to_dev())
        .build()
        .map(|mem| DeviceBuffer { mem, ctx: context.clone() })
        .unwrap()
    }

    fn len(&self) -> usize {
        self.mem.len()
    }
    fn context(&self) -> &DeviceContext {
        &self.ctx
    }

    fn load(&self, dst: &mut [T]) {
        T::load_from_buffer(dst, &self.mem);
    }
    fn store(&mut self, src: &[T]) {
        T::store_to_buffer(&mut self.mem, src);
    }
    fn copy_from(&mut self, src: &Self) {
        assert_eq!(self.len(), src.len());
        if self.context() == src.context() {
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

    /// Creates a new buffer in a specified context and copies the content to it.
    pub fn clone_to(&self, context: &DeviceContext) -> Self {
        let mut dst = unsafe { Self::new_uninit_in(context, self.len()) };
        dst.copy_from(self);
        dst
    }
}

impl<T: Prm + Interop> Clone for DeviceBuffer<T> {
    fn clone(&self) -> Self {
        self.clone_to(&self.context())
    }
}
