use crate::{
    Prm,
    Buffer
};


/// Buffer that stores data on the host. Simply a wrapper around `Vec`.
#[derive(Clone)]
pub struct HostBuffer<T: Prm> {
    vec: Vec<T>,
}

impl<T: Prm> Buffer<T> for HostBuffer<T> {
    type Context = ();

    unsafe fn new_uninit_in(_: &(), len: usize) -> Self {
        let mut vec = Vec::<T>::with_capacity(len);
        vec.set_len(len);
        Self { vec }
    }
    fn new_filled_in(_: &(), len: usize, value: T) -> Self {
        let mut vec = Vec::<T>::new();
        vec.resize(len, value);
        Self { vec }
    }
    
    fn len(&self) -> usize {
        self.vec.len()
    }

    fn load(&self, dst: &mut [T]) {
        dst.copy_from_slice(self.as_slice());
    }
    fn store(&mut self, src: &[T]) {
        self.as_mut_slice().copy_from_slice(src);
    }

    fn copy_from(&mut self, src: &Self) {
        self.store(src.as_slice());
    }
    fn copy_to(&self, dst: &mut Self) {
        dst.copy_from(self);
    }
}

impl<T: Prm> HostBuffer<T> {
    /// Create uninitialzed buffer.
    /// This is unsafe method, but it is helpful for allocation of storage for some subsequent operation.
    pub unsafe fn new_uninit(len: usize) -> Self {
        <Self as Buffer<T>>::new_uninit_in(&(), len)
    }
    /// Create buffer filled with a single value.
    pub fn new_filled(len: usize, value: T) -> Self {
        <Self as Buffer<T>>::new_filled_in(&(), len, value)
    }

    /// Provideas access to underlying memory.
    pub fn as_slice(&self) -> &[T] {
        self.vec.as_slice()
    }
    /// Provideas mutable access to underlying memory.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.vec.as_mut_slice()
    }
}
