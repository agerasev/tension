use crate::{Prm};


pub trait Buffer<T: Prm> {
    /// Context for buffer allocation.
    type Context;

    /// Create uninitialzed buffer.
    /// This is unsafe method, but it is helpful for pre-allocation of storage for some operations.
    unsafe fn new_uninit_in(context: &Self::Context, len: usize) -> Self;
    /// Create buffer filled with a single value.
    fn new_filled_in(context: &Self::Context, len: usize, value: T) -> Self;

    /// Returns the length of the buffer.
    fn len(&self) -> usize;
    /// Context of the buffer.
    fn context(&self) -> &Self::Context;

    /// Loads data from buffer to slice.
    fn load(&self, dst: &mut [T]);
    /// Stores data from slice to buffer.
    fn store(&mut self, src: &[T]);

    /// Copies content to `self` from another buffer.
    fn copy_from(&mut self, src: &Self);
    /// Copies content from `self` to another buffer.
    fn copy_to(&self, dst: &mut Self);
}
