use crate::{
    Prm, Interop,
    CommonTensor as TensorTrait,
    device::*,
};
use std::{
    rc::Rc,
};


/// Tensor structure.
/// It consists of a contiguous one-dimensional array and a shape.
/// Tensor tries to reuse resources as long as possible and implements copy-on-write mechanism.
pub struct Tensor<T: Prm + Interop> {
    shape: Vec<usize>,
    buffer: Rc<Buffer<T>>,
}

impl<T: Prm + Interop> Tensor<T> {
    /// Create tensor from shared buffer and shape
    fn from_shared_buffer(rc_buffer: Rc<Buffer<T>>, shape: &[usize]) -> Self {
        assert_eq!(rc_buffer.len(), shape.iter().product());
        Self {
            shape: shape.iter().cloned().collect(),
            buffer: rc_buffer,
        }
    }
    /// Create tensor from specified buffer and shape
    fn from_buffer(buffer: Buffer<T>, shape: &[usize]) -> Self {
        Self::from_shared_buffer(Rc::new(buffer), shape)
    }

    /// Create unitialized tensor in the specified location.
    pub unsafe fn new_uninit(loc: &Location, shape: &[usize]) -> Self {
        Self::from_buffer(Buffer::new_uninit(loc, shape.iter().product()), shape)
    }
    /// Create tensor filled with value in the specified location.
    pub fn new_filled(loc: &Location, shape: &[usize], value: T) -> Self {
        Self::from_buffer(Buffer::new_filled(loc, shape.iter().product(), value), shape)
    }
    /// Create tensor filled with zeros in the specified location.
    pub fn new_zeroed(loc: &Location, shape: &[usize]) -> Self {
        Self::new_filled(loc, shape, T::zero())
    }
}

impl<T: Prm + Interop> TensorTrait<T> for Tensor<T> {
    fn shape(&self) -> &[usize] {
        return self.shape.as_slice();
    }
    fn reshape(&self, shape: &[usize]) -> Self {
        Self::from_shared_buffer(self.buffer.clone(), shape)
    }
    fn load(&self, dst: &mut [T]) {
        self.buffer.load(dst);
    }
    fn store(&mut self, src: &[T]) {
        Rc::make_mut(&mut self.buffer).store(src);
    }
}
