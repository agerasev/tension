use crate::{
    Prm,
    CommonTensor as TensorTrait,
    host::*,
};
use std::{
    rc::Rc,
};

/*
pub struct Iter<'a, T: Prm> {
    tensor: &'a Tensor<T>,
    indices: Vec<usize>,
}
*/

/// Tensor structure.
/// It consists of a contiguous one-dimensional array and a shape.
/// Tensor tries to reuse resources as long as possible and implements copy-on-write mechanism.
pub struct Tensor<T: Prm> {
    shape: Vec<usize>,
    buffer: Rc<Buffer<T>>,
}

impl<T: Prm> Tensor<T> {
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

    /// Create unitialized tensor
    pub unsafe fn new_uninit(shape: &[usize]) -> Self {
        Self::from_buffer(Buffer::new_uninit(shape.iter().product()), shape)
    }
    /// Create tensor filled with value on the specified hardware
    pub fn new_filled(shape: &[usize], value: T) -> Self {
        Self::from_buffer(Buffer::new_filled(shape.iter().product(), value), shape)
    }
    /// Create tensor filled with zeros on the specified hardware
    pub fn new_zeroed(shape: &[usize]) -> Self {
        Self::new_filled(shape, T::zero())
    }
}

impl<T: Prm> TensorTrait<T> for Tensor<T> {
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
