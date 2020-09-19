use crate::{
    Prm,
    Tensor, HostBuffer,
};
use std::{
    rc::Rc,
};

/*
pub struct Iter<'a, T: Prm> {
    tensor: &'a HostTensor<T>,
    indices: Vec<usize>,
}
*/

/// Tensor structure.
/// It consists of a contiguous one-dimensional array and a shape.
/// Tensor tries to reuse resources as long as possible and implements copy-on-write mechanism.
pub struct HostTensor<T: Prm> {
    shape: Vec<usize>,
    buffer: Rc<HostBuffer<T>>,
}

impl<T: Prm> HostTensor<T> {
    /// Create unitialized tensor
    pub unsafe fn new_uninit(shape: &[usize]) -> Self {
        Self::new_uninit_in(&(), shape)
    }
    /// Create tensor filled with value on the specified hardware
    pub fn new_filled(shape: &[usize], value: T) -> Self {
        Self::new_filled_in(&(), shape, value)
    }
    /// Create tensor filled with zeros on the specified hardware
    pub fn new_zeroed(shape: &[usize]) -> Self {
        Self::new_zeroed_in(&(), shape)
    }
}

impl<T: Prm> Tensor<T> for HostTensor<T> {
    type Buffer = HostBuffer<T>;

    unsafe fn from_shared_buffer_unchecked(rc_buffer: Rc<HostBuffer<T>>, shape: &[usize]) -> Self {
        Self {
            shape: shape.iter().cloned().collect(),
            buffer: rc_buffer,
        }
    }

    fn shape(&self) -> &[usize] {
        self.shape.as_slice()
    }

    unsafe fn shared_buffer(&self) -> &Rc<HostBuffer<T>> {
        &self.buffer
    }
    unsafe fn shared_buffer_mut(&mut self) -> &mut Rc<HostBuffer<T>> {
        &mut self.buffer
    }
}
