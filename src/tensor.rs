use crate::{
    Prm,
    Buffer, Location,
};
use std::{
    rc::Rc,
};

/// Struture representing range for one dimension for tensor slicing operation.
///
/// Both `begin` and `end` indices can be negative that means indexing from the end (e.g. `-1` means last element).
#[derive(Clone, Copy, Debug)]
pub struct Range {
    /// The *inclusive* start index of range.
    pub start: isize,
    /// The *exclusive* end index of range.
    pub end: isize,
    /// The step between elements. May be negative that means stepping in reversed order.
    /// May not be zero, this will cause error on slice operation.
    pub step: isize,
}

/// Index for one dimension for tensor slicing operaion.
#[derive(Clone, Copy, Debug)]
pub enum Index {
    /// Single index that extract a corresponding section of tensor.
    /// Removes corresponding dimension from shape.
    Single(isize),
    /// Range of indices with step.
    /// Extracts corresponding sections and leaves its dimension on its place.
    Range(Range),
}
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
    /// Create uninitialized tensor on the host
    pub unsafe fn new_uninit(shape: &[usize]) -> Self {
        Self::new_uninit_on(&Location::Host, shape)
    }
    /// Create unitialized tensor on the specified hardware
    pub unsafe fn new_uninit_on(loc: &Location, shape: &[usize]) -> Self {
        Self::from_buffer(Buffer::new_uninit(loc, shape.iter().product()), shape)
    }
    /// Create tensor and fill it with single value on the host
    pub fn new_filled(shape: &[usize], value: T) -> Self {
        Self::new_filled_on(&Location::Host, shape, value)
    }
    /// Create tensor filled with value on the specified hardware
    pub fn new_filled_on(loc: &Location, shape: &[usize], value: T) -> Self {
        Self::from_buffer(Buffer::new_filled(loc, shape.iter().product(), value), shape)
    }
    /// Create tensor filled with zeros on the host
    pub fn new_zeroed(shape: &[usize]) -> Self {
        Self::new_zeroed_on(&Location::Host, shape)
    }
    /// Create tensor filled with zeros on the specified hardware
    pub fn new_zeroed_on(loc: &Location, shape: &[usize]) -> Self {
        Self::new_filled_on(loc, shape, T::zero())
    }

    /// Returns shape of the tensor - slice containing all tensor dimensions.
    pub fn shape(&self) -> &[usize] {
        return self.shape.as_slice();
    }
    /// Returns a new tensor that shares the same data but has other shape.
    /// Failed if the product of all shape dimensions is not equal to buffer size.
    pub fn reshape(&self, shape: &[usize]) -> Self {
        Self::from_shared_buffer(self.buffer.clone(), shape)
    }
    /// Load flattened data from tensor to slice.
    pub fn load(&self, dst: &mut [T]) {
        self.buffer.load(dst);
    }
    /// Store data from slice to a tensor in a flattened manner.
    pub fn store(&mut self, src: &[T]) {
        Rc::make_mut(&mut self.buffer).store(src);
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_filled() {
        let value: i32 = -123;
        let a = Tensor::new_filled(&[4, 3, 2], value);

        let mut v = Vec::new();
        v.resize(24, 0);
        a.load(v.as_mut_slice());

        assert!(v.iter().all(|&x| x == value));
    }

    #[test]
    fn new_zeroed() {
        let a = Tensor::new_zeroed(&[4, 3, 2]);

        let mut v = Vec::new();
        v.resize(24, -1);
        a.load(v.as_mut_slice());

        assert!(v.iter().all(|&x| x == 0));
    }
}
