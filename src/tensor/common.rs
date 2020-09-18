use std::{
    iter::once,
    rc::Rc,
};
use crate::{
    Prm, Buffer,
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
    /// Marker for adding a new dimension.
    NewAxis,
}

pub trait Tensor<T: Prm, B: Buffer<T>>: Sized {
    /// Create tensor from shared buffer and shape
    unsafe fn from_shared_buffer_unchecked(rc_buffer: Rc<B>, shape: &[usize], strides: &[isize]) -> Self;

    /// Create tensor from shared buffer and shape
    fn from_shared_buffer(rc_buffer: Rc<B>, shape: &[usize], strides: &[isize]) -> Self {
        assert_eq!(shape.len() + 1, strides.len());
        for i in 0..shape.len() {
            // TODO: Check all indices are inside bounds
        }
        Self::from_shared_buffer_unchecked(rc_buffer, shape, strides)
    }
    /// Create tensor from specified buffer and shape
    fn from_buffer(buffer: B, shape: &[usize], strides: &[isize]) -> Self {
        Self::from_shared_buffer(Rc::new(buffer), shape, strides)
    }
    /// Create tensor from specified buffer and shape
    unsafe fn from_buffer_plain(buffer: B, shape: &[usize]) -> Self {
        assert_eq!(buffer.len(), shape.iter().product());
        Self::from_buffer(
            buffer, shape,
            shape.iter().scan(1, |state, &x| {
                *state = *state * (x as isize);
                Some(*state)
            }).chain(once(0))
            .collect::<Vec<_>>().as_slice(),
        )
    }

    /// Create unitialized tensor
    unsafe fn new_uninit_in(context: &B::Context, shape: &[usize]) -> Self {
        Self::from_buffer_plain(
            B::new_uninit_in(context, shape.iter().product()),
            shape,
        )
    }
    /// Create tensor filled with value on the specified hardware
    fn new_filled_in(context: &B::Context, shape: &[usize], value: T) -> Self {
        Self::from_buffer_plain(
            B::new_filled_in(context, shape.iter().product(), value),
            shape,
        )
    }
    /// Create tensor filled with zeros on the specified hardware
    fn new_zeroed_in(context: &B::Context, shape: &[usize]) -> Self {
        Self::new_filled_in(context, shape, T::zero())
    }

    /// Shape of the tensor - a slice containing all tensor dimensions.
    fn shape(&self) -> &[usize];

    /// Strides of the tensor.
    fn strides(&self) -> &[isize];

    /// Returns a new tensor that shares the same data but has other shape.
    /// Failed if the product of all shape dimensions is not equal to buffer size.
    fn reshape(&self, shape: &[usize]) -> Self;

    /// Load flattened data from tensor to slice.
    fn load(&self, dst: &mut [T]);

    /// Store data from slice to a tensor in a flattened manner.
    fn store(&mut self, src: &[T]);
}
