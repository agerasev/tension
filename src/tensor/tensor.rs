use crate::{
    Prm, Buffer, Shape,
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

/// Tensor a.k.a. N-dimensional array.
pub trait Tensor<T: Prm>: Sized {
    /// Inner buffer type.
    type Buffer : Buffer<T>;

    /// Create unitialized tensor
    unsafe fn new_uninit_in(context: &<Self::Buffer as Buffer<T>>::Context, shape: &Shape) -> Self;
    /// Create tensor filled with value on the specified hardware
    fn new_filled_in(context: &<Self::Buffer as Buffer<T>>::Context, shape: &Shape, value: T) -> Self;
    /// Create tensor filled with zeros on the specified hardware
    fn new_zeroed_in(context: &<Self::Buffer as Buffer<T>>::Context, shape: &Shape) -> Self;

    /// Shape of the tensor - a slice containing all tensor dimensions.
    fn shape(&self) -> &Shape;

    /// Returns a new tensor that shares the same data but has other shape.
    /// Failed if the product of all shape dimensions is not equal to buffer size.
    fn reshape(&self, shape: &Shape) -> Self;

    /// Load flattened data from tensor to slice.
    fn load(&self, dst: &mut [T]);
    /// Store data from slice to a tensor in a flattened manner.
    fn store(&mut self, src: &[T]);
}
