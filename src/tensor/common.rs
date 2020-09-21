use std::{
    rc::Rc,
    marker::PhantomData,
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

/// Tensor a.k.a. N-dimensional array.
pub trait Tensor<T: Prm>: Sized {
    /// Inner buffer type.
    type Buffer : Buffer<T>;

    /// Create unitialized tensor
    unsafe fn new_uninit_in(context: &<Self::Buffer as Buffer<T>>::Context, shape: &[usize]) -> Self;
    /// Create tensor filled with value on the specified hardware
    fn new_filled_in(context: &<Self::Buffer as Buffer<T>>::Context, shape: &[usize], value: T) -> Self;
    /// Create tensor filled with zeros on the specified hardware
    fn new_zeroed_in(context: &<Self::Buffer as Buffer<T>>::Context, shape: &[usize]) -> Self;

    /// Shape of the tensor - a slice containing all tensor dimensions.
    fn shape(&self) -> &[usize];

    /// Returns a new tensor that shares the same data but has other shape.
    /// Failed if the product of all shape dimensions is not equal to buffer size.
    fn reshape(&self, shape: &[usize]) -> Self;

    /// Load flattened data from tensor to slice.
    fn load(&self, dst: &mut [T]);
    /// Store data from slice to a tensor in a flattened manner.
    fn store(&mut self, src: &[T]);
}

/// An intermediate structure that contains most of the Tensor functionality.
pub struct CommonTensor<T: Prm, Buf: Buffer<T>> {
    pub buffer: Rc<Buf>,
    pub shape: Vec<usize>,
    phantom: PhantomData<T>,
}

impl<T: Prm, Buf: Buffer<T>> CommonTensor<T, Buf> {
    /// Create tensor from shared buffer and shape
    pub fn from_shared_buffer(rc_buffer: Rc<Buf>, shape: &[usize]) -> Self {
        Self {
            buffer: rc_buffer,
            shape: shape.iter().cloned().collect(),
            phantom: PhantomData::<T>,
        }
    }
    /// Create tensor from specified buffer and shape
    pub fn from_buffer(buffer: Buf, shape: &[usize]) -> Self {
        assert_eq!(buffer.len(), shape.iter().product());
        Self::from_shared_buffer(Rc::new(buffer), shape)
    }

    /// Provides access to inner buffer.
    pub fn buffer(&self) -> &Buf {
        self.buffer.as_ref()
    }
    /// Clones inner buffer if it is shared and provides mutable access to it.
    pub fn buffer_mut(&mut self) -> &mut Buf {
        Rc::make_mut(&mut self.buffer)
    }
}

impl<T: Prm, Buf: Buffer<T>> Tensor<T> for CommonTensor<T, Buf> {
    type Buffer = Buf;

    unsafe fn new_uninit_in(context: &Buf::Context, shape: &[usize]) -> Self {
        Self::from_buffer(
            Self::Buffer::new_uninit_in(context, shape.iter().product()),
            shape,
        )
    }
    fn new_filled_in(context: &Buf::Context, shape: &[usize], value: T) -> Self {
        Self::from_buffer(
            Self::Buffer::new_filled_in(context, shape.iter().product(), value),
            shape,
        )
    }
    fn new_zeroed_in(context: &Buf::Context, shape: &[usize]) -> Self {
        Self::new_filled_in(context, shape, T::zero())
    }

    fn shape(&self) -> &[usize] {
        self.shape.as_slice()
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
