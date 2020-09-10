use crate::{
    Prm,
    Buffer, Location,
    Error,
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

/// Tensor structure.
/// It consists of a contiguous one-dimensional array and a shape.
#[derive(Clone)]
pub struct Tensor<T: Prm> {
    shape: Vec<usize>,
    shared_data: Rc<Buffer<T>>,
}

impl<T: Prm> Tensor<T> {
    /// Create tensor from shared buffer and shape
    fn from_shared_buffer(rc_buffer: Rc<Buffer<T>>, shape: &[usize]) -> Result<Self, Error> {
        if rc_buffer.len() == shape.iter().product() {
            Ok(Self {
                shape: shape.iter().cloned().collect(),
                shared_data: rc_buffer,
            })
        } else {
            Err(Error::ShapeMismatch(format!(
                "Buffer length {:?} != shape dims product {:?} = {:?}",
                rc_buffer.len(), shape, shape.iter().product::<usize>()
            )))
        }
    }
    /// Create tensor from specified buffer and shape
    fn from_buffer(buffer: Buffer<T>, shape: &[usize]) -> Result<Self, Error> {
        Self::from_shared_buffer(Rc::new(buffer), shape)
    }
    /// Create uninitialized tensor on the host
    pub unsafe fn new_uninit(shape: &[usize]) -> Result<Self, Error> {
        Self::new_uninit_on(&Location::Host, shape)
    }
    /// Create unitialized tensor on the specified hardware
    pub unsafe fn new_uninit_on(loc: &Location, shape: &[usize]) -> Result<Self, Error> {
        Buffer::new_uninit(loc, shape.iter().product())
        .and_then(|buffer| Self::from_buffer(buffer, shape))
    }
    /// Create tensor and fill it with single value on the host
    pub fn new_filled(shape: &[usize], value: T) -> Result<Self, Error> {
        Self::new_filled_on(&Location::Host, shape, value)
    }
    /// Create tensor filled with value on the specified hardware
    pub fn new_filled_on(loc: &Location, shape: &[usize], value: T) -> Result<Self, Error> {
        Buffer::new_filled(loc, shape.iter().product(), value)
        .and_then(|buffer| Self::from_buffer(buffer, shape))
    }
    /// Create tensor filled with zeros on the host
    pub fn new_zeroed(shape: &[usize]) -> Result<Self, Error> {
        Self::new_zeroed_on(&Location::Host, shape)
    }
    /// Create tensor filled with zeros on the specified hardware
    pub fn new_zeroed_on(loc: &Location, shape: &[usize]) -> Result<Self, Error> {
        Self::new_filled_on(loc, shape, T::zero())
    }

    /// Returns shape of the tensor - slice containing all tensor dimensions.
    pub fn shape(&self) -> &[usize] {
        return self.shape.as_slice();
    }
    /// Returns a new tensor that shares the same data but has other shape.
    /// Failed if the product of all shape dimensions is not equal to buffer size.
    pub fn reshape(&self, shape: &[usize]) -> Result<Tensor<T>, Error> {
        Self::from_shared_buffer(self.shared_data.clone(), shape)
    }
    /// Load flattened data from tensor to slice.
    pub fn load(&self, dst: &mut [T]) -> Result<(), Error> {
        self.shared_data.load(dst)
    }
    /// Store data from slice to a tensor in a flattened manner.
    pub fn store(&mut self, src: &[T]) -> Result<(), Error> {
        match Rc::get_mut(&mut self.shared_data) {
            Some(buffer) => buffer.store(src),
            None => Err(Error::NotExclusive(
                format!("Tensor data have {} owners", Rc::strong_count(&self.shared_data))
            ))
        }
    }
}
