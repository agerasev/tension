use crate::{
    Prm,
    HostBuffer,
    Tensor, CommonTensor,
};


type InnerTensor<T> = CommonTensor<T, HostBuffer<T>>;

/// Tensor structure.
/// It consists of a contiguous one-dimensional array and a shape.
/// Tensor tries to reuse resources as long as possible and implements copy-on-write mechanism.
pub struct HostTensor<T: Prm> {
    inner: InnerTensor<T>,
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

    unsafe fn new_uninit_in(_: &(), shape: &[usize]) -> Self {
        Self { inner: InnerTensor::<T>::new_uninit_in(&(), shape) }
    }
    fn new_filled_in(_: &(), shape: &[usize], value: T) -> Self {
        Self { inner: InnerTensor::<T>::new_filled_in(&(), shape, value) }
    }
    fn new_zeroed_in(_: &(), shape: &[usize]) -> Self {
        Self { inner: InnerTensor::<T>::new_zeroed_in(&(), shape) }
    }

    fn shape(&self) -> &[usize] {
        self.inner.shape()
    }

    fn reshape(&self, shape: &[usize]) -> Self {
        Self { inner: self.inner.reshape(shape) }
    }

    fn load(&self, dst: &mut [T]) {
        self.inner.load(dst);
    }
    fn store(&mut self, src: &[T]) {
        self.inner.store(src);
    }
}
