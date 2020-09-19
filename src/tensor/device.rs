use crate::{
    Prm, Interop,
    DeviceBuffer, DeviceContext,
    Tensor, CommonTensor,
};

type InnerTensor<T> = CommonTensor<T, DeviceBuffer<T>>;

/// Tensor structure.
/// It consists of a contiguous one-dimensional array and a shape.
/// Tensor tries to reuse resources as long as possible and implements copy-on-write mechanism.
pub struct DeviceTensor<T: Prm + Interop> {
    inner: InnerTensor<T>,
}

impl<T: Prm + Interop> DeviceTensor<T> {}

impl<T: Prm + Interop> Tensor<T> for DeviceTensor<T> {
    type Buffer = DeviceBuffer<T>;

    unsafe fn new_uninit_in(context: &DeviceContext, shape: &[usize]) -> Self {
        Self { inner: InnerTensor::<T>::new_uninit_in(context, shape) }
    }
    fn new_filled_in(context: &DeviceContext, shape: &[usize], value: T) -> Self {
        Self { inner: InnerTensor::<T>::new_filled_in(context, shape, value) }
    }
    fn new_zeroed_in(context: &DeviceContext, shape: &[usize]) -> Self {
        Self { inner: InnerTensor::<T>::new_zeroed_in(context, shape) }
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
