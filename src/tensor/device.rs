use crate::{
    Prm, Interop,
    Tensor,
    DeviceBuffer,
};
use std::{
    rc::Rc,
};


/// Tensor structure.
/// It consists of a contiguous one-dimensional array and a shape.
/// Tensor tries to reuse resources as long as possible and implements copy-on-write mechanism.
pub struct DeviceTensor<T: Prm + Interop> {
    shape: Vec<usize>,
    buffer: Rc<DeviceBuffer<T>>,
}

impl<T: Prm + Interop> DeviceTensor<T> {
    
}

impl<T: Prm + Interop> Tensor<T> for DeviceTensor<T> {
    type Buffer = DeviceBuffer<T>;

    unsafe fn from_shared_buffer_unchecked(rc_buffer: Rc<DeviceBuffer<T>>, shape: &[usize]) -> Self {
        Self {
            shape: shape.iter().cloned().collect(),
            buffer: rc_buffer,
        }
    }

    fn shape(&self) -> &[usize] {
        self.shape.as_slice()
    }

    unsafe fn shared_buffer(&self) -> &Rc<DeviceBuffer<T>> {
        &self.buffer
    }
    unsafe fn shared_buffer_mut(&mut self) -> &mut Rc<DeviceBuffer<T>> {
        &mut self.buffer
    }
}
