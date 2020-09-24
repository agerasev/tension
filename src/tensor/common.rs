use std::{
    rc::Rc,
    marker::PhantomData,
};
use crate::{
    Prm, Buffer, Shape, Tensor,
};

/// An intermediate structure that contains most of the Tensor functionality.
pub struct CommonTensor<T: Prm, Buf: Buffer<T>> {
    buffer: Rc<Buf>,
    pub shape: Shape,
    phantom: PhantomData<T>,
}

impl<T: Prm, Buf: Buffer<T>> CommonTensor<T, Buf> {
    /// Create tensor from shared buffer and shape
    pub fn from_shared_buffer(rc_buffer: Rc<Buf>, shape: &Shape) -> Self {
        Self {
            buffer: rc_buffer,
            shape: shape.clone(),
            phantom: PhantomData::<T>,
        }
    }
    /// Create tensor from specified buffer and shape
    pub fn from_buffer(buffer: Buf, shape: &Shape) -> Self {
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

    unsafe fn new_uninit_in(context: &Buf::Context, shape: &Shape) -> Self {
        Self::from_buffer(
            Self::Buffer::new_uninit_in(context, shape.iter().product()),
            shape,
        )
    }
    fn new_filled_in(context: &Buf::Context, shape: &Shape, value: T) -> Self {
        Self::from_buffer(
            Self::Buffer::new_filled_in(context, shape.iter().product(), value),
            shape,
        )
    }
    fn new_zeroed_in(context: &Buf::Context, shape: &Shape) -> Self {
        Self::new_filled_in(context, shape, T::zero())
    }

    fn shape(&self) -> &Shape {
        &self.shape
    }

    fn reshape(&self, shape: &Shape) -> Self {
        Self::from_shared_buffer(self.buffer.clone(), shape)
    }

    fn load(&self, dst: &mut [T]) {
        self.buffer.load(dst);
    }
    fn store(&mut self, src: &[T]) {
        Rc::make_mut(&mut self.buffer).store(src);
    }
}
