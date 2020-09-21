use crate::{
    Prm,
    HostBuffer,
    Shape, Tensor, CommonTensor,
};


type InnerTensor<T> = CommonTensor<T, HostBuffer<T>>;

/// Tensor structure.
/// It consists of a contiguous one-dimensional array and a shape.
/// Tensor tries to reuse resources as long as possible and implements copy-on-write mechanism.
pub struct HostTensor<T: Prm> {
    inner: InnerTensor<T>,
}

//pub struct HostTensorIter<'a, T: Prm> {
//    tensor: &'a HostTensor<T>,
//    pos: Vec<usize>,
//    exhausted: bool,
//}

//pub struct HostTensorIterMut<'a, T: Prm> {
//    tensor: &'a mut HostTensor<T>,
//    pos: Vec<usize>,
//}

impl<T: Prm> HostTensor<T> {
    /// Create unitialized tensor
    pub unsafe fn new_uninit(shape: &Shape) -> Self {
        Self::new_uninit_in(&(), shape)
    }
    /// Create tensor filled with value on the specified hardware
    pub fn new_filled(shape: &Shape, value: T) -> Self {
        Self::new_filled_in(&(), shape, value)
    }
    /// Create tensor filled with zeros on the specified hardware
    pub fn new_zeroed(shape: &Shape) -> Self {
        Self::new_zeroed_in(&(), shape)
    }

    /// Provideas access to underlying memory.
    pub fn as_slice(&self) -> &[T] {
        self.inner.buffer().as_slice()
    }
    /// Provideas mutable access to underlying memory.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.inner.buffer_mut().as_mut_slice()
    }
}

impl<T: Prm> Tensor<T> for HostTensor<T> {
    type Buffer = HostBuffer<T>;

    unsafe fn new_uninit_in(_: &(), shape: &Shape) -> Self {
        Self { inner: InnerTensor::<T>::new_uninit_in(&(), shape) }
    }
    fn new_filled_in(_: &(), shape: &Shape, value: T) -> Self {
        Self { inner: InnerTensor::<T>::new_filled_in(&(), shape, value) }
    }
    fn new_zeroed_in(_: &(), shape: &Shape) -> Self {
        Self { inner: InnerTensor::<T>::new_zeroed_in(&(), shape) }
    }

    fn shape(&self) -> &Shape {
        self.inner.shape()
    }

    fn reshape(&self, shape: &Shape) -> Self {
        Self { inner: self.inner.reshape(shape) }
    }

    fn load(&self, dst: &mut [T]) {
        self.inner.load(dst);
    }
    fn store(&mut self, src: &[T]) {
        self.inner.store(src);
    }
}

//impl <'a, T: Prm> HostTensorIter<'a, T> {
//    fn new(tensor: &'a HostTensor<T>) -> Self {
//        Self {
//            tensor,
//            pos: tensor.shape().iter().map(|_| 0).collect(),
//            exhausted: false,
//        }
//    }
//}

//fn next_pos(pos: &mut [usize], shape: &[usize]) -> bool {
//    let len = pos.len();
//    assert_eq!(len, shape.len());
//    if len > 0 {
//        if pos[len - 1] + 1 < shape[len - 1] {
//            pos[len - 1] += 1;
//            true
//        } else {
//            if next_pos(&mut pos[..(len - 1)], &shape[..(len - 1)]) {
//                pos[len - 1] = 0;
//                true
//            } else {
//                false
//            }
//        }
//    } else {
//        false
//    }
//}

//impl <'a, T: Prm> HostTensorIterMut<'a, T> {
//
//}

//impl<'a, T: Prm> Iterator for HostTensorIter<'a, T> {
//    type Item = &'a T;
//    fn next(&mut self) -> Option<Self::Item> {
//        let elem = self.tensor.as_slice()[pos]
//    }
//}

//impl<'a, T: Prm> Iterator for HostTensorIterMut<'a, T> {
//    type Item = &'a mut T;
//    fn next(&mut self) -> Option<Self::Item> {
//
//    }
//}
