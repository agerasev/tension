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

struct PositionCounter {
    shape: Shape,
    position: Vec<usize>,
    first: bool,
    exhausted: bool,
}

pub struct HostTensorIter<'a, T: Prm> {
    slice: &'a [T],
    counter: PositionCounter,
}

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

    pub fn iter<'a>(&'a self) -> HostTensorIter<'a, T> {
        HostTensorIter::new(self)
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

impl PositionCounter {
    fn new(shape: Shape) -> Self {
        let mut position = Vec::<usize>::new();
        position.resize(shape.len(), 0);
        Self {
            shape, position,
            first: true,
            exhausted: false,
        }
    }

    fn next_slice(&mut self) -> Option<&[usize]> {
        if self.exhausted {
            None
        } else if self.first {
            if self.shape.content() > 0 {
                self.first = false;
                Some(self.position.as_slice())
            } else {
                self.exhausted = true;
                None
            }
        } else {
            let mut carry = true;
            for i in 0..self.shape.len() {
                if carry {
                    carry = false;
                    let mut pos = self.position[i] + 1;
                    if pos >= self.shape[i] {
                        pos = 0;
                        carry = true;
                    }
                    self.position[i] = pos;
                }
            }
            if carry {
                self.exhausted = true;
                None
            } else {
                Some(self.position.as_slice())
            }
        }
    }

    fn next(&mut self) -> Option<usize> {
        let shape = self.shape.clone();
        self.next_slice().map(|slice| {
            slice.iter().zip(shape.iter())
            .fold((1, 0), |(stride, pos), (x, len)| {
                (stride*len, pos + stride*x)
            }).1
        })
    }
}

impl <'a, T: Prm> HostTensorIter<'a, T> {
    fn new(tensor: &'a HostTensor<T>) -> Self {
        Self {
            slice: tensor.as_slice(),
            counter: PositionCounter::new(tensor.shape().clone()),
        }
    }
}

impl<'a, T: Prm> Iterator for HostTensorIter<'a, T> {
    type Item = &'a T;
    fn next(&mut self) -> Option<Self::Item> {
        self.counter.next()
        .map(|pos| &self.slice[pos])
    }
}
