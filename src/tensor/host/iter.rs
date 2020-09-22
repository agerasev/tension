use crate::{
    Prm,
    Shape,
    Tensor, HostTensor,
};


struct PositionCounter {
    shape: Shape,
    position: Vec<usize>,
    first: bool,
    exhausted: bool,
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

/// Iterator over host tensor content.
pub struct HostTensorIter<'a, T: Prm> {
    slice: &'a [T],
    counter: PositionCounter,
}

/// Mutable iterator over host tensor content.
///
/// *This iterator doesn't implement `Iterator` trait due to lifetime issue,
/// but provides the similar `next` method instead.*
pub struct HostTensorIterMut<'a, T: Prm> {
    slice: &'a mut [T],
    counter: PositionCounter,
}

impl <'a, T: Prm> HostTensorIter<'a, T> {
    /// Create iterator over specified tensor.
    pub fn new(tensor: &'a HostTensor<T>) -> Self {
        Self {
            slice: tensor.as_slice(),
            counter: PositionCounter::new(tensor.shape().clone()),
        }
    }
}

impl <'a, T: Prm> HostTensorIterMut<'a, T> {
    /// Create iterator over specified tensor.
    pub fn new(tensor: &'a mut HostTensor<T>) -> Self {
        let shape = tensor.shape().clone();
        Self {
            slice: tensor.as_mut_slice(),
            counter: PositionCounter::new(shape),
        }
    }
    /// Iterate to next element.
    ///
    /// *This method is provided instead of `Iterator::next` -
    /// with additional lifetime requirement.*
    pub fn next<'b: 'a>(&'b mut self) -> Option<&'a mut T> {
        let pos = self.counter.next()?;
        Some(&mut self.slice[pos])
    }
}

impl<'a, T: Prm> Iterator for HostTensorIter<'a, T> {
    type Item = &'a T;
    fn next(&mut self) -> Option<Self::Item> {
        self.counter.next()
        .map(|pos| &self.slice[pos])
    }
}

//impl<'a, T: Prm> Iterator for HostTensorIterMut<'a, T> {
//    type Item = &'a mut T;
//    fn next(&mut self) -> Option<Self::Item> {
//        self.counter.next()
//        .map(|pos| &mut self.slice[pos])
//    }
//}
