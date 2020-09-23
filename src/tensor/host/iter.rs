use std::{
    slice,
};
use crate::{
    Prm,
    Tensor, HostTensor,
};

/// Iterator over host tensor content.
pub struct HostTensorIter<'a, T: Prm> {
    tensor: &'a HostTensor<T>,
    position: Vec<usize>,
    first: bool,
    exhausted: bool,
}

/// Mutable iterator over host tensor content.
pub type HostTensorIterMut<'a, T> = slice::IterMut<'a, T>;

impl <'a, T: Prm> HostTensorIter<'a, T> {
    /// Create iterator over specified tensor.
    pub(crate) fn new(tensor: &'a HostTensor<T>) -> Self {
        let mut position = Vec::<usize>::new();
        position.resize(tensor.shape().len(), 0);
        Self {
            tensor,
            position,
            first: true,
            exhausted: false,
        }
    }

    fn next_position(&mut self) -> Option<&[usize]> {
        let shape = self.tensor.shape();
        if self.exhausted {
            None
        } else if self.first {
            if shape.content() > 0 {
                self.first = false;
                Some(self.position.as_slice())
            } else {
                self.exhausted = true;
                None
            }
        } else {
            let mut carry = true;
            for i in 0..shape.len() {
                if carry {
                    carry = false;
                    let mut pos = self.position[i] + 1;
                    if pos >= shape[i] {
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

    fn next_index(&mut self) -> Option<usize> {
        let shape = self.tensor.shape().clone();
        self.next_position().map(|slice| {
            slice.iter().zip(shape.iter())
            .fold((1, 0), |(stride, pos), (x, len)| {
                (stride*len, pos + stride*x)
            }).1
        })
    }
}

impl<'a, T: Prm> Iterator for HostTensorIter<'a, T> {
    type Item = &'a T;
    fn next(&mut self) -> Option<Self::Item> {
        self.next_index()
        .map(|pos| &self.tensor.buffer().as_slice()[pos])
    }
}
