use std::{
    ops::{Index, IndexMut},
    slice::{Iter, IterMut},
};


fn count_non_one(slice: &[usize]) -> usize {
    slice.iter().cloned().rev().skip_while(|&x| x == 1).count()
}
fn trim_slice(slice: &[usize]) -> &[usize] {
    &slice[..count_non_one(slice)]
}
fn trim_mut_slice(slice: &mut [usize]) -> &mut [usize] {
    let len = count_non_one(slice);
    &mut slice[..len]
}
fn trim_vec(vec: &mut Vec<usize>) {
    vec.truncate(count_non_one(vec.as_slice()));
}

/// Structure representing shape of the tensor.
///
/// Tensor supposed to have an infinite number of axes. E.g. tensor of shape `(x,y,z)` supposed to be `(x,y,z,1,1,1,...)`.
/// That means that trailing axes of size `1` are ignored, so shapes `(x,y,z)` and `(x,y,z,1)` are equal.
///
/// There may be a tensor of shape `(,)` (0-dimensional tensor or scalar).
#[derive(Clone, Debug)]
pub struct Shape {
    vec: Vec<usize>,
}

impl From<Vec<usize>> for Shape {
    fn from(vec: Vec<usize>) -> Self {
        Self { vec }
    }
}
impl From<&[usize]> for Shape {
    fn from(slice: &[usize]) -> Self {
        Self::from(slice.iter().cloned().collect::<Vec<_>>())
    }
}

impl Into<Vec<usize>> for Shape {
    fn into(mut self) -> Vec<usize> {
        trim_vec(&mut self.vec);
        self.vec
    }
}

impl Shape {
    /// Count of dimensions without trailing `1`s in the end.
    pub fn len(&self) -> usize {
        count_non_one(self.vec.as_slice())
    }

    /// Slice of dimension sizes.
    pub fn as_slice(&self) -> &[usize] {
        trim_slice(self.vec.as_slice())
    }
    /// Mutable slice of dimension sizes.
    pub fn as_mut_slice(&mut self) -> &mut [usize] {
        trim_mut_slice(self.vec.as_mut_slice())
    }

    /// Iterator over dimension sizes.
    pub fn iter(&self) -> Iter<usize> {
        trim_slice(self.vec.as_slice()).iter()
    }
    /// Mutable iterator over dimension sizes.
    pub fn iter_mut(&mut self) -> IterMut<usize> {
        trim_mut_slice(self.vec.as_mut_slice()).iter_mut()
    }
}

impl PartialEq<Shape> for Shape {
    fn eq(&self, other: &Shape) -> bool {
        trim_slice(self.vec.as_slice()) == trim_slice(other.vec.as_slice())
    }
}

impl Index<usize> for Shape {
    type Output = usize;

    fn index(&self, i: usize) -> &usize {
        if i < self.vec.len() {
            &self.vec[i]
        } else {
            &1
        }
    }
}

impl IndexMut<usize> for Shape {
    fn index_mut(&mut self, i: usize) -> &mut usize {
        if i >= self.vec.len() {
            self.vec.resize(i + 1, 1);
        }
        &mut self.vec[i]
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from() {
        let shape = Shape::from([1, 2, 3].as_ref());
        assert_eq!(shape.as_slice(), [1, 2, 3]);
    }

    #[test]
    fn trim() {
        let shape = Shape::from([1, 2, 1, 3, 1, 1].as_ref());
        assert_eq!(shape.len(), 4);
        assert_eq!(shape.as_slice(), [1, 2, 1, 3]);
    }

    #[test]
    fn eq() {
        assert_eq!(
            Shape::from([1, 2, 1, 3, 1].as_ref()),
            Shape::from([1, 2, 1, 3].as_ref()),
        );
    }

    #[test]
    fn index() {
        let mut shape = Shape::from([1, 2, 1, 3, 1].as_ref());

        assert_eq!(shape[1], 2);
        assert_eq!(shape[5], 1);
        assert_eq!(shape.as_slice(), [1, 2, 1, 3]);

        shape[5] = 1;
        assert_eq!(shape.as_slice(), [1, 2, 1, 3]);

        shape[5] = 4;
        assert_eq!(shape.as_slice(), [1, 2, 1, 3, 1, 4]);
    }

    #[test]
    fn iter() {
        let mut shape = Shape::from([1, 2, 1].as_ref());

        let mut iter = shape.iter();
        assert_eq!(iter.next(), Some(&1));
        assert_eq!(iter.next(), Some(&2));
        assert_eq!(iter.next(), None);

        let mut iter_mut = shape.iter_mut();
        *iter_mut.next().unwrap() = 3;
        assert_eq!(shape, Shape::from([3, 2, 1].as_ref()));
    }
}
