use std::{
    cmp,
    ops::{Index, IndexMut, RangeBounds, Bound},
    slice::{Iter, IterMut},
};

#[macro_export]
macro_rules! shape {
    [ $( $x:expr ),* $(,)? ] => {
        $crate::Shape::from([ $( $x ),* ].as_ref())
    };
}

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

impl PartialEq<Shape> for Shape {
    fn eq(&self, other: &Shape) -> bool {
        trim_slice(self.vec.as_slice()) == trim_slice(other.vec.as_slice())
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

impl Shape {
    /// Slicing operation.
    ///
    /// Returns a new shape sliced from the original one.
    ///
    /// Sadly, you cannot use `shape[a..b]` syntax because Rust `Index` trait is required to return a reference.
    pub fn slice<R: RangeBounds<usize>>(&self, range: R) -> Shape {
        let len = self.len();
        let sidx = cmp::min(match range.start_bound() {
            Bound::Included(i) => *i,
            Bound::Excluded(i) => *i + 1,
            Bound::Unbounded => 0,
        }, len);
        let eidx = cmp::min(match range.end_bound() {
            Bound::Included(i) => *i + 1,
            Bound::Excluded(i) => *i,
            Bound::Unbounded => len,
        }, len);
        Self::from(&self.as_slice()[sidx..eidx])
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
    fn macro_() {
        let shape = shape![1, 2, 3];
        assert_eq!(shape.as_slice(), [1, 2, 3]);
    }

    #[test]
    fn trim() {
        let shape = shape![1, 2, 1, 3, 1, 1];
        assert_eq!(shape.len(), 4);
        assert_eq!(shape, shape![1, 2, 1, 3]);
    }

    #[test]
    fn eq() {
        assert_eq!(
            shape![1, 2, 1, 3, 1],
            shape![1, 2, 1, 3],
        );
    }

    #[test]
    fn index() {
        let mut shape = shape![1, 2, 1, 3, 1];

        assert_eq!(shape[1], 2);
        assert_eq!(shape[5], 1);
        assert_eq!(shape, shape![1, 2, 1, 3]);

        shape[5] = 1;
        assert_eq!(shape, shape![1, 2, 1, 3]);

        shape[5] = 4;
        assert_eq!(shape, shape![1, 2, 1, 3, 1, 4]);
    }

    #[test]
    fn iter() {
        let mut shape = shape![1, 2, 1];

        let mut iter = shape.iter();
        assert_eq!(iter.next(), Some(&1));
        assert_eq!(iter.next(), Some(&2));
        assert_eq!(iter.next(), None);

        let mut iter_mut = shape.iter_mut();
        *iter_mut.next().unwrap() = 3;
        assert_eq!(shape, shape![3, 2, 1]);
    }

    #[test]
    fn slice() {
        let shape = shape![1, 2, 1, 3, 1];

        assert_eq!(shape.slice(..), shape);
        assert_eq!(shape.slice(..3), shape![1, 2]);
        assert_eq!(shape.slice(1..=3), shape![2, 1, 3]);
        assert_eq!(shape.slice(2..5), shape![1, 3]);
        assert_eq!(shape.slice(5..10), shape![]);
        assert_eq!(shape.slice(5..), shape![]);
    }
}
