use crate::{
    Prm,
    Buffer, Location,
    Error,
};
use std::rc::Rc;


#[derive(Clone, Copy, Debug)]
pub struct Range {
    pub start: Option<isize>,
    pub end: Option<isize>,
    pub step: isize,
}

#[derive(Clone, Copy, Debug)]
pub enum Index {
    Single(isize),
    Range(Range),
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Slicing {
    pub start: usize,
    pub length: usize,
    pub stride: isize
}
impl Slicing {
    fn new(start: usize, length: usize, stride: isize) -> Self {
        Slicing { start, length, stride }
    }
}

#[derive(Clone)]
pub struct Tensor<T: Prm> {
    dims: Vec<Slicing>,
    shared_data: Rc<Buffer<T>>,
}

impl<T: Prm> Tensor<T> {
    pub unsafe fn new_uninit(shape: &[usize]) -> Result<Self, Error> {
        Self::new_uninit_loc(&Location::Host, shape)
    }
    pub unsafe fn new_uninit_loc(loc: &Location, shape: &[usize]) -> Result<Self, Error> {
        let len = shape.iter().product();
        Buffer::new_uninit(loc, len).map(|buf| Self {
            dims: shape.iter().map(|s| Slicing::new(0, *s, 1)).collect(),
            shared_data: Rc::new(buf),
        })
    }
    pub fn new_filled(shape: &[usize], value: T) -> Result<Self, Error> {
        Self::new_filled_loc(&Location::Host, shape, value)
    }
    pub fn new_filled_loc(loc: &Location, shape: &[usize], value: T) -> Result<Self, Error> {
        let len = shape.iter().product();
        Buffer::new_filled(loc, len, value).map(|buf| Self {
            dims: shape.iter().map(|s| Slicing::new(0, *s, 1)).collect(),
            shared_data: Rc::new(buf),
        })
    }
    pub fn new_zeroed(shape: &[usize]) -> Result<Self, Error> {
        Self::new_zeroed_loc(&Location::Host, shape)
    }
    pub fn new_zeroed_loc(loc: &Location, shape: &[usize]) -> Result<Self, Error> {
        Self::new_filled_loc(loc, shape, T::zero())
    }

    pub fn shape(&self) -> Vec<usize> {
        return self.dims.iter().map(|s| s.length).collect();
    }
    /*
    pub fn reshape(&self, shape: &[usize]) -> Result<Tensor<T>, Error> {
        if self.shared_data.len() == shape.iter().product() {
            Ok(Self {
                dims: shape.to_vec(),
                data: self.data.clone(),
            })
        } else {
            Err(Error::BadSize)
        }
    }
    pub fn load(&self, dst: &mut [T]) -> Result<(), Error> {
        if self.data.len() == dst.len() {
            dst.copy_from_slice(self.data.as_slice());
            Ok(())
        } else {
            Err(Error::BadSize)
        }
    }
    pub fn store(&mut self, src: &[T]) -> Result<(), Error> {
        if self.data.len() == src.len() {
            self.data.copy_from_slice(src);
            Ok(())
        } else {
            Err(Error::BadSize)
        }
    }
    */
}
