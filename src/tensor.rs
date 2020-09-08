use crate::{
    Prm,
    Buffer, Location,
    Error,
};
use std::{
    iter::once,
    rc::Rc,
};


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

#[derive(Clone)]
pub struct Tensor<T: Prm> {
    shape: Vec<usize>,
    stride: Vec<usize>,
    shared_data: Rc<Buffer<T>>,
}

impl<T: Prm> Tensor<T> {
    fn from_buffer(buffer: Buffer<T>, shape: &[usize]) -> Result<Self, Error> {
        if buffer.len() == shape.iter().product() {
            Ok(Self {
                shape: shape.iter().cloned().collect(),
                stride: once(0).chain(shape.iter().scan(1, |state, &x| {
                    *state = *state * x;
                    Some(*state)
                })).collect(),
                shared_data: Rc::new(buffer),
            })
        } else {
            Err(Error::ShapeMismatch(format!(
                "Buffer length {:?} != shape product {:?} = {:?}",
                buffer.len(), shape, shape.iter().product::<usize>()
            )))
        }
    }
    pub unsafe fn new_uninit(shape: &[usize]) -> Result<Self, Error> {
        Self::new_uninit_on(&Location::Host, shape)
    }
    pub unsafe fn new_uninit_on(loc: &Location, shape: &[usize]) -> Result<Self, Error> {
        Buffer::new_uninit(loc, shape.iter().product())
        .and_then(|buffer| Self::from_buffer(buffer, shape))
    }
    pub fn new_filled(shape: &[usize], value: T) -> Result<Self, Error> {
        Self::new_filled_on(&Location::Host, shape, value)
    }
    pub fn new_filled_on(loc: &Location, shape: &[usize], value: T) -> Result<Self, Error> {
        Buffer::new_filled(loc, shape.iter().product(), value)
        .and_then(|buffer| Self::from_buffer(buffer, shape))
    }
    pub fn new_zeroed(shape: &[usize]) -> Result<Self, Error> {
        Self::new_zeroed_on(&Location::Host, shape)
    }
    pub fn new_zeroed_on(loc: &Location, shape: &[usize]) -> Result<Self, Error> {
        Self::new_filled_on(loc, shape, T::zero())
    }

    pub fn shape(&self) -> &[usize] {
        return self.shape.as_slice();
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
