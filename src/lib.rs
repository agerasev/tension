//use std::rc::Rc;
use num_traits::Num;

/*
struct Range {
    start: Option<isize>,
    end: Option<isize>,
    step: isize,
}

struct DimMap {
    start: usize,
    length: usize,
    stride: usize
}

struct TensorData<T: Num> {
    vec: Vec<T>,
}

struct Tensor<T: Num> {
    dims: Vec<DimMap>,
    shared_data: Rc<TensorData<T>>,
}
*/

pub enum Error {
    BadSize
}

pub struct Tensor<T: Num + Copy> {
    dims: Vec<usize>,
    data: Vec<T>,
}

impl<T: Num + Copy> Tensor<T> {
    pub fn zeros(shape: &[usize]) -> Self {
        let mut vec = Vec::new();
        vec.resize(shape.iter().product(), T::zero());
        Self {
            dims: shape.to_vec(),
            data: vec,
        }
    }
    pub fn shape(&self) -> &[usize] {
        return self.dims.as_slice();
    }
    pub fn reshape(&self, shape: &[usize]) -> Result<Tensor<T>, Error> {
        if self.data.len() == shape.iter().product() {
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
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
