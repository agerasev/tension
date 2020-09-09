use crate::{Prm, Error};

#[cfg(feature = "device")]
use ocl::{Buffer as OclBuffer, Queue, MemFlags};


pub struct HostBuffer<T: Prm> {
    vec: Vec<T>,
}
impl<T: Prm> HostBuffer<T> {
    pub unsafe fn new_uninit(len: usize) -> Result<Self, Error> {
        let mut vec = Vec::<T>::with_capacity(len);
        vec.set_len(len);
        Ok(Self { vec })
    }
    pub fn new_filled(len: usize, value: T) -> Result<Self, Error> {
        let mut vec = Vec::<T>::new();
        vec.resize(len, value);
        Ok(Self { vec })
    }
    pub fn len(&self) -> usize {
        self.vec.len()
    }
    pub fn as_slice(&self) -> &[T] {
        self.vec.as_slice()
    }
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.vec.as_mut_slice()
    }
}

#[cfg(feature = "device")]
pub struct DeviceBuffer<T: Prm> {
    mem: OclBuffer<T::Dev>,
}
#[cfg(feature = "device")]
impl<T: Prm> DeviceBuffer<T> {
    pub unsafe fn new_uninit(queue: &Queue, len: usize) -> Result<Self, Error> {
        OclBuffer::builder()
        .queue(queue.clone())
        .flags(MemFlags::READ_WRITE)
        .len(len)
        .build()
        .map(|mem| DeviceBuffer { mem })
        .map_err(|e| Error::Ocl(e))
    }
    pub fn new_filled(queue: &Queue, len: usize, value: T) -> Result<Self, Error> {
        OclBuffer::builder()
        .queue(queue.clone())
        .flags(MemFlags::READ_WRITE)
        .len(len)
        .fill_val(value.to_dev())
        .build()
        .map(|mem| DeviceBuffer { mem })
        .map_err(|e| Error::Ocl(e))
    }
    pub fn len(&self) -> usize {
        self.mem.len()
    }
    /*
    pub fn load(&self, dst: &mut [T]) -> Result<(), Error> {
        if self.len() == dst.len() {
            let mut dst_dev = Vec::<T::Dev>::with_capacity(dst.len());
            dst_dev.set_len(dst.len());
            self.mem.read(&mut dst_dev).enq()
            .map_err(|e| Error::Ocl(e))
            .and_then(|_| {
                dst_dev.iter().map(|&x| T::from_dev(x))
                Ok(())
            })
        } else {
            Err(Error::ShapeMismatch(format!(
                "buffer length {} != dst length {}", self.len(), dst.len()
            )))
        }
    }
    pub fn store(&mut self, src: &[T]) -> Result<(), Error> {
        if self.len() == src.len() {

        } else {
            Err(Error::ShapeMismatch(format!(
                "buffer length {} != src length {}", self.len(), src.len()
            )))
        }
    }
    */
}

pub enum Buffer<T: Prm> {
    Host(HostBuffer<T>),
    #[cfg(feature = "device")]
    Device(DeviceBuffer<T>),
}

#[derive(Clone, Debug)]
pub enum Location {
    Host,
    #[cfg(feature = "device")]
    Device(Queue),
}

impl PartialEq for Location {
    fn eq(&self, other: &Self) -> bool {
        match self {
            Self::Host => match other {
                Self::Host => true,
                #[cfg(feature = "device")]
                _ => false,
            },

            #[cfg(feature = "device")]
            Self::Device(sq) => match other {
                Self::Device(oq) => sq.as_ptr() == oq.as_ptr(),
                _ => false,
            }
        }
    }
}

impl<T: Prm> Buffer<T> {
    pub unsafe fn new_uninit(location: &Location, len: usize) -> Result<Self, Error> {
        match location {
            Location::Host => {
                HostBuffer::new_uninit(len)
            }.map(|hbuf| Self::Host(hbuf)),

            #[cfg(feature = "device")]
            Location::Device(queue) => {
                DeviceBuffer::new_uninit(queue, len)
            }.map(|dbuf| Self::Device(dbuf)),
        }
    }
    pub fn new_filled(location: &Location, len: usize, value: T) -> Result<Self, Error> {
        match location {
            Location::Host => {
                HostBuffer::new_filled(len, value)
            }.map(|hbuf| Self::Host(hbuf)),

            #[cfg(feature = "device")]
            Location::Device(queue) => {
                DeviceBuffer::new_filled(queue, len, value)
            }.map(|dbuf| Self::Device(dbuf)),
        }
    }

    pub fn location(&self) -> Location {
        match self {
            Self::Host(_) => Location::Host,

            #[cfg(feature = "device")]
            Self::Device(buf) => Location::Device(buf.mem.default_queue().unwrap().clone()),
        }
    }
    pub fn len(&self) -> usize {
        match self {
            Self::Host(hbuf) => hbuf.len(),

            #[cfg(feature = "device")]
            Self::Device(dbuf) => dbuf.len(),
        }
    }
}
