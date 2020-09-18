use num_traits as num;
use num_complex::Complex;

/// Analog of `num_traits::Zero` but also implemented for `bool` type.
pub trait Zero {
    fn zero() -> Self;
}
/// Analog of `num_traits::One` but also implemented for `bool` type.
pub trait One {
    fn one() -> Self;
}

/// Wrapper for `num_traits::Num`.
pub trait Num: num::Num {}

/// Wrapper for `num_traits::Float`.
pub trait Float: Num + num::Float {}

impl Num for u8 {}
impl Num for u16 {}
impl Num for u32 {}
impl Num for u64 {}

impl Num for i8 {}
impl Num for i16 {}
impl Num for i32 {}
impl Num for i64 {}

impl Num for usize {}
impl Num for isize {}

impl Num for f32 {}
impl Num for f64 {}

impl Float for f32 {}
impl Float for f64 {}

impl<T: Float> Num for Complex<T> {}


impl<T: Num> Zero for T {
    fn zero() -> Self {
        <T as num::Zero>::zero()
    }
}
impl Zero for bool {
    fn zero() -> Self {
        false
    }
}
impl<T: Num> One for T {
    fn one() -> Self {
        <T as num::One>::one()
    }
}
impl One for bool {
    fn one() -> Self {
        false
    }
}


/// Type that could be put in tensor.
pub trait Prm : Sized + Copy + PartialEq + Zero + One {}

impl<T: Num + Copy> Prm for T {}

impl Prm for bool {}


#[cfg(feature = "device")]
mod interop {
    use super::*;
    use std::mem::transmute;
    use ocl::{OclPrm, Buffer};
    use num_complex_v01::{Complex as ComplexV01};


    /// Types that can be transformed from host representation to device one and back.
    pub trait Interop: Copy {
        type Dev: OclPrm + Copy;

        /// Transform from host to device type.
        fn to_dev(self) -> Self::Dev;
        /// Transform from device to host type.
        fn from_dev(x: Self::Dev) -> Self;

        /// Copy data from OpenCL buffer to host slice.
        fn load_from_buffer(dst: &mut [Self], src: &Buffer<Self::Dev>) {
            assert_eq!(dst.len(), src.len());
            let mut tmp = Vec::<Self::Dev>::new();
            src.read(&mut tmp).enq().unwrap();
            for (d, &s) in dst.iter_mut().zip(tmp.iter()) {
                *d = Self::from_dev(s);
            }
        }

        /// Copy data from host slice to OpenCL buffer.
        fn store_to_buffer(dst: &mut Buffer<Self::Dev>, src: &[Self]) {
            assert_eq!(dst.len(), src.len());
            let tmp = src.iter().map(|x| x.to_dev()).collect::<Vec<_>>();
            dst.write(&tmp).enq().unwrap();
        }
    }

    /// Type which representation remains the same for both host and device.
    pub trait IdentInterop: Interop<Dev=Self> + OclPrm {}

    impl <T: IdentInterop> Interop for T {
        type Dev = Self;

        fn to_dev(self) -> Self::Dev {
            self
        }
        fn from_dev(x: Self::Dev) -> Self {
            x
        }
        fn load_from_buffer(dst: &mut [Self], src: &Buffer<Self::Dev>) {
            assert_eq!(dst.len(), src.len());
            src.read(dst).enq().unwrap();
        }
        fn store_to_buffer(dst: &mut Buffer<Self::Dev>, src: &[Self]) {
            assert_eq!(dst.len(), src.len());
            dst.write(src).enq().unwrap();
        }
    }

    impl Interop for bool {
        type Dev = u8;
        fn to_dev(self) -> Self::Dev {
            if self {
                0xFF
            } else {
                0x00
            }
        }
        fn from_dev(x: Self::Dev) -> Self {
            x != 0
        }
    }

    impl IdentInterop for u8 {}
    impl IdentInterop for u16 {}
    impl IdentInterop for u32 {}
    impl IdentInterop for u64 {}

    impl IdentInterop for i8 {}
    impl IdentInterop for i16 {}
    impl IdentInterop for i32 {}
    impl IdentInterop for i64 {}

    impl IdentInterop for f32 {}
    impl IdentInterop for f64 {}

    impl Interop for usize {
        type Dev = u32;
        fn to_dev(self) -> Self::Dev {
            self as Self::Dev
        }
        fn from_dev(x: Self::Dev) -> Self {
            x as Self
        }
    }
    impl Interop for isize {
        type Dev = i32;
        fn to_dev(self) -> Self::Dev {
            self as Self::Dev
        }
        fn from_dev(x: Self::Dev) -> Self {
            x as Self
        }
    }

    impl<T: Float> Interop for Complex<T> where ComplexV01<T>: OclPrm {
        type Dev = ComplexV01<T>;
        fn to_dev(self) -> Self::Dev {
            Self::Dev::new(self.re, self.im)
        }
        fn from_dev(x: Self::Dev) -> Self {
            Self::new(x.re, x.im)
        }
        fn load_from_buffer(dst: &mut [Self], src: &Buffer<Self::Dev>) {
            assert_eq!(dst.len(), src.len());
            src.read(
                unsafe { transmute::<_, &mut [Self::Dev]>(dst) }
            ).enq().unwrap();
        }
        fn store_to_buffer(dst: &mut Buffer<Self::Dev>, src: &[Self]) {
            assert_eq!(dst.len(), src.len());
            dst.write(
                unsafe { transmute::<_, &[Self::Dev]>(src) }
            ).enq().unwrap();
        }
    }
}
#[cfg(feature = "device")]
pub use interop::*;
