use num_traits as num;
use num_complex::Complex;


pub trait Zero {
    fn zero() -> Self;
}
pub trait One {
    fn one() -> Self;
}

pub trait Num: num::Num {}

impl Num for u8 {}
impl Num for u16 {}
impl Num for u32 {}
impl Num for u64 {}

impl Num for i8 {}
impl Num for i16 {}
impl Num for i32 {}
impl Num for i64 {}

impl Num for f32 {}
impl Num for f64 {}

impl Num for Complex<f32> {}
impl Num for Complex<f64> {}


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


#[cfg(feature = "device")]
mod interop {
    use super::*;
    use ocl::{OclPrm};
    use num_complex_v01::{Complex as ComplexV01};


    pub trait Interop: Copy {
        type Dev: OclPrm + Copy;
        fn to_dev(self) -> Self::Dev;
        fn from_dev(x: Self::Dev) -> Self;
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

    impl Interop for u8 {
        type Dev = u8;
        fn to_dev(self) -> Self::Dev {
            self
        }
        fn from_dev(x: Self::Dev) -> Self {
            x
        }
    }
    impl Interop for u16 {
        type Dev = u16;
        fn to_dev(self) -> Self::Dev {
            self
        }
        fn from_dev(x: Self::Dev) -> Self {
            x
        }
    }
    impl Interop for u32 {
        type Dev = u32;
        fn to_dev(self) -> Self::Dev {
            self
        }
        fn from_dev(x: Self::Dev) -> Self {
            x
        }
    }
    impl Interop for u64 {
        type Dev = u64;
        fn to_dev(self) -> Self::Dev {
            self
        }
        fn from_dev(x: Self::Dev) -> Self {
            x
        }
    }

    impl Interop for i8 {
        type Dev = i8;
        fn to_dev(self) -> Self::Dev {
            self
        }
        fn from_dev(x: Self::Dev) -> Self {
            x
        }
    }
    impl Interop for i16 {
        type Dev = i16;
        fn to_dev(self) -> Self::Dev {
            self
        }
        fn from_dev(x: Self::Dev) -> Self {
            x
        }
    }
    impl Interop for i32 {
        type Dev = i32;
        fn to_dev(self) -> Self::Dev {
            self
        }
        fn from_dev(x: Self::Dev) -> Self {
            x
        }
    }
    impl Interop for i64 {
        type Dev = i64;
        fn to_dev(self) -> Self::Dev {
            self
        }
        fn from_dev(x: Self::Dev) -> Self {
            x
        }
    }

    impl Interop for f32 {
        type Dev = f32;
        fn to_dev(self) -> Self::Dev {
            self
        }
        fn from_dev(x: Self::Dev) -> Self {
            x
        }
    }
    impl Interop for f64 {
        type Dev = f64;
        fn to_dev(self) -> Self::Dev {
            self
        }
        fn from_dev(x: Self::Dev) -> Self {
            x
        }
    }

    impl Interop for Complex<f32> {
        type Dev = ComplexV01<f32>;
        fn to_dev(self) -> Self::Dev {
            Self::Dev::new(self.re, self.im)
        }
        fn from_dev(x: Self::Dev) -> Self {
            Self::new(x.re, x.im)
        }
    }
    impl Interop for Complex<f64> {
        type Dev = ComplexV01<f64>;
        fn to_dev(self) -> Self::Dev {
            Self::Dev::new(self.re, self.im)
        }
        fn from_dev(x: Self::Dev) -> Self {
            Self::new(x.re, x.im)
        }
    }
}

#[cfg(feature = "device")]
pub use interop::*;

#[cfg(feature = "device")]
pub trait Prm : Copy + PartialEq + Zero + One + Interop {}
#[cfg(not(feature = "device"))]
pub trait Prm : Copy + PartialEq + Zero + One {}
