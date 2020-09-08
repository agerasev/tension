use num_complex::Complex;


#[cfg(not(feature = "device"))]
mod wrap {
    use super::*;

    pub trait Prm: PartialEq + Copy {}

    impl Prm for bool {}
    
    impl Prm for u8 {}
    impl Prm for u16 {}
    impl Prm for u32 {}
    impl Prm for u64 {}
    
    impl Prm for i8 {}
    impl Prm for i16 {}
    impl Prm for i32 {}
    impl Prm for i64 {}
    
    impl Prm for f32 {}
    impl Prm for f64 {}
    
    impl Prm for Complex<f32> {}
    impl Prm for Complex<f64> {}    
}


#[cfg(feature = "device")]
mod wrap {
    use super::*;
    use ocl::{OclPrm};
    use num_complex_v01::{Complex as ComplexV01};


    pub trait Prm: PartialEq + Copy {
        type Dev: OclPrm + Copy;
        fn to_dev(self) -> Self::Dev;
        fn from_dev(x: Self::Dev) -> Self;
    }
    
    impl Prm for bool {
        type Dev = u8;
        fn to_dev(self) -> Self::Dev {
            if self {
                0xFF
            } else {
                0x00
            }
        }
        fn from_dev(x: Self::Dev) -> Self {
            if x != 0 {
                true
            } else {
                false
            }
        }
    }

    impl Prm for u8 {
        type Dev = u8;
        fn to_dev(self) -> Self::Dev {
            self
        }
        fn from_dev(x: Self::Dev) -> Self {
            x
        }
    }
    impl Prm for u16 {
        type Dev = u16;
        fn to_dev(self) -> Self::Dev {
            self
        }
        fn from_dev(x: Self::Dev) -> Self {
            x
        }
    }
    impl Prm for u32 {
        type Dev = u32;
        fn to_dev(self) -> Self::Dev {
            self
        }
        fn from_dev(x: Self::Dev) -> Self {
            x
        }
    }
    impl Prm for u64 {
        type Dev = u64;
        fn to_dev(self) -> Self::Dev {
            self
        }
        fn from_dev(x: Self::Dev) -> Self {
            x
        }
    }

    impl Prm for i8 {
        type Dev = i8;
        fn to_dev(self) -> Self::Dev {
            self
        }
        fn from_dev(x: Self::Dev) -> Self {
            x
        }
    }
    impl Prm for i16 {
        type Dev = i16;
        fn to_dev(self) -> Self::Dev {
            self
        }
        fn from_dev(x: Self::Dev) -> Self {
            x
        }
    }
    impl Prm for i32 {
        type Dev = i32;
        fn to_dev(self) -> Self::Dev {
            self
        }
        fn from_dev(x: Self::Dev) -> Self {
            x
        }
    }
    impl Prm for i64 {
        type Dev = i64;
        fn to_dev(self) -> Self::Dev {
            self
        }
        fn from_dev(x: Self::Dev) -> Self {
            x
        }
    }

    impl Prm for f32 {
        type Dev = f32;
        fn to_dev(self) -> Self::Dev {
            self
        }
        fn from_dev(x: Self::Dev) -> Self {
            x
        }
    }
    impl Prm for f64 {
        type Dev = f64;
        fn to_dev(self) -> Self::Dev {
            self
        }
        fn from_dev(x: Self::Dev) -> Self {
            x
        }
    }

    impl Prm for Complex<f32> {
        type Dev = ComplexV01<f32>;
        fn to_dev(self) -> Self::Dev {
            Self::Dev::new(self.re, self.im)
        }
        fn from_dev(x: Self::Dev) -> Self {
            Self::new(x.re, x.im)
        }
    }
    impl Prm for Complex<f64> {
        type Dev = ComplexV01<f64>;
        fn to_dev(self) -> Self::Dev {
            Self::Dev::new(self.re, self.im)
        }
        fn from_dev(x: Self::Dev) -> Self {
            Self::new(x.re, x.im)
        }
    }
}


pub use wrap::*;
