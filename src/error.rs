#[cfg(feature = "device")]
use ocl::{Error as OclError};

pub enum Error {
    /// Buffer length mismatch.
    LengthMismatch(usize, usize),
    /// Tensor shape dimension number or some of dimensions mismatch.
    ShapeMismatch(String),
    /// Mutable access is not exclusive - shared resource has more that one owner.
    NotExclusive(String),
    /// Some of OpenCL errors.
    #[cfg(feature = "device")]
    Ocl(OclError),
}
