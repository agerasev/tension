#[cfg(feature = "device")]
use ocl::{Error as OclError};

pub enum Error {
    /// Shape dimensions or dimension number mismatch.
    ShapeMismatch(String),
    /// Mutable access is not exclusive - shared resource has more that one owner.
    NotExclusive(String),
    /// Some of OpenCL errors.
    #[cfg(feature = "device")]
    Ocl(OclError),
}
