#[cfg(feature = "device")]
use ocl::{Error as OclError};

pub enum Error {
    ShapeMismatch(String),
    #[cfg(feature = "device")]
    OclError(OclError),
}
