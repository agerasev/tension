pub enum Error<B: Backend> {
    Backend(B::Error),
}

pub trait DType {}

/// Scalar value.
pub trait Value {}

pub enum AnyValue {}
impl Value for AnyValue {}

pub trait FinalValue: Value + Copy + Default {}

/// Owned data.
#[allow(clippy::len_without_is_empty)]
pub trait Data<B: Backend + ?Sized, T: Value = AnyValue> {
    fn len(&self) -> Result<usize, B::Error>;

    fn upcast_dtype<U: Value>(self) -> Result<B::Data<U>, B::Error>;
    fn downcast_dtype<U: Value>(self) -> Result<B::Data<U>, B::Error>;

    fn to_slice(&self, slice: &mut [T]) -> Result<(), B::Error>
    where
        T: FinalValue;

    fn to_vec(&self) -> Result<Vec<T>, B::Error>
    where
        T: FinalValue,
    {
        // TODO: Use `reserve` and then `to_uninit_slice`.
        let mut data = vec![T::default(); self.len()?];
        self.to_slice(&mut data)?;
        Ok(data)
    }
}

pub trait Backend {
    type DType;
    type Data<T: Value>: Data<Self, T>;
    type Error;
    fn alloc_data<T: FinalValue>(&self, slice: &[T]) -> Result<Self::Data<T>, Self::Error>;
}
