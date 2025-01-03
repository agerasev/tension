pub trait Type {
    fn id(&self) -> TypeId;
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum TypeId {
    Bool,
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    U64,
    I64,
    // F16,
    // BF16,
    F32,
    F64,
}

pub trait Scalar: Copy {
    const TYPE_ID: TypeId;
}

macro_rules! impl_cpu_type {
    ($type:ty, $id:ident) => {
        impl Scalar for $type {
            const TYPE_ID: TypeId = TypeId::$id;
        }
    };
}

impl_cpu_type!(bool, Bool);

impl_cpu_type!(u8, U8);
impl_cpu_type!(i8, I8);
impl_cpu_type!(u16, U16);
impl_cpu_type!(i16, I16);
impl_cpu_type!(u32, U32);
impl_cpu_type!(i32, I32);
impl_cpu_type!(u64, U64);
impl_cpu_type!(i64, I64);

// impl_cpu_type!(f16, F16);
// impl_cpu_type!(bf16, Bf16);
impl_cpu_type!(f32, F32);
impl_cpu_type!(f64, F64);
