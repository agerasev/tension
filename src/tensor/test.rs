use crate::{prelude::*, Shape, HostTensor as Tensor};

#[test]
fn new_filled() {
    let value: i32 = -123;
    let a = Tensor::new_filled(&Shape::from([4, 3, 2].as_ref()), value);

    let mut v = Vec::new();
    v.resize(24, 0);
    a.load(v.as_mut_slice());

    assert!(v.iter().all(|&x| x == value));
}

#[test]
fn new_zeroed() {
    let a = Tensor::new_zeroed(&Shape::from([4, 3, 2].as_ref()));

    let mut v = Vec::new();
    v.resize(24, -1);
    a.load(v.as_mut_slice());

    assert!(v.iter().all(|&x| x == 0));
}
