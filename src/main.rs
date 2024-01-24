use crate::tensor::Tensor;

mod tensor;

fn main() {
    let t = Tensor::new(&vec![3, 2]);
    println!("{}", t.index(&[2 as usize, 1 as usize]).unwrap());
    println!("{}", t);
}
