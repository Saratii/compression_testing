

use compression_algorithm::benchmark::benchmark;

use crate::tensor::Tensor;

mod tensor;

fn main() {
    let mut tm1 = Tensor::new(&vec![400, 400]);
    let mut tm2 = Tensor::new(&vec![400, 400]);
    tm1.randomize();
    tm2.randomize();
    benchmark( || {tm1.clone().matrix_muliply(tm2.clone()); }, 50);
    benchmark( || {tm1.clone().matrix_multiply_transposed(tm2.clone()); }, 50);
}
