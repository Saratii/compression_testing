use std::fmt::{Display, Formatter, Result};

#[derive(Debug)]
pub struct Tensor {
    dims: Vec<usize>,
    data: Vec<f32>,
    size: usize,
}

impl Display for Tensor {
    fn fmt(&self, f: &mut Formatter) -> Result {
        write!(f, "Dims: {:?} Size: {}", self.dims, self.size)
    }
}

impl Tensor {
    pub fn new(dims: &Vec<usize>) -> Tensor{
        let mut prod = 1;
        for dim in dims{
            prod *= dim;
        }
        Tensor{dims: dims.clone(), data: vec![0.0; prod], size: prod}
    }
    fn index_to_flat(&self, index: &[usize]) -> usize{
        let mut i = 0;
        let mut stride = 1;
        for (dim, len) in index.iter().zip(&self.dims) {
            i += dim * stride;
            stride *= len;
        }
        i
    }
    pub fn index(&self, index: &[usize]) -> Option<f32> {
        if index.len() != self.dims.len(){
            panic!("index dimension does not match data dimension");
        }
        let mut i = 0;
        let mut stride = 1;
        for (dim, len) in index.iter().zip(&self.dims) {
            i += dim * stride;
            stride *= len;
        }
        return Some(self.data[i]);
    }
    pub fn set(&mut self, index: &[usize], val: f32){
        (|| {
            let i = self.index_to_flat(index);
            self.data[i] = val;
            Ok(())
        })().unwrap_or_else(|_err: String| {
            panic!("Failed to set value at index: {:?}", index);
        }) 
    }

    pub fn matrix_muliply(self, matrix_b: Tensor) -> Option<Tensor> {
        if self.dims.len() != 2 || matrix_b.dims.len() != 2{
            panic!("Matrix multiplication not defined for dims: A{:?}, B{:?}", self.dims, matrix_b.dims);
        } else if self.dims[1] != matrix_b.dims[0]{
            panic!("Matrix A.dims[1]: {:?} must match matrix_b.dims[0]: {:?}", self.dims, matrix_b.dims);
        }
        let mut result = Tensor::new(&vec![self.dims[0], matrix_b.dims[1]]);
        for i in 0..self.dims[0]{
            for j in 0..matrix_b.dims[1]{
                for k in 0..matrix_b.dims[0]{
                    result.set(&[i, j], result.index(&[i, j]).unwrap() + self.index(&[i, k]).unwrap() * matrix_b.index(&[k, j]).unwrap())
                }
            }
        }
        Some(result)
    }

    pub fn data(&self) -> &Vec<f32>{
        &self.data
    }

}