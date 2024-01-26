use std::{fmt::{Display, Formatter, Result}, fs::{create_dir, remove_dir_all, OpenOptions}};
use std::path::Path;
use std::io::{prelude::*, BufWriter};
use rand::Rng;

#[derive(Debug, Clone)]
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

impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
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
    pub fn convert_index_to_flat(&self, index: &[usize]) -> usize{
        if index.len() != self.dims.len(){
            panic!("Length of Index: {:?} does not match the Tensor's dims: {:?}", index, self.dims)
        }
        let mut flat_index = 0;
        for (i, dim) in index.iter().enumerate() {
            if i == index.len() - 1{
                flat_index += dim;
                continue;
            }
            flat_index += dim * self.dims[i + 1];
        }
        flat_index
    }
    pub fn index_flat(&self, index: usize) -> Option<f32> {
        if index >= self.size {
            panic!("Index: {:?} exceeds maximum lenght of: {:?}", index, self.size);
        }
        Some(self.data[index])
    }
    pub fn index(&self, index: &[usize]) -> f32 {
        if index.len() != self.dims.len(){
            panic!("Index: {:?} dimension does not match data dimension: {:?}", index, self.dims);
        }
        return self.data[self.convert_index_to_flat(index)];
    }
    pub fn set(&mut self, index: &[usize], val: f32){
        let i = self.convert_index_to_flat(index);
        self.data[i] = val;
    }

    pub fn matrix_muliply(self, matrix_b: Tensor) -> Tensor {
        if self.dims.len() != 2 || matrix_b.dims.len() != 2{
            panic!("Matrix multiplication not defined for dims: A{:?}, B{:?}", self.dims, matrix_b.dims);
        } else if self.dims[1] != matrix_b.dims[0]{
            panic!("Matrix A.dims[1]: {:?} must match matrix_b.dims[0]: {:?}", self.dims, matrix_b.dims);
        }
        let mut result = Tensor::new(&vec![self.dims[0], matrix_b.dims[1]]);
        for i in 0..self.dims[0]{
            for j in 0..matrix_b.dims[1]{
                for k in 0..self.dims[1]{
                    let a = self.index(&[i, k]);
                    let b = matrix_b.index(&[k, j]);
                    let c = a * b;
                    result.set(&[i, j], result.index(&[i, j]) + self.index(&[i, k]) * matrix_b.index(&[k, j]));
                }
            }
        }
        result
    }

    pub fn data(&self) -> &Vec<f32> {
        &self.data
    }
    
    pub fn log_data(&self) {
        if self.dims.len() > 3{
            panic!("Cannot display tensor with dimensions: {:?}", self.dims)
        } else if Path::new("temp_tensors").exists() {
            let _ = remove_dir_all("temp_tensors");
        }
        let _ = create_dir("temp_tensors");
        let mut file = BufWriter::new(OpenOptions::new()
            .create_new(true)
            .append(true)
            .open("temp_tensors/tensor.txt")
            .unwrap());
        
        if self.dims.len() == 3{
            for (i, val) in self.data.iter().enumerate(){
                writeln!(file, "{:?}", val).expect("Couldn't write to file");
                if i % self.dims[1] * self.dims[2] == 0 {
                    writeln!(file, "/n").expect("Couldn't write to file");
                }
                if i % self.dims[2] == 0{
                    writeln!(file, "/n").expect("Couldn't write to file");
                }
            }
        } else if self.dims.len() == 2 {
            writeln!(file, "dims: {:?} size: {}", self.dims, self.size).expect("Couldn't write to file");
            for i in 0..self.dims[0]{
                let mut line = "".to_owned();
                for j in 0..self.dims[1]{
                    line.push_str(&self.index(&[i, j]).to_string());
                    line.push_str(", ")
                }
                writeln!(file, "{}", line).expect("Couldn't write to file");
            }
        } else if self.dims.len() == 1 {
            for val in self.data(){
                writeln!(file, "{:?}", val).expect("Couldn't write to file");
            }
        }
    }

    pub fn randomize(&mut self){
        for i in 0..self.size{
            self.data[i] = rand::thread_rng().gen::<f32>() * 2.0 - 1.0;
        }
    }
    
    pub fn transposed(&self) -> Tensor{
        if self.dims.len() > 2{
            panic!("Transpose is not defined for dims: {:?}", self.dims)
        }
        let mut result = Tensor::new(&vec![self.dims[1], self.dims[0]]);
        for i in 0..self.dims[0]{
            for j in 0..self.dims[1]{
                result.set(&[j, i],  self.index(&[i, j]))
            }
        }
        result
    }

    pub fn matrix_multiply_transposed(&self, matrix_b: Tensor) -> Tensor{
        if self.dims.len() != 2 || matrix_b.dims.len() != 2{
            panic!("Matrix multiplication not defined for dims: A{:?}, B{:?}", self.dims, matrix_b.dims);
        } else if self.dims[1] != matrix_b.dims[0]{
            panic!("Matrix A.dims[1]: {:?} must match matrix_b.dims[0]: {:?}", self.dims, matrix_b.dims);
        }
        let mut result = Tensor::new(&vec![self.dims[0], matrix_b.dims[1]]);
        let mut bt: Vec<f32> = vec![0.0; matrix_b.size];
        transpose::transpose(&matrix_b.data, &mut bt, matrix_b.dims[1], matrix_b.dims[0]);

        for i in 0..self.dims[0]{
            for j in 0..matrix_b.dims[1]{
                let mut sum = 0.0;
                for k in 0..self.dims[1]{
                    sum += self.index(&[i, k]) * bt[j * matrix_b.dims[0] + k];
                }
                result.set(&[i, j], sum);
            }
        }
        result
    }

}