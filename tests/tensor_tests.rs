#[cfg(test)]
mod tests {
    use compression_algorithm::tensor::Tensor;

    #[test]
    fn set_correctly_sets_2_2(){
        let mut t1 = Tensor::new(&vec![10 as usize, 50 as usize]);
        t1.set(&[1 as usize, 5 as usize], 66.0);
        t1.log_data();
        assert_eq!(t1.index(&[1 as usize, 5 as usize]), 66.0)
    }
    #[test]
    fn matrix_multiplication_correct_2x2(){
        let mut t1 = Tensor::new(&vec![2 as usize, 2 as usize]);
        let mut t2 = Tensor::new(&vec![2 as usize, 2 as usize]);
        t1.set(&[0 as usize, 0 as usize], 1.0);
        t1.set(&[0 as usize, 1 as usize], 2.0);
        t1.set(&[1 as usize, 0 as usize], 3.0);
        t1.set(&[1 as usize, 1 as usize], 4.0);
        t2.set(&[0 as usize, 0 as usize], 5.0);
        t2.set(&[0 as usize, 1 as usize], 6.0);
        t2.set(&[1 as usize, 0 as usize], 7.0);
        t2.set(&[1 as usize, 1 as usize], 8.0);
        let t3 = t1.matrix_muliply(t2);
        println!("{:?}", t3.data());
        assert_eq!(t3.data(), &vec![19.0, 22.0, 43.0, 50.0]);

    }
    #[test]
    fn matrix_multiplication_correct_3x1_1x2(){
        let mut t1 = Tensor::new(&vec![3 as usize, 1 as usize]);
        let mut t2 = Tensor::new(&vec![1 as usize, 2 as usize]);
        t1.set(&[0 as usize, 0 as usize], 1.0);
        t1.set(&[1 as usize, 0 as usize], 2.0);
        t1.set(&[2 as usize, 0 as usize], 3.0);
        t2.set(&[0 as usize, 0 as usize], 4.0);
        t2.set(&[0 as usize, 1 as usize], 5.0);
        let t3 = t1.matrix_muliply(t2);
        println!("{:?}", t3);
        println!("eeeeeeee {:?}", t3.index(&[2, 0]));
        assert_eq!(t3.data(), &vec![4.0, 5.0, 8.0, 10.0, 12.0, 15.0]);
    }
    #[test]
    fn convert_index_to_flat_works_1_2(){
        let t1 = Tensor::new(&vec![10, 50]);
        let flattened_index = t1.convert_index_to_flat(&[1, 2]);
        assert_eq!(flattened_index, 52)
    }
    #[test]
    fn convert_index_to_flat_works_3_5(){
        let t1 = Tensor::new(&vec![10, 50]);
        let flattened_index = t1.convert_index_to_flat(&[3, 5]);
        assert_eq!(flattened_index, 155)
    }
    #[test]
    fn matrix_multiply_transposed_test(){
        let mut t1 = Tensor::new(&vec![10, 50]);
        let mut t2 = Tensor::new(&vec![50, 10]);
        t1.randomize();
        t2.randomize();
        assert_eq!(t1.clone().matrix_muliply(t2.clone()), t1.matrix_multiply_transposed(t2));
    }
}