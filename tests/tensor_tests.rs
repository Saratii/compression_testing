#[cfg(test)]
mod tests {
    use compression_algorithm::tensor::Tensor;

    #[test]
    fn set_correctly_sets(){
        let mut t1 = Tensor::new(&vec![1 as usize, 2 as usize, 2 as usize]);
        t1.set(&[0 as usize, 1 as usize, 0 as usize], 66.0);
        assert_eq!(t1.index(&[0 as usize, 1 as usize, 0 as usize]).unwrap(), 66.0)
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
        let t3 = t1.matrix_muliply(t2).unwrap();
        println!("{:?}", t3.data());
        assert_eq!(t3.data(), &vec![19.0, 22.0, 43.0, 50.0]);

    }
    #[test]
    fn matrix_multiplication_correct_4x1_1x2(){
        let mut t1 = Tensor::new(&vec![3 as usize, 1 as usize]);
        let mut t2 = Tensor::new(&vec![1 as usize, 2 as usize]);
        t1.set(&[0 as usize, 0 as usize], 1.0);
        t1.set(&[1 as usize, 0 as usize], 2.0);
        t1.set(&[2 as usize, 0 as usize], 3.0);
        t2.set(&[0 as usize, 0 as usize], 4.0);
        t2.set(&[0 as usize, 1 as usize], 5.0);
        let t3 = t1.matrix_muliply(t2).unwrap();
        println!("{:?}", t3.data());
        assert_eq!(t3.data(), &vec![4.0, 5.0, 8.0, 10.0, 12.0, 15.0]);

    }
    #[test]
    fn index_to_flat_works(){
        let mut t1 = Tensor::new(&vec![2 as usize, 2 as usize]);
        t1.set(&[0 as usize, 0 as usize], 1.0);
        t1.set(&[0 as usize, 1 as usize], 2.0);
        t1.set(&[1 as usize, 0 as usize], 3.0);
        t1.set(&[1 as usize, 1 as usize], 4.0);
        assert_eq!(t1.index_flat(t1.index_to_flat(&[0, 0])).unwrap(), 1.0);
        assert_eq!(t1.index_flat(t1.index_to_flat(&[0, 1])).unwrap(), 2.0);
        assert_eq!(t1.index_flat(t1.index_to_flat(&[1, 0])).unwrap(), 3.0);
        assert_eq!(t1.index_flat(t1.index_to_flat(&[1, 1])).unwrap(), 4.0);
    }
}