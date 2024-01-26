use std::time::Instant;

pub fn benchmark<F>(func: F, iterations: i32)
where
  F: Fn(),
{
    let start_time = Instant::now();
    for _ in 0..iterations{
        let _ = func();
    }
    let end_time = Instant::now();
    println!("Benchmarking Function: {:?}", name(func));
    println!("Completed {} iterations in a total of {} ms", iterations, (end_time - start_time).as_millis());
    println!("Average time per iteration: {}", (end_time - start_time).as_millis() / iterations as u128);
}

fn name<F: Fn()>(_: F) {
    println!("{}", std::any::type_name::<F>());
}