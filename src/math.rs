pub fn sufficient_statistics(x: &f64) -> [f64; 2] {
    [*x, x.powf(2.0)]
}