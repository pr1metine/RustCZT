use rustfft::{num_complex::Complex, num_traits::Zero, Direction, FftNum, Length};

pub mod plan;

pub fn add(left: usize, right: usize) -> usize {
    left + right
}

pub trait Czt<T: FftNum>: Sync + Send {
    fn process(&self, buffer: &mut [Complex<T>]) {
        let mut scratch = vec![Complex::zero(); buffer.len()];
        self.process_with_scratch(buffer, &mut scratch);
    }

    fn process_with_scratch(&self, buffer: &mut [Complex<T>], scratch: &mut [Complex<T>]);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
