use rustfft::{num_complex::Complex, num_traits::Zero, FftNum};

pub mod plan;
pub use plan::CztPlanner;

pub trait Czt<T: FftNum>: Sync + Send {
    fn process(&self, buffer: &mut [Complex<T>]) {
        let mut scratch = vec![Complex::zero(); buffer.len()];
        self.process_with_scratch(buffer, &mut scratch);
    }

    fn process_with_scratch(&self, buffer: &mut [Complex<T>], scratch: &mut [Complex<T>]);
}
