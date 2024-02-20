use rustfft::{num_complex::Complex, FftNum};

use crate::Czt;

pub struct NaiveCzt<T: FftNum> {
    a: Complex<T>,
    w: Complex<T>,
    czt_size: usize,
}

impl<T: FftNum> NaiveCzt<T> {
    pub fn new(czt_size: usize, a: Complex<T>, w: Complex<T>) -> Self {
        Self { czt_size, a, w }
    }
}

impl<T: FftNum> Czt<T> for NaiveCzt<T> {
    fn process_with_scratch(&self, buffer: &mut [Complex<T>], scratch: &mut [Complex<T>]) {
        for k in 0..self.czt_size {
            let z = self.a * self.w.powi(-1 * (k as i32));
            for n in 0..self.czt_size {
                scratch[k] = scratch[k] + buffer[n] * z.powi(-1 * (n as i32));
            }
        }

        for k in 0..self.czt_size {
            buffer[k] = scratch[k];
        }
    }

    fn get_scratch_len(&self) -> usize {
        self.czt_size
    }
}
