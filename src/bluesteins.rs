use std::sync::Arc;

use rustfft::{
    num_complex::Complex,
    num_traits::{Float, FromPrimitive, Zero},
    Fft, FftNum,
};

use crate::Czt;

pub struct BluesteinsAlgorithm<T: FftNum> {
    y_coefficients: Vec<Complex<T>>,
    v_coefficients: Vec<Complex<T>>,
    x_coefficients: Vec<Complex<T>>,
    fft_forward: Arc<dyn Fft<T>>,
}

impl<T: FftNum + Float> BluesteinsAlgorithm<T> {
    pub fn new(czt_len: usize, a: Complex<T>, w: Complex<T>, fft_forward: Arc<dyn Fft<T>>) -> Self {
        fn square_and_half<T>(n: i32) -> T
        where
            T: Float + FromPrimitive,
        {
            T::from_i32(n * n).unwrap() / T::from_usize(2).unwrap()
        }
        fn compute_y_coefficients<T: Float + FftNum>(
            n: usize,
            a: Complex<T>,
            w: Complex<T>,
        ) -> Vec<Complex<T>> {
            (0..n as i32)
                .map(|n| a.powi(-n) * w.powf(square_and_half(n)))
                .collect()
        }
        fn compute_v_coefficients<T: Float + FftNum>(
            n: usize,
            w: Complex<T>,
            fft_forward: Arc<dyn Fft<T>>,
        ) -> Vec<Complex<T>> {
            let mut out: Vec<_> = (0..n as i32)
                .map(|n| w.powf(-square_and_half::<T>(n)))
                .chain([Complex::zero()])
                .chain((n + 1..2 * n).map(|i| w.powf(-square_and_half::<T>((2 * n - i) as i32))))
                .collect();
            fft_forward.process(&mut out);
            out
        }
        fn compute_x_coefficients<T: Float + FftNum>(m: usize, w: Complex<T>) -> Vec<Complex<T>> {
            (0..m as i32).map(|k| w.powf(square_and_half(k))).collect()
        }

        let y_coefficients = compute_y_coefficients(czt_len, a, w);
        let v_coefficients = compute_v_coefficients(czt_len, w, fft_forward.clone());
        let x_coefficients = compute_x_coefficients(czt_len, w);

        Self {
            y_coefficients,
            v_coefficients,
            x_coefficients,
            fft_forward,
        }
    }
}

impl<T: FftNum> Czt<T> for BluesteinsAlgorithm<T> {
    fn process_with_scratch(&self, buffer: &mut [Complex<T>], scratch: &mut [Complex<T>]) {
        let (expanded_buffer, scratch) =
            scratch.split_at_mut(self.fft_forward.get_inplace_scratch_len());

        // Perform step one of CZT: y_n = x_n * A^-n * W ^ (n^2 / 2)
        for i in 0..buffer.len() {
            expanded_buffer[i] = buffer[i] * self.y_coefficients[i];
        }
        for i in buffer.len()..buffer.len() * 2 {
            expanded_buffer[i] = Complex::zero();
        }

        // Perform step two of CZT
        self.fft_forward
            .process_with_scratch(expanded_buffer, scratch);

        for i in 0..expanded_buffer.len() {
            expanded_buffer[i] = (expanded_buffer[i] * self.v_coefficients[i]).conj();
        }

        self.fft_forward
            .process_with_scratch(expanded_buffer, scratch);

        let n = T::from_usize(self.fft_forward.len().into()).unwrap();

        // Perform step three of CZT
        for i in 0..buffer.len() {
            buffer[i] = expanded_buffer[i].conj() * self.x_coefficients[i] / n;
        }
    }

    fn get_scratch_len(&self) -> usize {
        self.fft_forward.get_inplace_scratch_len() * 2
    }
}
