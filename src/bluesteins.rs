use std::sync::Arc;

use rustfft::{
    num_complex::Complex,
    num_traits::{Float, FromPrimitive, Zero},
    Fft, FftNum, FftPlanner,
};

use crate::Czt;

pub struct BluesteinsAlgorithm<T: FftNum> {
    y_coefficients: Vec<Complex<T>>,
    v_coefficients: Vec<Complex<T>>,
    x_coefficients: Vec<Complex<T>>,
    fft_forward: Arc<dyn Fft<T>>,
}

impl<T: FftNum + Float> BluesteinsAlgorithm<T> {
    pub fn new(
        n: usize,
        m: usize,
        a: Complex<T>,
        w: Complex<T>,
        fft_planner: &mut FftPlanner<T>,
    ) -> Self {
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
            l: usize,
            m: usize,
            n: usize,
            w: Complex<T>,
            fft_forward: Arc<dyn Fft<T>>,
        ) -> Vec<Complex<T>> {
            let mut out: Vec<_> = (0..m as i32)
                .map(|n| w.powf(-square_and_half::<T>(n)))
                .chain((m..l - n + 1).map(|_| Complex::zero()))
                .chain((l - n + 1..l).map(|n| w.powf(-square_and_half::<T>((l - n) as i32))))
                .collect();
            fft_forward.process(&mut out);
            out
        }
        fn compute_x_coefficients<T: Float + FftNum>(m: usize, w: Complex<T>) -> Vec<Complex<T>> {
            (0..m as i32).map(|k| w.powf(square_and_half(k))).collect()
        }

        let l = (m + n - 1).next_power_of_two();

        let fft_forward = fft_planner.plan_fft_forward(l);

        let y_coefficients = compute_y_coefficients(n, a, w);
        let v_coefficients = compute_v_coefficients(l, m, n, w, fft_forward.clone());
        let x_coefficients = compute_x_coefficients(m, w);

        Self {
            y_coefficients,
            v_coefficients,
            x_coefficients,
            fft_forward,
        }
    }
}

impl<T: FftNum> BluesteinsAlgorithm<T> {
    fn m(&self) -> usize {
        self.x_coefficients.len()
    }

    fn n(&self) -> usize {
        self.y_coefficients.len()
    }

    fn l(&self) -> usize {
        self.fft_forward.len()
    }
}

impl<T: FftNum> Czt<T> for BluesteinsAlgorithm<T> {
    fn process_with_scratch(&self, buffer: &mut [Complex<T>], scratch: &mut [Complex<T>]) {
        assert_eq!(buffer.len(), self.n());
        assert_eq!(scratch.len(), self.get_scratch_len());

        let (expanded_buffer, scratch) = scratch.split_at_mut(self.l());

        // Perform step one of CZT: y_n = x_n * A^-n * W ^ (n^2 / 2)
        for i in 0..self.n() {
            expanded_buffer[i] = buffer[i] * self.y_coefficients[i];
        }
        for i in self.n()..self.l() {
            expanded_buffer[i] = Complex::zero();
        }

        // Perform step two of CZT
        self.fft_forward
            .process_with_scratch(expanded_buffer, scratch);

        for i in 0..self.l() {
            expanded_buffer[i] = (expanded_buffer[i] * self.v_coefficients[i]).conj();
        }

        self.fft_forward
            .process_with_scratch(expanded_buffer, scratch);

        let l = T::from_usize(self.l()).unwrap();

        // Perform step three of CZT
        for i in 0..self.m() {
            buffer[i] = expanded_buffer[i].conj() * self.x_coefficients[i] / l;
        }
    }

    fn get_scratch_len(&self) -> usize {
        self.fft_forward.get_inplace_scratch_len() + self.l()
    }
}
