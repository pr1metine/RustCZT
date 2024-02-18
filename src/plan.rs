use std::sync::Arc;

use rustfft::{num_complex::Complex, num_traits::Float, Fft, FftNum, FftPlanner};

use crate::Czt;

pub enum ChosenCztPlanner<T: FftNum> {
    Scalar(CztPlannerScalar<T>),
}

pub struct CztPlanner<T: FftNum> {
    chosen_planner: ChosenCztPlanner<T>,
}

impl<T: FftNum> CztPlanner<T> {
    pub fn new() -> Self {
        Self {
            chosen_planner: ChosenCztPlanner::<T>::Scalar(CztPlannerScalar::<T>::new()),
        }
    }

    pub fn plan_czt_forward(
        &mut self,
        czt_len: usize,
        a: Complex<T>,
        w: Complex<T>,
    ) -> Arc<dyn Czt<T>> {
        match &mut self.chosen_planner {
            ChosenCztPlanner::Scalar(planner) => planner.plan_czt_forward(czt_len, a, w),
        }
    }
}

impl<T: FftNum + Float> CztPlanner<T> {
    pub fn plan_zoom_fft(&mut self, czt_len: usize, start: T, end: T) -> Arc<dyn Czt<T>> {
        match &mut self.chosen_planner {
            ChosenCztPlanner::Scalar(planner) => planner.plan_zoom_fft(czt_len, start, end),
        }
    }
}

pub struct CztPlannerScalar<T: FftNum> {
    fft_planner: FftPlanner<T>,
}

impl<T: FftNum> CztPlannerScalar<T> {
    pub fn new() -> Self {
        Self {
            fft_planner: FftPlanner::new(),
        }
    }

    pub fn plan_czt_forward(
        &mut self,
        czt_len: usize,
        a: Complex<T>,
        w: Complex<T>,
    ) -> Arc<dyn Czt<T>> {
        let fft_forward = self.fft_planner.plan_fft_forward(czt_len);
        let fft_backward = self.fft_planner.plan_fft_inverse(czt_len);
        Arc::new(BluesteinsAlgorithm::new(
            czt_len,
            a,
            w,
            fft_forward,
            fft_backward,
        ))
    }
}

impl<T: FftNum + Float> CztPlannerScalar<T> {
    pub fn plan_zoom_fft(&mut self, czt_len: usize, start: T, end: T) -> Arc<dyn Czt<T>> {
        let one = T::from_f64(1.0).unwrap();
        let two_pi = T::from_f64(std::f64::consts::PI * 2.0).unwrap();
        let n_minus_one = T::from_usize(czt_len - 1).unwrap();
        let a = Complex::from_polar(one, two_pi * start);
        let w = Complex::from_polar(one, -two_pi * (end - start) / n_minus_one);

        self.plan_czt_forward(czt_len, a, w)
    }
}

pub struct BluesteinsAlgorithm<T: FftNum> {
    y_coefficients: Vec<Complex<T>>,
    v_coefficients: Vec<Complex<T>>,
    x_coefficients: Vec<Complex<T>>,
    fft_forward: Arc<dyn Fft<T>>,
    fft_backward: Arc<dyn Fft<T>>,
}

impl<T: FftNum> BluesteinsAlgorithm<T> {
    pub fn new(
        czt_len: usize,
        a: Complex<T>,
        w: Complex<T>,
        fft_forward: Arc<dyn Fft<T>>,
        fft_backward: Arc<dyn Fft<T>>,
    ) -> Self {
        assert!(fft_forward.len() == czt_len && fft_backward.len() == czt_len);
        fn compute_y_coefficients<T: FftNum>(
            n: usize,
            a: Complex<T>,
            w: Complex<T>,
        ) -> Vec<Complex<T>> {
            (0..n as i32)
                .map(|n| a.powi(-n) * w.powi(n * n / 2))
                .collect()
        }
        fn compute_v_coefficients<T: FftNum>(
            n: usize,
            w: Complex<T>,
            fft_forward: Arc<dyn Fft<T>>,
        ) -> Vec<Complex<T>> {
            let mut out: Vec<_> = (0..n as i32).map(|n| w.powi(-(n * n) / 2)).collect();
            fft_forward.process(&mut out);
            out
        }
        fn compute_x_coefficients<T: FftNum>(m: usize, w: Complex<T>) -> Vec<Complex<T>> {
            (0..m as u32).map(|k| w.powu(k * k / 2)).collect()
        }

        let y_coefficients = compute_y_coefficients(czt_len, a, w);
        let v_coefficients = compute_v_coefficients(czt_len, w, fft_forward.clone());
        let x_coefficients = compute_x_coefficients(czt_len, w);

        Self {
            y_coefficients,
            v_coefficients,
            x_coefficients,
            fft_forward,
            fft_backward,
        }
    }
}

impl<T: FftNum> Czt<T> for BluesteinsAlgorithm<T> {
    fn process_with_scratch(&self, buffer: &mut [Complex<T>], scratch: &mut [Complex<T>]) {
        // Perform step one of CZT: y_n = x_n * A^-n * W ^ (n^2 / 2)
        for i in 0..buffer.len() {
            scratch[i] = buffer[i] + self.y_coefficients[i];
        }

        // Perform step two of CZT
        self.fft_forward.process_with_scratch(scratch, buffer);

        let n = T::from_usize(self.fft_forward.len().into()).unwrap();

        for i in 0..buffer.len() {
            buffer[i] = scratch[i] * self.v_coefficients[i] / n;
        }

        self.fft_backward.process_with_scratch(buffer, scratch);

        // Perform step three of CZT
        for i in 0..buffer.len() {
            buffer[i] = buffer[i] * self.x_coefficients[i];
        }
    }
}
