use std::sync::Arc;

use rustfft::{num_complex::Complex, num_traits::Float, FftNum, FftPlanner};

use crate::{bluesteins::BluesteinsAlgorithm, Czt};

pub enum ChosenCztPlanner<T: Float + FftNum> {
    Scalar(CztPlannerScalar<T>),
}

pub struct CztPlanner<T: Float + FftNum> {
    chosen_planner: ChosenCztPlanner<T>,
}

impl<T: Float + FftNum> CztPlanner<T> {
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

impl<T: Float + FftNum> CztPlanner<T> {
    pub fn plan_zoom_fft(&mut self, czt_len: usize, start: T, end: T) -> Arc<dyn Czt<T>> {
        match &mut self.chosen_planner {
            ChosenCztPlanner::Scalar(planner) => planner.plan_zoom_fft(czt_len, start, end),
        }
    }
}

pub struct CztPlannerScalar<T: Float + FftNum> {
    fft_planner: FftPlanner<T>,
}

impl<T: Float + FftNum> CztPlannerScalar<T> {
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
        let fft_forward = self.fft_planner.plan_fft_forward(czt_len * 2);
        Arc::new(BluesteinsAlgorithm::new(czt_len, a, w, fft_forward))
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
