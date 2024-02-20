use rand::{
    distributions::{uniform::SampleUniform, Distribution, Uniform},
    rngs::StdRng,
    SeedableRng,
};
use rustczt::{naive_czt::NaiveCzt, Czt, CztPlanner};
use rustfft::{
    num_complex::{Complex, ComplexFloat},
    num_traits::Float,
    FftNum, FftPlanner,
};
use std::fmt::Display;

const RNG_SEED: [u8; 32] = [
    1, 9, 1, 0, 1, 1, 4, 3, 1, 4, 9, 8, 4, 1, 4, 8, 2, 8, 1, 2, 2, 2, 6, 1, 2, 3, 4, 5, 6, 7, 8, 9,
];

/// Chirp Z transform
///
/// Implementation is O(n^2) and is simply looping over the expression.
fn czt<T>(buffer: &[Complex<T>], a: &Complex<T>, w: &Complex<T>) -> Vec<Complex<T>>
where
    T: FftNum,
{
    let czt_obj = NaiveCzt::new(buffer.len(), *a, *w);
    let mut out = Vec::from(buffer);
    czt_obj.process(&mut out);
    out
}

fn compare_float_vector<T>(expected: &[T], actual: &[T])
where
    T: ComplexFloat + Display + std::fmt::Debug,
    T::Real: From<f64>,
{
    let threshold = T::Real::try_from(0.00001).unwrap();

    for (i, (&ex, &ac)) in expected.iter().zip(actual.iter()).enumerate() {
        assert!(
            Float::abs(ac.re() - ex.re()) < threshold || Float::abs(ac.im() - ex.im()) < threshold,
            "Element {i} is not equal: {ex} != {ac}"
        );
    }
}

fn random_signal<T: FftNum + SampleUniform>(length: usize) -> Vec<Complex<T>> {
    let mut sig = Vec::with_capacity(length);
    let dist: Uniform<T> = Uniform::new(T::zero(), T::from_f64(10.0).unwrap());
    let mut rng: StdRng = SeedableRng::from_seed(RNG_SEED);
    for _ in 0..length {
        sig.push(Complex {
            re: (dist.sample(&mut rng)),
            im: (dist.sample(&mut rng)),
        });
    }

    sig
}

#[test]
fn test_unit_circle_contour_czt_accuracy() {
    let signal = random_signal(64);
    let mut planner = CztPlanner::new();
    let a = Complex::from_polar(1.0, 5.0);
    let w = Complex::from_polar(1.0, -1.0 * std::f64::consts::PI / signal.len() as f64);
    let czt_obj = planner.plan_czt_forward(signal.len(), a, w);

    let mut actual = signal.clone();
    czt_obj.process(&mut actual);
    let expected = czt(&signal, &a, &w);
    compare_float_vector(&expected, &actual);
}

#[test]
fn test_fft_like_czt_accuracy() {
    let signal = random_signal(64);
    let mut planner = CztPlanner::new();
    let a = Complex::from_polar(1.0, 0.0);
    let w = Complex::from_polar(1.0, -2.0 * std::f64::consts::PI / signal.len() as f64);
    let czt_obj = planner.plan_czt_forward(signal.len(), a, w);

    let mut actual = signal.clone();
    czt_obj.process(&mut actual);
    let expected = czt(&signal, &a, &w);
    compare_float_vector(&expected, &actual);
}

#[test]
fn test_naive_czt_accuracy() {
    let signal = random_signal(128);
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(signal.len());

    let mut actual = signal.clone();
    fft.process(&mut actual);

    let a = Complex::new(1.0, 0.0);
    let w = Complex::from_polar(1.0, -2.0 * std::f64::consts::PI / signal.len() as f64);
    let expected = czt(&signal, &a, &w);
    compare_float_vector(&expected, &actual);
}
