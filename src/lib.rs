use rustfft::{FftNum, FftPlanner, num_complex::Complex, num_traits::FromPrimitive};

/// Hilbert_transform is a library written in Rust to perform the hilbert transformation like
/// Matlab/Octave or scipy.signals.hilbert.
///
/// Hilbert_transform is implemented based on scipy implementation of same function.
///
/// # Usage
///
/// ```
/// use hilbert_transform::{hilbert};
///
/// let input = vec![1.0, 2.0, 3.0, 4.0];     
/// let hilbert_output = hilbert(&input);
/// ```
///
/// hilbert_output will be equal to: [Complex { re: 1.0, im: 1.0 }, Complex { re: 2.0, im: -1.0 }, Complex { re: 3.0, im: -1.0 }, Complex { re: 4.0, im: 1.0 }]
///

pub fn hilbert<T: FftNum + FromPrimitive>(input: &[T]) -> Vec<Complex<T>> {

    let len = input.len();
    let mut planner = FftPlanner::<T>::new();
    let fft = planner.plan_fft_forward(len);

    let mut fft_complex = input
        .iter()
        .map(|&val| Complex::new(val, T::zero()))
        .collect::<Vec<_>>();

    fft.process(&mut fft_complex);

    let mut h_spectrum = vec![Complex::new(T::zero(), T::zero()); len];

    let two = T::from_usize(2).unwrap();

    if len % 2 == 0 {
        h_spectrum[0] = Complex::new(T::one(), T::zero());
        h_spectrum[len / 2] = Complex::new(T::one(), T::zero());
        for i in 1..(len / 2) {
            h_spectrum[i] = Complex::new(two, T::zero());
        }
    } else {
        h_spectrum[0] = Complex::new(T::one(), T::zero());
        for i in 1..((len + 1) / 2) {
        h_spectrum[i] = Complex::new(two, T::zero());
        }
    }

    for i in 0..len {
        fft_complex[i] = fft_complex[i] * h_spectrum[i];
    }

    let mut ifft_complex = fft_complex.clone();
    let ifft = planner.plan_fft_inverse(len);
    ifft.process(&mut ifft_complex);

    let len = T::from_usize(len).expect("failed to convert len to T");
    for val in ifft_complex.iter_mut() {
        *val = *val / len;
    }

    ifft_complex
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hilbert_f64() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let expected_output = vec![
            Complex::new(1.0, 1.0), 
            Complex::new(2.0, -1.0), 
            Complex::new(3.0, -1.0), 
            Complex::new(4.0, 1.0)
        ];
        assert_eq!(hilbert(&input), expected_output);
    }


    #[test]
    fn test_hilbert_f32() {
        let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let expected_output = vec![
            Complex::new(1.0, 1.0), 
            Complex::new(2.0, -1.0), 
            Complex::new(3.0, -1.0), 
            Complex::new(4.0, 1.0)
        ];
        assert_eq!(hilbert(&input), expected_output);
    }
}
