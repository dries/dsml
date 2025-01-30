use crate::gguf;
use half::f16;
use ndarray::{ArrayD, IxDyn};
use std::io::{Read, Result, Seek};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub fn multiply_u8_by_scalar_avx2(input: &[u8; 32], scalar: f32, output: &mut [f32; 32]) {
    unsafe {
        // Load 32 bytes (u8) into two 128-bit registers (processing 16 at a time)
        let u8_low = _mm_loadu_si128(input.as_ptr() as *const __m128i);
        let u8_high = _mm_loadu_si128(input.as_ptr().add(16) as *const __m128i);

        // Convert u8 -> u16
        let u16_low = _mm256_cvtepu8_epi16(u8_low);
        let u16_high = _mm256_cvtepu8_epi16(u8_high);

        // Convert u16 -> u32
        let u32_low = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(u16_low));
        let u32_high = _mm256_cvtepu16_epi32(_mm256_extracti128_si256::<1>(u16_low));
        let u32_low2 = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(u16_high));
        let u32_high2 = _mm256_cvtepu16_epi32(_mm256_extracti128_si256::<1>(u16_high));

        // Convert u32 -> f32
        let f32_low = _mm256_cvtepi32_ps(u32_low);
        let f32_high = _mm256_cvtepi32_ps(u32_high);
        let f32_low2 = _mm256_cvtepi32_ps(u32_low2);
        let f32_high2 = _mm256_cvtepi32_ps(u32_high2);

        // Multiply by scalar
        let scalar_vec = _mm256_set1_ps(scalar);
        let result_low = _mm256_mul_ps(f32_low, scalar_vec);
        let result_high = _mm256_mul_ps(f32_high, scalar_vec);
        let result_low2 = _mm256_mul_ps(f32_low2, scalar_vec);
        let result_high2 = _mm256_mul_ps(f32_high2, scalar_vec);

        // Store results
        _mm256_storeu_ps(output.as_mut_ptr(), result_low);
        _mm256_storeu_ps(output.as_mut_ptr().add(8), result_high);
        _mm256_storeu_ps(output.as_mut_ptr().add(16), result_low2);
        _mm256_storeu_ps(output.as_mut_ptr().add(24), result_high2);
    }
}


pub fn load_tensor<T: Seek + Read>(stream: &mut T, info: &gguf::TensorInfo) -> Result<ArrayD<f32>> {
    let dimensions: Vec<usize> = info.dimensions.iter().map(|&x| x as usize).collect();
    let num_blocks = dimensions.iter().product::<usize>() / 32;

    let mut data_block: [u8; 34] = [0; 34];
    //let mut data_block_decoded: [f32; 32] = [0.0; 32];
    let mut data: Vec<f32> = Vec::new();
    data.reserve_exact(dimensions.iter().product::<usize>());

    for _block in 0..num_blocks {
        stream.read_exact(&mut data_block)?;
        let value = f16::from_le_bytes(data_block[0..2].try_into().unwrap());

        data_block
            .iter()
            .skip(2)
            .for_each(|x| data.push(f32::from(*x) * f32::from(value)));

        //multiply_u8_by_scalar_avx2(&data_block[2..].try_into().unwrap(), f32::from(value), &mut data_block_decoded);
        //data.extend_from_slice(&data_block_decoded);
    }

    let shape = IxDyn(dimensions.as_slice());
    Ok(ArrayD::<f32>::from_shape_vec(shape, data).unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_avx() {
        let input: [u8; 32] = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
            17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
        ];
        let scalar = 2.5f32;
        let mut output = [0.0; 32];
        multiply_u8_by_scalar_avx2(&input, scalar, &mut output);
        let expected_output: [f32; 32] = [
            2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0, 22.5, 25.0, 27.5, 30.0, 32.5,
            35.0, 37.5, 40.0, 42.5, 45.0, 47.5, 50.0, 52.5, 55.0, 57.5, 60.0, 62.5, 65.0, 67.5, 70.0, 72.5, 75.0, 77.5, 80.0
        ];
        assert_eq!(output, expected_output);
    }
}

