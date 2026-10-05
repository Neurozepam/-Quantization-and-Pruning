import numpy as np

from quantization.learning.D01.day1_quantization import (
    quantize_tensor, dequantize_tensor,
)
from quantization.learning.D02.day2_quantization import (
    quantize_groupwise, dequantize_groupwise,
)

rng = np.random.default_rng(42)
w = rng.normal(size=(64, 256))
w *= np.geomspace(0.1, 10.0, 64)[:, None]
bits = 4

flat = w.ravel()
q_flat, scale_flat, zero_point_flat = quantize_tensor(flat, bits, symmetric=True)

dequantized_flat = dequantize_tensor(q_flat, scale_flat, zero_point_flat)

n_scales = 1
code_bits = w.size * bits
scale_bits = n_scales * 16

MSE_flat = np.mean((flat - dequantized_flat)**2)

print("Расчётный размер: packed INT4 + FP16 scales")

print(f"Flat quantization MSE: {MSE_flat:.6f} | n_scales: {n_scales} \
      | overhead_pct: {100 * scale_bits / code_bits:.2f}% \
        | bits per weight: {(code_bits + scale_bits) / w.size:.2f}\n")

for group_size in [32, 64, 128, 256]:
    q_groups, scales = quantize_groupwise(w, bits, group_size)

    n_scales = scales.size
    code_bits = w.size * bits
    scale_bits = n_scales * 16
    
    dequantized_w = dequantize_groupwise(q_groups, scales)

    MSE_groupwise = np.mean((w - dequantized_w)**2)

    print(f"Group size: {group_size}, MSE: {MSE_groupwise:.6f} | n_scales: {n_scales} \
          | overhead_pct: {100 * scale_bits / code_bits:.2f}% \
            | bits per weight: {(code_bits + scale_bits) / w.size:.2f}\n")