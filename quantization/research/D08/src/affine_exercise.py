"""CPU affine fake quantization exercise, checked against installed PyTorch.
No model/data/search. Forward only: no claim of QAT gradient equivalence.
"""
import torch


def affine_fake_quant(x, scale, zero_point, quant_min, quant_max):
    """Return (codes, restored_values); codes may remain floating tensors."""
    if scale <= 0 or quant_min >= quant_max:
        raise ValueError("Invalid scale or code range")
    if not isinstance(zero_point, int) or not quant_min <= zero_point <= quant_max:
        raise ValueError("zero_point must be a representable integer")
    
    x_rounded = (x/scale).round()
    x_after_zero = x_rounded + zero_point
    x_clamped = x_after_zero.clamp(quant_min, quant_max)

    x_hat = x_clamped - zero_point
    x_hat = x_hat * scale

    return x_clamped, x_hat


def main():
    cases = (
        (0.5, 0, -7, 7),
        (0.5, 3, 0, 7),
    )
    x = torch.tensor([-8., -3.5, -1.25, -0.25, 0., 0.25, 0.75, 1.25, 3.5, 8.])
    for scale, z, lo, hi in cases:
        original = x.clone()
        codes, restored = affine_fake_quant(x, scale, z, lo, hi)
        expected_codes = (torch.round(x / scale) + z).clamp(lo, hi)
        reference = torch.fake_quantize_per_tensor_affine(x, scale, z, lo, hi)
        torch.testing.assert_close(x, original, rtol=0, atol=0)
        torch.testing.assert_close(codes, expected_codes, rtol=0, atol=0)
        torch.testing.assert_close(restored, reference, rtol=0, atol=0)
        assert restored[x == 0].item() == 0
        print(f"PASS scale={scale}, zero_point={z}, codes=[{lo}, {hi}]")
    print("Forward matches torch.fake_quantize_per_tensor_affine on these controls")


if __name__ == "__main__":
    main()
