import unittest
import torch

def logit_squared_error(fp_logits, quant_logits):

    """Compute the squared error between the logits of the FP and quantized models."""

    if fp_logits.ndim != 3 or quant_logits.ndim != 3:
        raise ValueError("Expected [batch, sequence, vocabulary] tensors")
    if fp_logits.shape != quant_logits.shape:
        raise ValueError("Logit shapes must match")
    if fp_logits.device != quant_logits.device:
        raise ValueError("Logit devices must match")
    fp_logits_replaced = fp_logits[:, :-1, :].float()
    quant_logits_replaced = quant_logits[:, :-1, :].float()

    squared_error_sum = torch.sum((fp_logits_replaced - quant_logits_replaced) ** 2)
    element_count = fp_logits_replaced.numel()

    return squared_error_sum.item(), element_count


class TestLogitSquaredError(unittest.TestCase):
    def test_identical(self):
        x = torch.zeros(1, 3, 2)
        self.assertEqual(logit_squared_error(x, x), (0.0, 4))

    def test_difference_and_no_mutation(self):
        fp = torch.zeros(1, 3, 2, dtype=torch.float16)
        quant = torch.full_like(fp, 2)
        originals = fp.clone(), quant.clone()
        self.assertEqual(logit_squared_error(fp, quant), (16.0, 4))
        self.assertTrue(torch.equal(fp, originals[0]))
        self.assertTrue(torch.equal(quant, originals[1]))

    def test_last_position_ignored(self):
        fp = torch.zeros(1, 3, 2)
        quant = fp.clone()
        quant[:, -1, :] = 100
        self.assertEqual(logit_squared_error(fp, quant), (0.0, 4))

    def test_cast_before_subtraction(self):
        fp = torch.full((1, 2, 1), 60000, dtype=torch.float16)
        self.assertEqual(logit_squared_error(fp, -fp), (14400000000.0, 1))

    def test_invalid_shapes(self):
        for fp, quant in [(torch.zeros(2, 3), torch.zeros(2, 3)),
                          (torch.zeros(1, 3, 2), torch.zeros(1, 3, 1))]:
            with self.assertRaises(ValueError):
                logit_squared_error(fp, quant)


if __name__ == "__main__":
    unittest.main()