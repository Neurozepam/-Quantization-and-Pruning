"""One CPU W8A8 QAT step and conversion; not an LLM comparator.
Recipe: PyTorch eager torch.ao.quantization, x86, observer-based qparams.
Synthetic data only. No accuracy, storage or performance claim.
"""
import torch
from torch import nn
from torch.ao.quantization import (
    DeQuantStub, FakeQuantize, MovingAverageMinMaxObserver, QConfig,
    QuantStub, convert, disable_observer, prepare_qat,
)


class TinyLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.quant = QuantStub()
        self.linear = nn.Linear(3, 2)
        self.dequant = DeQuantStub()

    def forward(self, x):
        return self.dequant(self.linear(self.quant(x)))


def main():
    torch.manual_seed(8)
    torch.set_num_threads(1)
    if "x86" not in torch.backends.quantized.supported_engines:
        raise RuntimeError("This demo requires the x86 quantized CPU backend")
    torch.backends.quantized.engine = "x86"
    model = TinyLinear().train()
    model.qconfig = QConfig(
        activation=FakeQuantize.with_args(
            observer=MovingAverageMinMaxObserver, quant_min=0, quant_max=255,
            dtype=torch.quint8, qscheme=torch.per_tensor_affine,
        ),
        weight=FakeQuantize.with_args(
            observer=MovingAverageMinMaxObserver, quant_min=-128, quant_max=127,
            dtype=torch.qint8, qscheme=torch.per_tensor_symmetric,
        ),
    )
    qat = prepare_qat(model, inplace=False)
    x = torch.randn(8, 3)
    target = torch.randn(8, 2)
    with torch.no_grad():
        qat(x)
    qat.apply(disable_observer)
    before = qat.linear.weight.detach().clone()
    optimizer = torch.optim.SGD(qat.parameters(), lr=0.01)
    optimizer.zero_grad()
    loss = (qat(x) - target).square().mean()
    loss.backward()
    assert qat.linear.weight.grad is not None
    assert qat.linear.weight.grad.abs().sum().item() > 0
    optimizer.step()
    assert not torch.equal(before, qat.linear.weight.detach())
    assert qat.linear.weight.dtype == torch.float32
    assert set(dict(qat.named_parameters())) == {"linear.weight", "linear.bias"}
    print("Trainable parameters:", list(dict(qat.named_parameters())))
    print("After QAT step: master weight dtype", qat.linear.weight.dtype)
    print("Max weight update:", (before - qat.linear.weight.detach()).abs().max().item())
    qat.eval()
    with torch.no_grad():
        qat_output = qat(x)
        quantized = convert(qat, inplace=False)
        converted_output = quantized(x)
    assert quantized.linear.weight().is_quantized
    assert converted_output.shape == qat_output.shape
    assert torch.isfinite(converted_output).all()
    print("Converted Linear type:", type(quantized.linear).__name__)
    print("Converted weight dtype:", quantized.linear.weight().dtype)
    print("DeQuantStub output dtype:", converted_output.dtype)
    print("Max QAT/converted output difference:",
          (qat_output - converted_output).abs().max().item())
    print("PASS: FP master weights trained; quantized CPU Linear executed")


if __name__ == "__main__":
    main()
