"""Standard FakeQuantize: observer updates buffers, optimizer does not."""
import torch
from torch.ao.quantization import FakeQuantize, MinMaxObserver


def main():
    fq = FakeQuantize(observer=MinMaxObserver, quant_min=-7, quant_max=7,
                     dtype=torch.qint8, qscheme=torch.per_tensor_symmetric)
    assert list(fq.named_parameters()) == []
    print("FakeQuantize named parameters:", list(fq.named_parameters()))
    a = torch.tensor([-1.0, 0.3, 1.0])
    fq(a)
    first_scale = fq.scale.clone()
    print("Observed range [-1,1], scale:", first_scale.item())
    fq.disable_observer()
    b = torch.tensor([-2.0, 0.3, 2.0])
    frozen_output = fq(b)
    torch.testing.assert_close(fq.scale, first_scale, rtol=0, atol=0)
    assert frozen_output[0] > b[0] and frozen_output[-1] < b[-1]
    print("Observer OFF, fake quant ON:", frozen_output.tolist())
    fq.enable_observer()
    fq.disable_fake_quant()
    unquantized_output = fq(b)
    torch.testing.assert_close(unquantized_output, b, rtol=0, atol=0)
    assert fq.scale.item() > first_scale.item()
    print("Observer ON, fake quant OFF:", unquantized_output.tolist())
    print("New scale:", fq.scale.item())
    print("PASS: observer and fake quant switches are independent")


if __name__ == "__main__":
    main()
