"""Forward equality does not imply backward equality. CPU, fixed qparams."""
import torch
from affine_exercise import affine_fake_quant


def main():
    values = torch.tensor([-4.0, -0.3, 0.3, 3.6, 4.0], dtype=torch.float32)
    manual_x = values.clone().requires_grad_()
    pytorch_x = values.clone().requires_grad_()
    _, manual_y = affine_fake_quant(manual_x, 0.5, 0, -7, 7)
    pytorch_y = torch.fake_quantize_per_tensor_affine(pytorch_x, 0.5, 0, -7, 7)
    torch.testing.assert_close(manual_y, pytorch_y, rtol=0, atol=0)
    manual_y.sum().backward()
    pytorch_y.sum().backward()
    torch.testing.assert_close(manual_x.grad, torch.zeros_like(values))
    torch.testing.assert_close(pytorch_x.grad, torch.tensor([0., 1., 1., 1., 0.]))
    print("torch", torch.__version__)
    print("x       output   ordinary-round grad   PyTorch fake-quant grad")
    for x, y, gm, gp in zip(values, pytorch_y.detach(), manual_x.grad, pytorch_x.grad):
        print(f"{x.item():5.1f}   {y.item():6.1f}        {gm.item():.0f}                       {gp.item():.0f}")
    print("PASS: equal forwards; distinct backwards; rounded-code boundary checked")


if __name__ == "__main__":
    main()
