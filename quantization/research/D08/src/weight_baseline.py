"""Restricted-site integration of unchanged official GPTQ core; RTN controls."""
import sys
import time
from pathlib import Path
import torch
from torch import nn

VENDOR = Path(__file__).resolve().parent / "vendor_gptq"
sys.path.insert(0, str(VENDOR))
from gptq import GPTQ
from quant import Quantizer, quantize

GROUP = 128


def qparams(group):
    lo = torch.minimum(group.amin(dim=1, keepdim=True), torch.zeros_like(group[:, :1]))
    hi = torch.maximum(group.amax(dim=1, keepdim=True), torch.zeros_like(group[:, :1]))
    zero_range = (lo == 0) & (hi == 0)
    lo = torch.where(zero_range, -torch.ones_like(lo), lo)
    hi = torch.where(zero_range, torch.ones_like(hi), hi)
    scale = (hi - lo) / 15
    zero = torch.round(-lo / scale)
    return scale, zero


def rtn(w):
    result = torch.empty_like(w)
    params = []
    for start in range(0, w.shape[1], GROUP):
        stop = min(start + GROUP, w.shape[1])
        group = w[:, start:stop]
        scale, zero = qparams(group)
        codes = (torch.round(group / scale) + zero).clamp(0, 15)
        result[:, start:stop] = (codes - zero) * scale
        params.append((scale.clone(), zero.clone()))
    return result, params


class RecordingQuantizer(Quantizer):
    def __init__(self):
        super().__init__()
        self.history = []

    def find_params(self, x, weight=False):
        super().find_params(x, weight=weight)
        self.history.append((self.scale.clone(), self.zero.clone()))


@torch.no_grad()
def gptq(w, x):
    layer = nn.Linear(w.shape[1], w.shape[0], bias=False, device=w.device, dtype=w.dtype)
    layer.weight.copy_(w)
    solver = GPTQ(layer)
    solver.quantizer = RecordingQuantizer()
    solver.quantizer.configure(4, perchannel=True, sym=False, mse=False)
    torch.cuda.synchronize()
    tick = time.perf_counter()
    for chunk in x.split(128, dim=0):
        solver.add_batch(chunk.unsqueeze(0), None)
    torch.cuda.synchronize()
    h_seconds = time.perf_counter() - tick
    tick = time.perf_counter()
    solver.fasterquant(blocksize=128, percdamp=0.01, groupsize=128,
                      actorder=False, static_groups=False)
    torch.cuda.synchronize()
    quant_seconds = time.perf_counter() - tick
    groups = (w.shape[1] + GROUP - 1) // GROUP
    assert len(solver.quantizer.history) == groups + 1
    params = solver.quantizer.history[1:]
    output = layer.weight.detach().clone()
    solver.free()
    del solver, layer
    return output, params, {"h_seconds": h_seconds, "quant_seconds": quant_seconds}


def occupancy(qw, params):
    hist = torch.zeros(16, device=qw.device, dtype=torch.long)
    occupied_sum = 0.0
    row_groups = 0
    for i, (scale, zero) in enumerate(params):
        part = qw[:, i * GROUP : (i + 1) * GROUP]
        codes_float = part / scale + zero
        codes = codes_float.round().long()
        assert codes.min() >= 0 and codes.max() <= 15
        torch.testing.assert_close((codes.to(part.dtype) - zero) * scale, part,
                                   rtol=1e-5, atol=1e-7)
        for k in range(16):
            used = codes == k
            hist[k] += used.sum()
            occupied_sum += used.any(dim=1).sum().item()
        row_groups += part.shape[0]
    assert hist.sum().item() == qw.numel()
    return {"code_histogram": hist.cpu().tolist(),
            "mean_occupied_codes_per_row_group": occupied_sum / row_groups,
            "row_groups": row_groups}


def reconstruction(x, w, qw):
    ew = qw - w
    reference = x @ w.T
    mse = (x @ ew.T).square().mean().item()
    power = reference.square().mean().item()
    diag = (ew.square() * x.square().mean(dim=0)).sum().item() / w.shape[0]
    assert power > 0
    return {"output_mse": mse, "fp_output_power": power,
            "normalized_output_mse": mse / power, "diagonal_only_mse": diag,
            "net_cross_channel_mse": mse - diag}


@torch.no_grad()
def tests():
    torch.manual_seed(8)
    w = torch.randn(5, 130)
    w[0].zero_()
    w[1].fill_(0.25)
    w[2].fill_(-0.25)
    original = w.clone()
    qw, params = rtn(w)
    for i, (s, z) in enumerate(params):
        part = w[:, i * GROUP : (i + 1) * GROUP]
        primary = Quantizer()
        primary.configure(4, perchannel=True, sym=False, mse=False)
        primary.find_params(part, weight=True)
        torch.testing.assert_close(s, primary.scale, rtol=0, atol=0)
        torch.testing.assert_close(z, primary.zero, rtol=0, atol=0)
        torch.testing.assert_close(qw[:, i * GROUP : (i + 1) * GROUP],
                                   primary.quantize(part), rtol=0, atol=0)
    torch.testing.assert_close(w, original, rtol=0, atol=0)
    occupancy(qw, params)
    print("PASS RTN: primary quantizer, tail group, zero/constant rows, nonmutation")
    w = torch.randn(3, 130, device="cuda")
    x = torch.randn(256, 130, device="cuda")
    x[:, -1] = 0
    w_before, x_before = w.clone(), x.clone()
    wrapped, params, _ = gptq(w, x)
    layer = nn.Linear(130, 3, bias=False, device="cuda")
    layer.weight.copy_(w)
    solver = GPTQ(layer)
    solver.quantizer = Quantizer()
    solver.quantizer.configure(4, perchannel=True, sym=False, mse=False)
    for chunk in x.split(128):
        solver.add_batch(chunk.unsqueeze(0), None)
    solver.fasterquant(blocksize=128, percdamp=.01, groupsize=128,
                      actorder=False, static_groups=False)
    torch.testing.assert_close(wrapped, layer.weight, rtol=0, atol=0)
    torch.testing.assert_close(w, w_before, rtol=0, atol=0)
    torch.testing.assert_close(x, x_before, rtol=0, atol=0)
    assert torch.isfinite(wrapped).all()
    occupancy(wrapped, params)
    solver.free()
    print("PASS GPTQ integration: exact primary parity, dead channel, tail group, nonmutation")


if __name__ == "__main__":
    torch.set_num_threads(4)
    tests()
