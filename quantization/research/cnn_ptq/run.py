"""LeNet/MNIST static PTQ with a checkpoint shared with pruning experiments."""

import argparse
import copy
import gzip
import hashlib
import json
import os
import platform
import statistics
import struct
import time
import urllib.request
from pathlib import Path

import torch
from torch import nn
from torch.ao.quantization import QConfig, QConfigMapping
from torch.ao.quantization.observer import HistogramObserver, MinMaxObserver, PerChannelMinMaxObserver
from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx
from torch.nn import functional as F
from torch.utils.benchmark import Timer
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parent
SEED = 42
ENGINE = "onednn"
CALIBRATION_SIZE = 512
MNIST_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.bn1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), 2)
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def read_idx(filename, expected_md5, dimensions):
    path = ROOT / "data" / filename
    path.parent.mkdir(exist_ok=True)
    if not path.exists():
        with urllib.request.urlopen(MNIST_URL + filename, timeout=60) as response:
            payload = response.read()
        if hashlib.md5(payload).hexdigest() != expected_md5:
            raise ValueError(f"MNIST checksum mismatch: {filename}")
        path.write_bytes(payload)
    payload = path.read_bytes()
    if hashlib.md5(payload).hexdigest() != expected_md5:
        raise ValueError(f"MNIST checksum mismatch: {filename}")
    content = bytearray(gzip.decompress(payload))
    magic, *shape = struct.unpack_from(f">{dimensions + 1}I", content)
    if magic != 0x0800 + dimensions:
        raise ValueError(f"Invalid MNIST IDX header: {filename}")
    return torch.frombuffer(content, dtype=torch.uint8, offset=4 * (dimensions + 1)).reshape(shape)


def load_mnist(train):
    if train:
        prefix = "train"
        image_hash = "f68b3c2dcbeaaa9fbdd348bbdeb94873"
        label_hash = "d53e105ee54ea40749a09fcbcd1e9432"
    else:
        prefix = "t10k"
        image_hash = "9fb629c4189551a2d022fa330f9573f3"
        label_hash = "ec29112dd5afa0611ce80d1b7f02629c"
    images = read_idx(f"{prefix}-images-idx3-ubyte.gz", image_hash, 3)
    labels = read_idx(f"{prefix}-labels-idx1-ubyte.gz", label_hash, 1)
    return images[:, None].float().div_(255).sub_(0.1307).div_(0.3081), labels.long()


def train_baseline(images, labels, device):
    if device == "cuda":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    model = LeNet().to(device)
    loader = DataLoader(TensorDataset(images, labels), batch_size=64, shuffle=True,
                        generator=torch.Generator().manual_seed(SEED))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(3):
        for x, y in loader:
            optimizer.zero_grad(set_to_none=True)
            F.cross_entropy(model(x.to(device)), y.to(device)).backward()
            optimizer.step()
        print(f"Training epoch {epoch + 1}/3 complete", flush=True)
    return model.cpu().eval()


def make_quantized(model, calibration, per_channel, activation_observer):
    activation = activation_observer.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_affine)
    if per_channel:
        weight = PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric)
    else:
        weight = MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric)
    config = QConfigMapping().set_global(QConfig(activation=activation, weight=weight))
    prepared = prepare_fx(copy.deepcopy(model), config, (calibration[:1],))
    for batch in calibration.split(64):
        prepared(batch)
    quantized = convert_fx(prepared)
    layers = [m for m in quantized.modules() if isinstance(m, (torch.ao.nn.quantized.Conv2d, torch.ao.nn.quantized.Linear))]
    if len(layers) != 5 or not all(layer.weight().dtype == torch.qint8 for layer in layers):
        raise RuntimeError("Expected two quantized convolutions and three quantized linear layers")
    expected_scheme = torch.per_channel_affine if per_channel else torch.per_tensor_affine
    if not all(layer.weight().qscheme() == expected_scheme for layer in layers):
        raise RuntimeError("Converted layers do not match the requested weight granularity")
    return quantized


def benchmark_ms(model, images):
    timer = Timer("model(images)", globals={"model": model, "images": images}, num_threads=1)
    medians = [timer.blocked_autorange(min_run_time=0.25).median * 1000 for _ in range(3)]
    return {"median": statistics.median(medians), "rounds": medians}


@torch.no_grad()
def evaluate(model, images, labels, state_path, fp_correct):
    predictions = torch.cat([model(x).argmax(1) for x in images.split(256)])
    correct = predictions.eq(labels)
    torch.save(model.state_dict(), state_path)
    return correct, {
        "correct": int(correct.sum()),
        "accuracy_pct": 100 * int(correct.sum()) / len(labels),
        "fp_correct_ptq_wrong": int((fp_correct & ~correct).sum()) if fp_correct is not None else 0,
        "fp_wrong_ptq_correct": int((~fp_correct & correct).sum()) if fp_correct is not None else 0,
        "state_dict_bytes": state_path.stat().st_size,
        "cpu_ms_batch_1": benchmark_ms(model, images[:1]),
        "cpu_ms_batch_256": benchmark_ms(model, images[:256]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, help="Load an existing FP32 LeNet-BN state_dict")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu", help="Training device")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts")
    args = parser.parse_args()
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    torch.backends.quantized.engine = ENGINE
    train_x, train_y = load_mnist(train=True)
    test_x, test_y = load_mnist(train=False)
    indices = torch.randperm(len(train_x), generator=torch.Generator().manual_seed(SEED))[:CALIBRATION_SIZE]
    if args.checkpoint:
        model = LeNet().eval()
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    else:
        model = train_baseline(train_x, train_y, args.device)

    rows = {}
    fp_correct = None
    baseline_path = args.output / "fp32" / "model.pt"
    for name, recipe in (
        ("fp32", None),
        ("int8_tensor_minmax", (False, MinMaxObserver)),
        ("int8_channel_minmax", (True, MinMaxObserver)),
        ("int8_channel_histogram", (True, HistogramObserver)),
    ):
        path = args.output / name / "model.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        if recipe is None:
            fp_correct, rows[name] = evaluate(model, test_x, test_y, path, None)
        else:
            with torch.no_grad():
                started = time.perf_counter()
                quantized = make_quantized(model, train_x[indices], *recipe)
                elapsed = time.perf_counter() - started
                _, rows[name] = evaluate(quantized, test_x, test_y, path, fp_correct)
                rows[name]["prepare_calibrate_convert_seconds"] = elapsed
                # The artifact is a state_dict to be loaded into the same converted graph.
                restored = copy.deepcopy(quantized)
                restored.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
                torch.testing.assert_close(restored(test_x[:64]), quantized(test_x[:64]), rtol=0, atol=0)
        print(name, rows[name], flush=True)

    result = {
        "torch": torch.__version__,
        "platform": platform.platform(),
        "cpu": os.environ.get("PROCESSOR_IDENTIFIER", platform.processor()),
        "test_examples": len(test_y),
        "training_device": "external checkpoint" if args.checkpoint else args.device,
        "training_seed": SEED if args.checkpoint is None else None,
        "calibration_seed": SEED,
        "calibration_indices": indices.tolist(),
        "quantized_engine": ENGINE,
        "cpu_threads": torch.get_num_threads(),
        "fp32_state_dict_sha256": hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
        "results": rows,
    }
    (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
