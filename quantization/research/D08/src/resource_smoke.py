"""Approved pinned FP resource smoke; no dataset/evaluation quality access."""
import gc
import hashlib
import json
import os
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "15")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
import ctypes
from ctypes import wintypes
import torch
import transformers
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
MODELS = [
    ("EleutherAI/pythia-160m", "50f5173d932e8e61f858120bcb800b97af589f46", "model.safetensors"),
    ("facebook/opt-125m", "27dcfa74d334bc871f3234de431e71c6eeba5dd6", "pytorch_model.bin"),
]
START = time.monotonic()


def check_time():
    if time.monotonic() - START > 25 * 60:
        raise TimeoutError("Approved global 25-minute budget exceeded")


def resources():
    raw = subprocess.check_output([
        "nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"
    ], text=True, timeout=15)
    class MemoryStatus(ctypes.Structure):
        _fields_ = [("length", wintypes.DWORD), ("load", wintypes.DWORD)] + [
            (name, ctypes.c_ulonglong) for name in (
                "total_phys", "avail_phys", "total_page", "avail_page",
                "total_virtual", "avail_virtual", "avail_extended",
            )
        ]
    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError("GlobalMemoryStatusEx failed")
    class ProcessMemory(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("page_faults", wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in (
                "peak_working_set", "working_set", "peak_paged", "paged",
                "peak_nonpaged", "nonpaged", "pagefile", "peak_pagefile",
            )
        ]
    counters = ProcessMemory()
    counters.cb = ctypes.sizeof(counters)
    ctypes.windll.kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    handle = ctypes.windll.kernel32.GetCurrentProcess()
    ctypes.windll.psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
    if not ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
        raise OSError("GetProcessMemoryInfo failed")
    return {"free_gpu_MiB": float(raw.strip().splitlines()[0]),
            "free_ram_GiB": status.avail_phys / 2**30,
            "process_rss_MiB": counters.working_set / 2**20}


def gate():
    check_time()
    r = resources()
    if r["free_gpu_MiB"] < 2048 or r["free_ram_GiB"] < 2.5:
        raise RuntimeError(f"Resource gate failed: {r}")
    return r


def hash_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(chunk)
            check_time()
    return h.hexdigest()


def run_one(repo, revision, filename, row):
    row["resources_before"] = gate()
    tick = time.perf_counter()
    folder = Path(snapshot_download(
        repo_id=repo, revision=revision,
        allow_patterns=[filename, "config.json", "tokenizer*", "special_tokens_map.json", "merges.txt", "vocab.json"],
    ))
    row["download_or_cache_seconds"] = time.perf_counter() - tick
    check_time()
    row["checkpoint_bytes"] = (folder / filename).stat().st_size
    row["checkpoint_sha256"] = hash_file(folder / filename)
    tick = time.perf_counter()
    config = AutoConfig.from_pretrained(folder, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float32,
                                            attn_implementation="eager")
    if filename.endswith(".safetensors"):
        state = load_file(str(folder / filename), device="cpu")
    else:
        state = torch.load(folder / filename, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not all(isinstance(k, str) and isinstance(v, torch.Tensor)
                                              for k, v in state.items()):
        raise TypeError("Checkpoint must be a tensor-only state dictionary")
    mismatch = model.load_state_dict(state, strict=False)
    row["state_dict_missing"] = list(mismatch.missing_keys)
    row["state_dict_unexpected"] = list(mismatch.unexpected_keys)
    allowed = (".attention.bias", ".attention.masked_bias", ".attention.rotary_emb.inv_freq")
    if mismatch.missing_keys or any(not k.endswith(allowed) for k in mismatch.unexpected_keys):
        raise RuntimeError(f"Unsupported checkpoint mismatch: {mismatch}")
    del state
    gc.collect()
    model.tie_weights()
    model.eval()
    model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True, trust_remote_code=False)
    row["cpu_load_seconds"] = time.perf_counter() - tick
    row["unique_parameters"] = sum(p.numel() for p in model.parameters())
    row["resources_after_cpu_load"] = gate()
    tick = time.perf_counter()
    model = model.to("cuda")
    torch.cuda.synchronize()
    row["cuda_transfer_seconds"] = time.perf_counter() - tick
    text = "This is a fixed resource timing example about numbers and simple sentences. " * 64
    ids = tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"]
    assert ids.shape[1] >= 256
    row["tokenizer_class"] = type(tokenizer).__name__
    row["timings"] = []
    with torch.inference_mode():
        for length in (128, 256):
            check_time()
            inp = ids[:, :length].to("cuda")
            mask = torch.ones_like(inp)
            output = model(input_ids=inp, attention_mask=mask, use_cache=False)
            assert torch.isfinite(output.logits).all()
            del output
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            durations = []
            for _ in range(3):
                check_time()
                torch.cuda.synchronize()
                tick = time.perf_counter()
                output = model(input_ids=inp, attention_mask=mask, use_cache=False)
                torch.cuda.synchronize()
                durations.append(time.perf_counter() - tick)
                assert tuple(output.logits.shape[:2]) == (1, length)
                assert torch.isfinite(output.logits).all()
                del output
            row["timings"].append({
                "sequence_length": length, "batch_size": 1, "seconds": durations,
                "peak_allocated_MiB": torch.cuda.max_memory_allocated() / 2**20,
                "peak_reserved_MiB": torch.cuda.max_memory_reserved() / 2**20,
                "resources": resources(),
            })
            del inp, mask
    row["status"] = "FP_SMOKE_PASS_FULL_CYCLE_PENDING"
    del model, tokenizer


def main():
    torch.manual_seed(8)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = ROOT / "results" / f"resource_smoke_{run_id}.json"
    report = {"run_id": run_id, "protocol": "resource_protocol.md", "torch": torch.__version__,
              "transformers": transformers.__version__, "cpu_threads": 4,
              "device": "cuda", "dtype": "float32", "corpus_access": False,
              "test_access": False, "development_access": False, "models": []}
    path.parent.mkdir(exist_ok=True)
    for repo, revision, filename in MODELS:
        row = {"repo": repo, "revision": revision, "checkpoint_file": filename}
        report["models"].append(row)
        try:
            run_one(repo, revision, filename, row)
        except Exception as exc:
            row["status"] = "FAILED"
            row["error"] = repr(exc)
            row["traceback"] = traceback.format_exc()
        finally:
            gc.collect()
            torch.cuda.empty_cache()
            report["wall_seconds"] = time.monotonic() - START
            path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(repo, row["status"], flush=True)
            if "error" in row:
                print(row["error"], flush=True)
        if time.monotonic() - START >= 25 * 60:
            break
    print("Saved", path, flush=True)


if __name__ == "__main__":
    main()
