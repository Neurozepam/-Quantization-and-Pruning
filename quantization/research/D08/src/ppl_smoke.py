"""Frozen 3-model FP/RTN/GPTQ W4 full-forward restricted-validation smoke.
No test or dataset loading. Native token manifests saved before quantization.
"""
import argparse
import contextlib
import gc
import json
import math
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from resource_smoke import resources
from weight_baseline import GPTQ, Quantizer, rtn
from weight_diagnostic import MANIFEST_SHA, SOURCE_HASHES, VENDOR, digest

ROOT = Path(__file__).resolve().parents[1]
D05 = ROOT.parent / "D05"
BUDGET_ID = "20261014T150623521707Z"
MODELS = [
    ("HuggingFaceTB/SmolLM2-135M", "93efa2f097d58c2a74874c7e644dbc9b0cee75a2", "model.safetensors", "model.layers"),
    ("EleutherAI/pythia-160m", "50f5173d932e8e61f858120bcb800b97af589f46", "model.safetensors", "gpt_neox.layers"),
    ("facebook/opt-125m", "27dcfa74d334bc871f3234de431e71c6eeba5dd6", "pytorch_model.bin", "model.decoder.layers"),
]
START = time.monotonic()


def check_time():
    if time.monotonic() - START > 1800:
        raise TimeoutError("Approved bounded PPL stage30-minute cap reached")


def gate():
    check_time()
    r = resources()
    if r["free_gpu_MiB"] < 2048 or r["free_ram_GiB"] < 2.5:
        raise RuntimeError(f"Resource gate failed: {r}")
    return r


def load_model(folder, filename):
    config = AutoConfig.from_pretrained(folder, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float32, attn_implementation="eager")
    state = load_file(str(folder / filename), device="cpu") if filename.endswith(".safetensors") else torch.load(
        folder / filename, map_location="cpu", weights_only=True)
    assert isinstance(state, dict) and all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in state.items())
    mismatches = model.load_state_dict(state, strict=False)
    allowed = (".attention.bias", ".attention.masked_bias", ".attention.rotary_emb.inv_freq")
    missing = set(mismatches.missing_keys)
    if missing == {"lm_head.weight"} and config.tie_word_embeddings:
        input_weight = model.get_input_embeddings().weight
        output_weight = model.get_output_embeddings().weight
        assert input_weight is output_weight
        assert "model.embed_tokens.weight" in state
        torch.testing.assert_close(input_weight, state["model.embed_tokens.weight"].float(), rtol=0, atol=0)
        missing.clear()
    if missing or any(not k.endswith(allowed) for k in mismatches.unexpected_keys):
        raise RuntimeError(f"Unexpected state mismatch: {mismatches}")
    del state
    model.tie_weights()
    gc.collect()
    model.eval()
    model.config.use_cache = False
    model = model.to("cuda")
    return model


@torch.inference_mode()
def evaluate(model, fragments):
    check_time()
    total, count = 0.0, 0
    torch.cuda.synchronize()
    tick = time.perf_counter()
    for tokens in fragments:
        check_time()
        ids = torch.tensor([tokens], dtype=torch.long, device="cuda")
        output = model(input_ids=ids, use_cache=False)
        assert torch.isfinite(output.logits).all()
        logits = output.logits[:, :-1].reshape(-1, output.logits.shape[-1])
        targets = ids[:, 1:].reshape(-1)
        total += F.cross_entropy(logits, targets, reduction="sum").item()
        count += targets.numel()
        del output, logits, targets, ids
    torch.cuda.synchronize()
    mean = total / count
    assert count > 0 and math.isfinite(mean)
    return {"total_nll": total, "target_tokens": count, "mean_nll": mean,
            "ppl": math.exp(mean), "evaluation_seconds": time.perf_counter() - tick}


@torch.inference_mode()
def quantize_model(model, block_path, calibration, method, log, save_progress):
    block_stats = []
    total_start = time.perf_counter()
    for block_index, block in enumerate(model.get_submodule(block_path)):
        check_time()
        linears = {n: m for n, m in block.named_modules() if isinstance(m, torch.nn.Linear)}
        assert linears and all(m.weight.dtype == torch.float32 for m in linears.values())
        row = {"block": block_index, "linears": list(linears)}
        if method == "RTN":
            tick = time.perf_counter()
            for module in linears.values():
                qw, _ = rtn(module.weight.detach())
                module.weight.copy_(qw)
                del qw
            torch.cuda.synchronize()
            row["quant_seconds"] = time.perf_counter() - tick
        else:
            solvers, handles = {}, []
            for name, module in linears.items():
                solver = GPTQ(module)
                solver.quantizer = Quantizer()
                solver.quantizer.configure(4, perchannel=True, sym=False, mse=False)
                solvers[name] = solver
                def capture(module, inp, out, solver=solver):
                    solver.add_batch(inp[0].detach(), None)
                handles.append(module.register_forward_hook(capture))
            torch.cuda.synchronize()
            tick = time.perf_counter()
            try:
                for tokens in calibration:
                    check_time()
                    ids = torch.tensor([tokens], dtype=torch.long, device="cuda")
                    output = model(input_ids=ids, use_cache=False)
                    assert torch.isfinite(output.logits).all()
                    del ids, output
            finally:
                for handle in handles:
                    handle.remove()
            torch.cuda.synchronize()
            row["h_collection_seconds"] = time.perf_counter() - tick
            tick = time.perf_counter()
            for name, solver in solvers.items():
                check_time()
                with contextlib.redirect_stdout(log):
                    solver.fasterquant(blocksize=128, percdamp=.01, groupsize=128,
                                      actorder=False, static_groups=False)
                assert torch.isfinite(linears[name].weight).all()
                solver.free()
            torch.cuda.synchronize()
            row["quant_seconds"] = time.perf_counter() - tick
            del solver, solvers
        assert all(not m._forward_hooks for m in linears.values())
        block_stats.append(row)
        save_progress(block_stats)
        print(method, block_index, "complete", flush=True)
    return {"blocks": block_stats, "quantization_cycle_seconds": time.perf_counter() - total_start,
            "quantized_linear_count": sum(len(r["linears"]) for r in block_stats)}


def nll_test():
    probs = torch.tensor([[.1, .7, .2], [.6, .3, .1]], dtype=torch.float64)
    target = torch.tensor([1, 0])
    got = F.cross_entropy(probs.log(), target, reduction="sum").item()
    assert abs(got - (-math.log(.7) - math.log(.6))) < 1e-12


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smol-only", action="store_true", help="Integration retry only; do not repeat completed additional models")
    args = parser.parse_args()
    selected_models = MODELS[:1] if args.smol_only else MODELS
    nll_test()
    torch.manual_seed(8)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_path = ROOT / "results" / f"ppl_smoke_{run_id}.json"
    report = {"run_id": run_id, "protocol": "ppl_smoke_protocol.md", "test_access": False,
              "scope": "restricted validation model-level full-forward; not whole-corpus benchmark",
              "dtype": "float32", "device": "cuda", "threads": 4,
              "calibration_subset": 0, "bits": 4, "group_size": 128,
              "torch": str(torch.__version__), "models": [], "errors": [],
              "selected_models": [m[0] for m in selected_models],
              "integration_retry": args.smol_only}
    def save():
        report["wall_seconds"] = time.monotonic() - START
        output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for filename, sha in SOURCE_HASHES.items():
        assert digest(VENDOR / filename) == sha
    original_manifest_path = D05 / "data" / "wikitext_blocks.json"
    assert digest(original_manifest_path) == MANIFEST_SHA
    original = json.loads(original_manifest_path.read_text(encoding="utf-8"))
    budget_manifest_path = ROOT / "data" / f"calibration_budget_manifest_{BUDGET_ID}.json"
    budget_result = json.loads((ROOT / "results" / f"calibration_budget_{BUDGET_ID}.json").read_text())
    assert budget_result["status"] == "COMPLETE"
    budget_hash = digest(budget_manifest_path)
    assert budget_hash == budget_result["new_manifest_sha256"]
    nested = json.loads(budget_manifest_path.read_text(encoding="utf-8"))
    calibration_ids = [b["token_ids"] for b in nested["calibration"][0]["blocks"]]
    dev_ids = [b["token_ids"] for b in original["development"]]
    smol_tokenizer = AutoTokenizer.from_pretrained(MODELS[0][0], revision=MODELS[0][1], local_files_only=True)
    cal_text = [smol_tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False) for ids in calibration_ids]
    dev_text = [smol_tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False) for ids in dev_ids]
    for repo, revision, filename, block_path in selected_models:
        row = {"repo": repo, "revision": revision, "block_path": block_path, "methods": [], "status": "RUNNING"}
        report["models"].append(row)
        save()
        try:
            gate()
            folder = Path(snapshot_download(repo, revision=revision, local_files_only=True))
            row["checkpoint_sha256"] = digest(folder / filename)
            tok = AutoTokenizer.from_pretrained(folder, local_files_only=True, trust_remote_code=False)
            if repo == MODELS[0][0]:
                cal, dev = calibration_ids, dev_ids
            else:
                cal = [tok.encode(t, add_special_tokens=False) for t in cal_text]
                dev = [tok.encode(t, add_special_tokens=False) for t in dev_text]
            if not all(2 <= len(ids) <= 256 for ids in cal + dev):
                raise RuntimeError("Native fragment length outside frozen [2,256] bound")
            row["calibration_tokens"] = sum(map(len, cal))
            row["development_input_tokens"] = sum(map(len, dev))
            row["expected_targets"] = sum(len(ids) - 1 for ids in dev)
            row["tokenizer_class"] = type(tok).__name__
            tokens_path = ROOT / "results" / f"ppl_tokens_{run_id}_{repo.split('/')[-1]}.json"
            tokens_path.write_text(json.dumps({"repo": repo, "revision": revision,
                "calibration_ids": cal, "development_ids": dev,
                "original_manifest_sha256": MANIFEST_SHA, "nested_manifest_sha256": budget_hash}) + "\n", encoding="utf-8")
            token_hash = digest(tokens_path)
            row["token_manifest"] = str(tokens_path)
            row["token_manifest_sha256"] = token_hash
            del tok
            for method in ("FP", "RTN", "GPTQ"):
                gate()
                entry = {"method": method, "status": "RUNNING"}
                row["methods"].append(entry)
                save()
                load_start = time.perf_counter()
                model = load_model(folder, filename)
                torch.cuda.synchronize()
                entry["load_seconds"] = time.perf_counter() - load_start
                entry["fp_reference_check"] = evaluate(model, dev)
                if method == "FP":
                    fp = entry["fp_reference_check"]
                    entry["evaluation"] = fp
                else:
                    assert entry["fp_reference_check"]["target_tokens"] == fp["target_tokens"]
                    assert abs(entry["fp_reference_check"]["mean_nll"] - fp["mean_nll"]) <= 1e-5
                    torch.cuda.reset_peak_memory_stats()
                    def progress(blocks):
                        entry["completed_blocks"] = blocks
                        save()
                    with (ROOT / "results" / f"gptq_core_{run_id}_{repo.split('/')[-1]}.log").open("a", encoding="utf-8") as log:
                        entry.update(quantize_model(model, block_path, cal, method, log, progress))
                    entry["quant_peak_allocated_MiB"] = torch.cuda.max_memory_allocated() / 2**20
                    entry["quant_peak_reserved_MiB"] = torch.cuda.max_memory_reserved() / 2**20
                    entry["evaluation"] = evaluate(model, dev)
                score = entry["evaluation"]
                assert score["target_tokens"] == row["expected_targets"]
                assert digest(tokens_path) == token_hash
                entry["delta_mean_nll_vs_FP"] = score["mean_nll"] - fp["mean_nll"]
                entry["ppl_rise_vs_FP_percent"] = 100 * (score["ppl"] / fp["ppl"] - 1)
                entry["status"] = "PASS"
                print(repo, method, "PPL", score["ppl"], flush=True)
                del model
                gc.collect()
                torch.cuda.empty_cache()
                save()
            row["status"] = "COMPLETE"
        except Exception as exc:
            row["status"] = "FAILED_OR_PARTIAL"
            row["error"] = repr(exc)
            row["traceback"] = traceback.format_exc()
            report["errors"].append(row["error"])
            if "entry" in locals() and entry.get("status") == "RUNNING":
                entry["status"] = "FAILED"
                entry["error"] = repr(exc)
            if "model" in locals():
                del model
            gc.collect()
            torch.cuda.empty_cache()
            print(repo, row["error"], flush=True)
        finally:
            save()
        if time.monotonic() - START >= 1800:
            break
    assert digest(original_manifest_path) == MANIFEST_SHA
    assert digest(budget_manifest_path) == budget_hash
    report["status"] = "COMPLETE" if len(report["models"]) == len(selected_models) and all(r["status"] == "COMPLETE" for r in report["models"]) else "PARTIAL"
    save()
    print("Saved", output_path, flush=True)


if __name__ == "__main__":
    main()
