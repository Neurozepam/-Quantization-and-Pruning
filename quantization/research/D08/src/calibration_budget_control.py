"""Approved nested2048-token control. Local train only; original dev cache.
No dataset/test split loader, no model/data network access, no sweep.
"""
import gc
import hashlib
import importlib.util
import json
import random
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from resource_smoke import resources
from weight_baseline import VENDOR, gptq, occupancy, reconstruction, rtn
from weight_diagnostic import D05, MANIFEST_SHA, REVISION, SITES, SOURCE_HASHES, digest

ROOT = Path(__file__).resolve().parents[1]
OLD_RUN = ROOT / "results" / "weight_diagnostic_20261014T144533848026Z.json"
START = time.monotonic()


def check_time():
    if time.monotonic() - START > 900:
        raise TimeoutError("Approved15-minute budget exceeded")


def source_helpers():
    path = D05 / "src" / "prepare_wikitext.py"
    spec = importlib.util.spec_from_file_location("d05_budget_source", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def expanded_manifest(original, pool):
    by_row = {a["article_start_row"]: a for a in pool}
    result = []
    for subset in original["calibration"]:
        sid = subset["subset_id"]
        prefix = subset["blocks"]
        assert len(prefix) == 2
        used = {b["article_start_row"] for b in prefix}
        assert len(used) == 2
        for block in prefix:
            article = by_row[block["article_start_row"]]
            assert article["article_sha256"] == block["article_sha256"]
            start = block["token_start"]
            assert article["token_ids"][start:start + 128] == block["token_ids"]
        rng = random.Random(8042 + sid)
        extra = []
        for article in rng.sample([a for a in pool if a["article_start_row"] not in used], 14):
            start = rng.randint(0, len(article["token_ids"]) - 128)
            extra.append({"split": "train", "article_title": article["article_title"],
                          "article_start_row": article["article_start_row"],
                          "article_sha256": article["article_sha256"], "token_start": start,
                          "token_ids": article["token_ids"][start:start + 128]})
        blocks = prefix + extra
        assert len(blocks) == 16 and len({b["article_start_row"] for b in blocks}) == 16
        assert blocks[:2] == prefix
        result.append({"subset_id": sid, "extension_seed": 8042 + sid, "blocks": blocks})
    assert [s["subset_id"] for s in result] == list(range(8))
    return result


@torch.inference_mode()
def capture(manifest, snapshot, report):
    check_time()
    r = resources()
    report["resources_before_model"] = r
    if r["free_gpu_MiB"] < 2048 or r["free_ram_GiB"] < 3:
        raise RuntimeError(f"Resource gate failed: {r}")
    tick = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        "HuggingFaceTB/SmolLM2-135M", revision=REVISION, local_files_only=True,
        use_safetensors=True, torch_dtype=torch.float32, attn_implementation="eager",
        trust_remote_code=False,
    ).to("cuda").eval()
    assert model.config._commit_hash == REVISION
    report["load_seconds"] = time.perf_counter() - tick
    original_weights = {}
    for name in SITES:
        weight = model.get_submodule(name).weight.detach().cpu().clone()
        assert torch.equal(weight, snapshot["weights"][name])
        original_weights[name] = weight
    torch.cuda.reset_peak_memory_stats()
    chunks = {name: [] for name in SITES}
    handles = []
    def hook(name):
        def receive(module, inputs, output):
            chunks[name].append(inputs[0].detach().squeeze(0).cpu())
        return receive
    for name in SITES:
        handles.append(model.get_submodule(name).register_forward_hook(hook(name)))
    expanded = {}
    try:
        torch.cuda.synchronize()
        tick = time.perf_counter()
        for subset in manifest["calibration"]:
            sid = subset["subset_id"]
            for name in SITES:
                chunks[name].clear()
            for block in subset["blocks"]:
                check_time()
                ids = torch.tensor([block["token_ids"]], device="cuda", dtype=torch.long)
                assert ids.shape == (1, 128)
                out = model(input_ids=ids, use_cache=False)
                assert torch.isfinite(out.logits).all()
                del out, ids
            expanded[f"calibration_{sid}"] = {}
            for name in SITES:
                x = torch.cat(chunks[name], dim=0)
                assert x.shape == (2048, 1536) and torch.isfinite(x).all()
                prefix = snapshot["inputs"][f"calibration_{sid}"][name]
                torch.testing.assert_close(x[:256], prefix, rtol=1e-5, atol=1e-6)
                expanded[f"calibration_{sid}"][name] = torch.cat((prefix, x[256:]), dim=0)
            print("Captured nested subset", sid, flush=True)
        torch.cuda.synchronize()
        report["capture_seconds"] = time.perf_counter() - tick
        report["capture_peak_allocated_MiB"] = torch.cuda.max_memory_allocated() / 2**20
        report["capture_peak_reserved_MiB"] = torch.cuda.max_memory_reserved() / 2**20
    finally:
        for handle in handles:
            handle.remove()
    for name in SITES:
        assert not model.get_submodule(name)._forward_hooks
        assert torch.equal(model.get_submodule(name).weight.detach().cpu(), original_weights[name])
    del model, chunks
    gc.collect()
    torch.cuda.empty_cache()
    return expanded


@torch.inference_mode()
def main():
    torch.manual_seed(8)
    torch.set_num_threads(4)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    report_path = ROOT / "results" / f"calibration_budget_{run_id}.json"
    report = {"run_id": run_id, "protocol": "calibration_budget_protocol.md",
              "model_revision": REVISION, "calibration_tokens": 2048,
              "sequence_length": 128, "batch_size": 1, "threads": 4,
              "test_access": False, "model_or_corpus_download": False,
              "original_run": str(OLD_RUN), "pairs": [], "errors": []}
    def save():
        report["wall_seconds"] = time.monotonic() - START
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    original_path = D05 / "data" / "wikitext_activations.pt"
    original_manifest = D05 / "data" / "wikitext_blocks.json"
    try:
        old_run = json.loads(OLD_RUN.read_text(encoding="utf-8"))
        assert old_run["status"] == "COMPLETE" and len(old_run["pairs"]) == 24
        old_pairs = {(p["site"], p["subset"]): p for p in old_run["pairs"]}
        assert digest(original_manifest) == MANIFEST_SHA
        before = digest(original_path)
        assert before == old_run["snapshot_sha256_before"] == old_run["snapshot_sha256_after"]
        report["original_snapshot_sha256_before"] = before
        for filename, sha in SOURCE_HASHES.items():
            assert digest(VENDOR / filename) == sha
        snapshot = torch.load(original_path, weights_only=True, map_location="cpu")
        helpers = source_helpers()
        tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M", revision=REVISION,
                                                  local_files_only=True, trust_remote_code=False)
        train_path = Path(hf_hub_download(
            "Salesforce/wikitext", "wikitext-2-raw-v1/train-00000-of-00001.parquet",
            repo_type="dataset", revision=helpers.DATA_REVISION, local_files_only=True,
        ))
        report["train_parquet_sha256"] = digest(train_path)
        rows = pq.read_table(train_path, columns=["text"])["text"].to_pylist()
        pool = helpers.build_pool(rows, tokenizer, "train")
        original = json.loads(original_manifest.read_text(encoding="utf-8"))
        manifest = {"model": original["model"], "tokenizer_revision": REVISION,
                    "dataset": original["dataset"], "dataset_revision": helpers.DATA_REVISION,
                    "sequence_length": 128, "original_manifest_sha256": MANIFEST_SHA,
                    "test_accessed": False, "calibration": expanded_manifest(original, pool)}
        all_blocks = [tuple(b["token_ids"]) for s in manifest["calibration"] for b in s["blocks"]]
        manifest["total_blocks"] = len(all_blocks)
        manifest["unique_blocks_across_subsets"] = len(set(all_blocks))
        data_dir = ROOT / "data"
        data_dir.mkdir(exist_ok=True)
        manifest_path = data_dir / f"calibration_budget_manifest_{run_id}.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        report["new_manifest"] = str(manifest_path)
        report["new_manifest_sha256"] = digest(manifest_path)
        del tokenizer, rows, pool
        save()
        expanded = capture(manifest, snapshot, report)
        expanded_path = data_dir / f"calibration_budget_snapshot_{run_id}.pt"
        torch.save({"inputs": expanded, "metadata": {"model_revision": REVISION,
                   "manifest_sha256": report["new_manifest_sha256"], "original_snapshot_sha256": before}}, expanded_path)
        report["expanded_snapshot"] = str(expanded_path)
        report["expanded_snapshot_sha256"] = digest(expanded_path)
        save()
        for site in SITES:
            w = snapshot["weights"][site].to("cuda")
            dev = snapshot["inputs"]["development"][site].to("cuda")
            qr, pr = rtn(w)
            rtn_dev = reconstruction(dev, w, qr)
            for sid in range(8):
                check_time()
                x = expanded[f"calibration_{sid}"][site].to("cuda")
                pair = {"site": site, "subset": sid, "old_budget_GPTQ": old_pairs[(site, sid)]["GPTQ"]}
                report["pairs"].append(pair)
                pair["RTN"] = {"weight_mse": (qr - w).square().mean().item(),
                               "development": rtn_dev, "calibration": reconstruction(x, w, qr),
                               "codes": occupancy(qr, pr)}
                assert abs(rtn_dev["output_mse"] / old_pairs[(site, sid)]["RTN"]["development"]["output_mse"] - 1) < 1e-5
                torch.cuda.reset_peak_memory_stats()
                qg, pg, cost = gptq(w, x)
                pair["GPTQ"] = {"weight_mse": (qg - w).square().mean().item(),
                                "development": reconstruction(dev, w, qg),
                                "calibration": reconstruction(x, w, qg), "codes": occupancy(qg, pg), **cost}
                pair["peak_allocated_MiB"] = torch.cuda.max_memory_allocated() / 2**20
                pair["peak_reserved_MiB"] = torch.cuda.max_memory_reserved() / 2**20
                new_mse = pair["GPTQ"]["development"]["output_mse"]
                pair["dev_delta_vs_RTN_percent"] = 100 * (new_mse / rtn_dev["output_mse"] - 1)
                pair["dev_delta_vs_256_GPTQ_percent"] = 100 * (new_mse / pair["old_budget_GPTQ"]["development"]["output_mse"] - 1)
                assert torch.equal(w.cpu(), snapshot["weights"][site])
                assert torch.equal(x.cpu(), expanded[f"calibration_{sid}"][site])
                assert torch.equal(dev.cpu(), snapshot["inputs"]["development"][site])
                pair["status"] = "PASS"
                print(site, sid, f"dev versus RTN {pair['dev_delta_vs_RTN_percent']:+.2f}%; versus old GPTQ {pair['dev_delta_vs_256_GPTQ_percent']:+.2f}%", flush=True)
                del x, qg, pg
                save()
            del w, dev, qr, pr
            gc.collect()
            torch.cuda.empty_cache()
        assert digest(original_manifest) == MANIFEST_SHA
        report["original_snapshot_sha256_after"] = digest(original_path)
        assert report["original_snapshot_sha256_after"] == before
        assert digest(manifest_path) == report["new_manifest_sha256"]
        assert digest(expanded_path) == report["expanded_snapshot_sha256"]
        report["status"] = "COMPLETE"
    except Exception as exc:
        report["status"] = "FAILED_OR_PARTIAL"
        report["error"] = repr(exc)
        report["errors"].append(traceback.format_exc())
        print(report["error"], flush=True)
    finally:
        save()
        print("Saved", report_path, flush=True)


if __name__ == "__main__":
    main()
