"""Approved frozen D08 cached-input W-only diagnostic; no model/test loading."""
import gc
import hashlib
import json
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
import torch
from weight_baseline import VENDOR, gptq, occupancy, reconstruction, rtn
from resource_smoke import resources

ROOT = Path(__file__).resolve().parents[1]
D05 = ROOT.parent / "D05"
MANIFEST_SHA = "ee5d038f8c1c37a45624ce2df92e2090277c28c9ab5fb823b755e395210d59b4"
REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
SITES = [f"model.layers.{i}.mlp.down_proj" for i in (0, 15, 29)]
SOURCE_HASHES = {
    "gptq.py": "f71e511ef46ee6ebadf0e343d171ee99ab1d763fb5308dbb6fa98dc1ed37b1bb",
    "quant.py": "daaf2064336d8aed1e8f37127fd89a17d0cd3465a26c08f966f6b3a63eff4aa5",
}


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


@torch.inference_mode()
def main():
    started = time.monotonic()
    torch.manual_seed(8)
    torch.set_num_threads(4)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = ROOT / "results" / f"weight_diagnostic_{run_id}.json"
    report = {"run_id": run_id, "protocol": "diagnostic_protocol.md", "torch": str(torch.__version__),
              "seed": 8, "cpu_threads": 4, "test_access": False,
              "scope": "cached FP-input restricted-site W-only; not full GPTQ/PPL",
              "bits": 4, "group_size": 128, "parameters": {"percdamp": .01, "blocksize": 128,
                "sym": False, "mse": False, "actorder": False, "static_groups": False},
              "pairs": [], "errors": [], "source_hashes": {}}
    path = D05 / "data" / "wikitext_activations.pt"
    manifest = D05 / "data" / "wikitext_blocks.json"
    try:
        assert digest(manifest) == MANIFEST_SHA
        for name, expected in SOURCE_HASHES.items():
            actual = digest(VENDOR / name)
            assert actual == expected
            report["source_hashes"][name] = actual
        available = resources()
        report["resources_before"] = available
        if available["free_gpu_MiB"] < 2048 or available["free_ram_GiB"] < 2.5:
            raise RuntimeError(f"Resource gate failed: {available}")
        before = digest(path)
        report["snapshot_sha256_before"] = before
        snapshot = torch.load(path, map_location="cpu", weights_only=True)
        assert snapshot["metadata"]["manifest_sha256"] == MANIFEST_SHA
        assert snapshot["metadata"]["revision"] == REVISION
        for site in SITES:
            w = snapshot["weights"][site].to("cuda")
            dev = snapshot["inputs"]["development"][site].to("cuda")
            assert w.shape == (576, 1536) and dev.shape == (2048, 1536)
            assert w.dtype == dev.dtype == torch.float32
            assert torch.isfinite(w).all() and torch.isfinite(dev).all()
            torch.cuda.synchronize()
            tick = time.perf_counter()
            qr, pr = rtn(w)
            torch.cuda.synchronize()
            rt = time.perf_counter() - tick
            r_meta = {"weight_mse": (qr - w).square().mean().item(),
                      "development": reconstruction(dev, w, qr),
                      "codes": occupancy(qr, pr), "quant_seconds": rt}
            for subset in range(8):
                if time.monotonic() - started > 1200:
                    raise TimeoutError("20-minute diagnostic cap reached")
                pair = {"site": site, "subset": subset}
                report["pairs"].append(pair)
                try:
                    x = snapshot["inputs"][f"calibration_{subset}"][site].to("cuda")
                    assert x.shape == (256, 1536) and x.dtype == torch.float32
                    assert torch.isfinite(x).all()
                    pair["RTN"] = {**r_meta, "calibration": reconstruction(x, w, qr)}
                    torch.cuda.reset_peak_memory_stats()
                    qg, pg, cost = gptq(w, x)
                    assert torch.isfinite(qg).all()
                    pair["GPTQ"] = {"weight_mse": (qg - w).square().mean().item(),
                                    "calibration": reconstruction(x, w, qg),
                                    "development": reconstruction(dev, w, qg),
                                    "codes": occupancy(qg, pg), **cost}
                    pair["peak_allocated_MiB"] = torch.cuda.max_memory_allocated() / 2**20
                    pair["peak_reserved_MiB"] = torch.cuda.max_memory_reserved() / 2**20
                    pair["dev_mse_relative_delta_percent"] = 100 * (
                        pair["GPTQ"]["development"]["output_mse"] /
                        pair["RTN"]["development"]["output_mse"] - 1
                    )
                    assert torch.equal(w.cpu(), snapshot["weights"][site])
                    assert torch.equal(x.cpu(), snapshot["inputs"][f"calibration_{subset}"][site])
                    assert torch.equal(dev.cpu(), snapshot["inputs"]["development"][site])
                    pair["status"] = "PASS"
                    print(site, subset, f"development output-MSE delta {pair['dev_mse_relative_delta_percent']:+.2f}%", flush=True)
                    del qg, pg, x
                except Exception as exc:
                    pair["status"] = "FAILED"
                    pair["error"] = repr(exc)
                    report["errors"].append(traceback.format_exc())
                    raise
                finally:
                    report["wall_seconds"] = time.monotonic() - started
                    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            del w, dev, qr, pr
            gc.collect()
            torch.cuda.empty_cache()
        report["snapshot_sha256_after"] = digest(path)
        assert report["snapshot_sha256_after"] == before
        assert digest(manifest) == MANIFEST_SHA
        report["status"] = "COMPLETE"
    except Exception as exc:
        report["status"] = "FAILED_OR_PARTIAL"
        report["error"] = repr(exc)
        report["errors"].append(traceback.format_exc())
        print(report["error"], flush=True)
    finally:
        report["wall_seconds"] = time.monotonic() - started
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print("Saved", output, flush=True)


if __name__ == "__main__":
    main()
