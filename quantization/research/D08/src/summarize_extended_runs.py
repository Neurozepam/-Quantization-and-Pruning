"""Summarize explicit approved runs; preserve original failures, no best-run selection."""
import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    return json.loads((ROOT / "results" / name).read_text(encoding="utf-8"))


def main():
    budget = read("calibration_budget_20261014T150623521707Z.json")
    assert budget["status"] == "COMPLETE" and len(budget["pairs"]) == 24
    bsummary = {"run_id": budget["run_id"], "wall_seconds": budget["wall_seconds"],
                "capture_seconds": budget["capture_seconds"], "sites": []}
    for site in dict.fromkeys(p["site"] for p in budget["pairs"]):
        pairs = [p for p in budget["pairs"] if p["site"] == site]
        assert all(p["status"] == "PASS" for p in pairs)
        bsummary["sites"].append({"site": site,
            "dev_delta_vs_RTN_percent_mean": statistics.mean(p["dev_delta_vs_RTN_percent"] for p in pairs),
            "dev_delta_vs_256_GPTQ_percent_mean": statistics.mean(p["dev_delta_vs_256_GPTQ_percent"] for p in pairs),
            "wins_vs_RTN": sum(p["dev_delta_vs_RTN_percent"] < 0 for p in pairs),
            "wins_vs_256_GPTQ": sum(p["dev_delta_vs_256_GPTQ_percent"] < 0 for p in pairs)})
    (ROOT / "results" / "calibration_budget_20261014T150623521707Z_summary.json").write_text(json.dumps(bsummary, indent=2) + "\n")
    first = read("ppl_smoke_20261014T151532726121Z.json")
    retry = read("ppl_smoke_20261014T152009405310Z.json")
    assert retry["status"] == "COMPLETE" and retry["integration_retry"]
    rows = []
    combined = {"sources": [first["run_id"], retry["run_id"]],
                "initial_errors_preserved": first["errors"], "models": []}
    models = [(retry, retry["models"][0])] + [(first, m) for m in first["models"][1:]]
    for source, model in models:
        assert model["status"] == "COMPLETE"
        brief = {k: model[k] for k in ("repo", "revision", "calibration_tokens", "expected_targets", "token_manifest_sha256")}
        brief["source_run"] = source["run_id"]
        brief["methods"] = []
        for result in model["methods"]:
            assert result["status"] == "PASS"
            score = result["evaluation"]
            row = {"model": model["repo"], "method": result["method"], "ppl": score["ppl"],
                   "ppl_rise_vs_FP_percent": result["ppl_rise_vs_FP_percent"],
                   "mean_nll": score["mean_nll"], "delta_mean_nll_vs_FP": result["delta_mean_nll_vs_FP"],
                   "calibration_tokens": model["calibration_tokens"], "target_tokens": score["target_tokens"],
                   "quantization_cycle_seconds": result.get("quantization_cycle_seconds", 0),
                   "quant_peak_allocated_MiB": result.get("quant_peak_allocated_MiB"),
                   "quant_peak_reserved_MiB": result.get("quant_peak_reserved_MiB"),
                   "quantized_linear_count": result.get("quantized_linear_count", 0),
                   "evaluation_seconds": score["evaluation_seconds"], "source_run": source["run_id"]}
            rows.append(row)
            brief["methods"].append(row)
        combined["models"].append(brief)
    (ROOT / "results" / "ppl_smoke_combined_summary.json").write_text(json.dumps(combined, indent=2) + "\n")
    with (ROOT / "results" / "ppl_smoke_combined.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("Saved budget summary and PPL combined JSON/CSV")


if __name__ == "__main__":
    main()
