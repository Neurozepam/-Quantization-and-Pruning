"""Deterministic descriptive summary of frozen diagnostic; no selection/tuning."""
import json
import statistics
import sys
from pathlib import Path


def mean(values):
    return statistics.mean(values)


def main():
    source = Path(sys.argv[1])
    run = json.loads(source.read_text(encoding="utf-8"))
    assert run["status"] == "COMPLETE"
    assert len(run["pairs"]) == 24 and all(p["status"] == "PASS" for p in run["pairs"])
    assert run["snapshot_sha256_before"] == run["snapshot_sha256_after"]
    summary = {"run_id": run["run_id"], "wall_seconds": run["wall_seconds"], "sites": [],
               "max_peak_allocated_MiB": max(p["peak_allocated_MiB"] for p in run["pairs"]),
               "max_peak_reserved_MiB": max(p["peak_reserved_MiB"] for p in run["pairs"]),
               "quant_seconds_sum": sum(p["GPTQ"]["quant_seconds"] for p in run["pairs"]),
               "h_seconds_sum": sum(p["GPTQ"]["h_seconds"] for p in run["pairs"])}
    for site in dict.fromkeys(p["site"] for p in run["pairs"]):
        pairs = [p for p in run["pairs"] if p["site"] == site]
        row = {"site": site, "pairs": len(pairs), "methods": {}}
        for method in ("RTN", "GPTQ"):
            result = {"weight_mse_mean": mean(p[method]["weight_mse"] for p in pairs),
                      "mean_occupied_codes": mean(p[method]["codes"]["mean_occupied_codes_per_row_group"] for p in pairs),
                      "mean_code_histogram": [mean(p[method]["codes"]["code_histogram"][k] for p in pairs) for k in range(16)]}
            for split in ("calibration", "development"):
                result[split] = {key: mean(p[method][split][key] for p in pairs) for key in (
                    "output_mse", "normalized_output_mse", "diagonal_only_mse", "net_cross_channel_mse")}
                result[split]["output_mse_sample_std"] = statistics.stdev(p[method][split]["output_mse"] for p in pairs)
            row["methods"][method] = result
        for split in ("calibration", "development"):
            delta = [100 * (p["GPTQ"][split]["output_mse"] / p["RTN"][split]["output_mse"] - 1) for p in pairs]
            row[split + "_delta_percent"] = {"mean": mean(delta), "min": min(delta), "max": max(delta),
                                               "wins": sum(v < -1e-6 for v in delta),
                                               "losses": sum(v > 1e-6 for v in delta)}
        summary["sites"].append(row)
    destination = source.with_name(source.stem + "_summary.json")
    destination.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("Saved", destination)


if __name__ == "__main__":
    main()
