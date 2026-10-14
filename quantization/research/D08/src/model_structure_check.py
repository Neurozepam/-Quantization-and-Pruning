"""Model structures from pinned configs on meta device; no model downloads.
Counts unique nn.Parameters; excludes buffers and activations from byte estimate.
"""
import json
from pathlib import Path
import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM


def main():
    root = Path(__file__).resolve().parents[1]
    results = {"torch": torch.__version__, "transformers": transformers.__version__,
               "device": "meta", "weight_downloads": False, "models": []}
    for name in ("pythia-160m", "opt-125m", "gpt2"):
        data = json.loads((root / "configs" / f"{name}.json").read_text())
        model_type = data.pop("model_type", "gpt2")
        config = AutoConfig.for_model(model_type, **data)
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(config)
        count = sum(p.numel() for p in model.parameters())
        suffix = {"pythia-160m": "mlp.dense_4h_to_h", "opt-125m": "fc2",
                  "gpt2": "mlp.c_proj"}[name]
        sites = [{"name": n, "class": type(m).__name__, "shape": list(m.weight.shape)}
                 for n, m in model.named_modules() if n.endswith(suffix)]
        assert len(sites) == 12, (name, len(sites))
        row = {"name": name, "unique_parameter_count": count,
               "fp32_parameters_MiB": count * 4 / 2**20,
               "ffn_output_sites": sites}
        results["models"].append(row)
        print(name, count, f"FP32 parameters {row['fp32_parameters_MiB']:.2f} MiB",
              "first site", sites[0])
        del model
    output = root / "results" / "model_structure_check.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print("Saved", output)


if __name__ == "__main__":
    main()
