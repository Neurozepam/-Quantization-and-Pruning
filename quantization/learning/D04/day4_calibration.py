"""DAY 4: сбор FP32-входов одного слоя для локального calibration-эксперимента."""

import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent / "artefacts"
MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
TARGET_LAYER = "model.layers.0.mlp.down_proj"
SEQUENCE_LENGTH = 32


def main():
    corpus = json.loads((ROOT / "calibration_texts.json").read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    ).to("cuda")
    model.eval()
    target = model.get_submodule(TARGET_LAYER)

    def collect_inputs(texts):
        chunks = []

        def capture_hook(module, inputs, output):
            chunks.append(inputs[0].detach().cpu().squeeze(0))

        handle = target.register_forward_hook(capture_hook)
        try:
            with torch.inference_mode():
                for text in texts:
                    tokens = tokenizer(
                        text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=SEQUENCE_LENGTH,
                    ).to("cuda")
                    assert tokens["input_ids"].shape == (1, SEQUENCE_LENGTH)
                    model(**tokens, use_cache=False)
        finally:
            handle.remove()

        # Строки — токены, столбцы — входные координаты слоя.
        pooled = torch.cat(chunks, dim=0)
        assert pooled.shape == (len(texts) * SEQUENCE_LENGTH, target.in_features)
        return pooled

    groups = {
        "representative": corpus["representative"],
        "unrelated": corpus["unrelated"],
        "tiny": corpus["representative"][:1],
        "evaluation": corpus["evaluation"],
    }
    input_sets = {}
    for name, texts in groups.items():
        input_sets[name] = collect_inputs(texts)
        print(name, tuple(input_sets[name].shape), input_sets[name].device)

    assert not target._forward_hooks
    snapshot = {
        "inputs": input_sets,
        "weight": target.weight.detach().cpu(),
        "metadata": {
            "model_id": MODEL_ID,
            "model_revision": model.config._commit_hash,
            "target_layer": TARGET_LAYER,
            "sequence_length": SEQUENCE_LENGTH,
            "torch_version": str(torch.__version__),
            "device": torch.cuda.get_device_name(0),
            "corpus_source": corpus["metadata"]["source"],
            "scope": "local input quantization of one Linear layer; model weights remain FP32",
        },
    }
    path = ROOT / "calibration_snapshot.pt"
    torch.save(snapshot, path)
    print("Saved:", path)


if __name__ == "__main__":
    main()
