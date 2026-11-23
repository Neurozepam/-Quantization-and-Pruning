"""Evaluate the uncompressed checkpoints used in the matched CV and LLM runs."""

import argparse
import hashlib
import json
import math
import runpy
import tempfile
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "results" / ".cache"
CV_SOURCE = ROOT / "quantization" / "research" / "cnn_ptq"
MODEL_ID = "HuggingFaceTB/SmolLM-135M"
MODEL_REVISION = "1d461723eec654e65efdc40cf49301c89c0c92f4"
DATASET_ID = "Salesforce/wikitext"
DATASET_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
CV_CHECKPOINT = CV_SOURCE / "artifacts" / "fp32" / "model.pt"
CV_CHECKPOINT_SHA256 = "5a62bdc523480c70a458b924be9af254c75628312d46f145df66f648974ee75a"


def evaluate_cv():
    ptq = runpy.run_path(str(CV_SOURCE / "run.py"))
    checkpoint_hash = hashlib.sha256(CV_CHECKPOINT.read_bytes()).hexdigest()
    if checkpoint_hash != CV_CHECKPOINT_SHA256:
        raise ValueError(f"Unexpected LeNet checkpoint SHA-256: {checkpoint_hash}")

    torch.set_num_threads(1)
    model = ptq["LeNet"]().eval()
    model.load_state_dict(torch.load(CV_CHECKPOINT, map_location="cpu", weights_only=True))
    images, labels = ptq["load_mnist"](train=False)
    with tempfile.TemporaryDirectory(dir=ROOT / "baseline") as directory:
        _, measured = ptq["evaluate"](model, images, labels, Path(directory) / "model.pt", None)

    return {
        "model": "LeNet + BatchNorm",
        "dtype": "float32",
        "dataset": "MNIST",
        "split": "test",
        "examples": len(labels),
        "normalization_mean": 0.1307,
        "normalization_std": 0.3081,
        "checkpoint": CV_CHECKPOINT.relative_to(ROOT).as_posix(),
        "checkpoint_sha256": checkpoint_hash,
        "params": sum(param.numel() for param in model.parameters()),
        "correct": measured["correct"],
        "accuracy_pct": measured["accuracy_pct"],
        "state_dict_bytes": measured["state_dict_bytes"],
        "cpu_threads": 1,
        "cpu_ms_batch_1": measured["cpu_ms_batch_1"],
        "cpu_ms_batch_256": measured["cpu_ms_batch_256"],
    }


@torch.inference_mode()
def score_llm(model, ids, context=512, stride=256, batch_windows=8):
    windows = []
    previous_end = 0
    for start in range(0, ids.numel(), stride):
        end = min(start + context, ids.numel())
        windows.append((ids[start:end], end - previous_end))
        previous_end = end
        if end == ids.numel():
            break

    total_nll, scored_tokens = 0.0, 0
    for offset in range(0, len(windows), batch_windows):
        subset = windows[offset:offset + batch_windows]
        full = [(chunk, added) for chunk, added in subset if chunk.numel() == context]
        tail = [(chunk, added) for chunk, added in subset if chunk.numel() != context]
        for batch in (full, tail):
            if not batch:
                continue
            chunks = torch.stack([chunk for chunk, _ in batch])
            # The first token of the stream has no preceding token to predict it.
            counts = [min(added, chunks.size(1) - 1) for _, added in batch]
            hidden = model.model(chunks, use_cache=False).last_hidden_state
            kept = torch.cat([hidden[i, -n-1:-1] for i, n in enumerate(counts)])
            labels = torch.cat([chunks[i, -n:] for i, n in enumerate(counts)])
            losses = F.cross_entropy(model.lm_head(kept).float(), labels, reduction="none")
            total_nll += sum(torch.stack([part.sum() for part in losses.split(counts)]).tolist())
            scored_tokens += sum(counts)
        if (offset + batch_windows) % 256 == 0:
            print(f"SmolLM: {offset + batch_windows}/{len(windows)} windows", flush=True)

    if scored_tokens != ids.numel() - 1:
        raise RuntimeError(f"Wrong token count: {scored_tokens} / {ids.numel() - 1}")
    return total_nll, scored_tokens, len(windows)


def evaluate_llm():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("Matching the SmolLM experiment requires CUDA FP16.")
    torch.set_num_threads(4)

    snapshot = CACHE / "hf/hub/models--HuggingFaceTB--SmolLM-135M/snapshots" / MODEL_REVISION
    model_source = str(snapshot) if (snapshot / "model.safetensors").is_file() else MODEL_ID
    test_parquet = CACHE / "wikitext-2-raw-test.parquet"
    if test_parquet.is_file():
        from pyarrow.parquet import read_table

        texts = read_table(test_parquet, columns=["text"]).column("text").to_pylist()
    else:
        from datasets import load_dataset

        texts = load_dataset(DATASET_ID, "wikitext-2-raw-v1", split="test",
                             revision=DATASET_REVISION)["text"]
    texts = [text for text in texts if text.strip()]
    tokenizer = AutoTokenizer.from_pretrained(model_source, revision=MODEL_REVISION)
    ids = tokenizer("\n\n".join(texts), return_tensors="pt", verbose=False).input_ids[0].cuda()
    model = AutoModelForCausalLM.from_pretrained(
        model_source, revision=MODEL_REVISION, torch_dtype=torch.float16,
    ).cuda().eval()
    total_nll, scored_tokens, windows = score_llm(model, ids)
    params = sum(param.numel() for param in model.parameters())
    mean_nll = total_nll / scored_tokens
    return {
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "dtype": "float16",
        "dataset": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "dataset_config": "wikitext-2-raw-v1",
        "split": "test",
        "nonblank_rows": len(texts),
        "scored_tokens": scored_tokens,
        "window_tokens": 512,
        "stride_tokens": 256,
        "windows": windows,
        "params": params,
        "parameter_payload_bytes": params * 2,
        "mean_nll": mean_nll,
        "perplexity": math.exp(mean_nll),
        "gpu": torch.cuda.get_device_name(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "baseline" / "results.json")
    args = parser.parse_args()
    cv = evaluate_cv()
    print(f"LeNet: {cv['correct']}/{cv['examples']}, {cv['accuracy_pct']:.2f}%", flush=True)
    llm = evaluate_llm()
    print(f"SmolLM: PPL {llm['perplexity']:.8f} on {llm['scored_tokens']:,} target tokens")
    result = {"torch": torch.__version__, "cv": cv, "transformers": llm}
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
