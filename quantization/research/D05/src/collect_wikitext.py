"""D05 learner scaffold: collect unchanged FP inputs of first/middle/last down_proj.

Fill capture_input and combine_chunks. No quantization, test, or parameter search.
"""
import functools
import hashlib
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[1]


def capture_input(module, inputs, output, *, chunks):
    """Append [128,in_features] input to chunks; detach and move to CPU."""
    inputs = inputs[0]
    inputs = inputs.detach().cpu().squeeze(0)
    chunks.append(inputs)


def combine_chunks(chunks):
    """Join [128,in_features] chunks along the token dimension."""
    chunks = [chunk.detach().cpu() for chunk in chunks]

    return torch.cat(chunks, dim=0)


def test_helpers():
    x = torch.arange(24, dtype=torch.float32).reshape(1, 4, 6).requires_grad_()
    chunks = []
    result = capture_input(None, (x,), None, chunks=chunks)
    assert result is None
    assert len(chunks) == 1 and chunks[0].shape == (4, 6)
    assert chunks[0].device.type == 'cpu' and not chunks[0].requires_grad
    assert torch.equal(chunks[0], x.detach().squeeze(0))
    combined = combine_chunks([chunks[0], chunks[0]])
    assert combined.shape == (8, 6)
    assert torch.equal(combined[:4], chunks[0])
    assert torch.equal(combined[4:], chunks[0])
    print('Hook and concatenation tests: PASS')


def main():
    test_helpers()
    manifest_path = ROOT / 'data' / 'wikitext_blocks.json'
    destination = ROOT / 'data' / 'wikitext_activations.pt'
    if destination.exists():
        raise FileExistsError(f'Refusing to overwrite: {destination}')
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    assert manifest['sequence_length'] == 128
    assert not manifest['test_accessed']
    if not torch.cuda.is_available():
        raise RuntimeError('This smoke configuration requires CUDA')
    model = AutoModelForCausalLM.from_pretrained(
        manifest['model'], revision=manifest['tokenizer_revision'], local_files_only=True,
        torch_dtype=torch.float32, attn_implementation='eager').to('cuda').eval()
    assert model.config._commit_hash == manifest['tokenizer_revision']
    n = model.config.num_hidden_layers
    layer_names = [f'model.layers.{i}.mlp.down_proj' for i in sorted({0, n // 2, n - 1})]
    targets = {name: model.get_submodule(name) for name in layer_names}
    weight_before = {name: module.weight.detach().cpu().clone() for name, module in targets.items()}

    def collect(blocks):
        chunks = {name: [] for name in targets}
        handles = [module.register_forward_hook(
            functools.partial(capture_input, chunks=chunks[name])) for name, module in targets.items()]
        try:
            with torch.inference_mode():
                for block in blocks:
                    ids = torch.tensor([block['token_ids']], dtype=torch.long, device='cuda')
                    assert ids.shape == (1, 128)
                    result = model(input_ids=ids, use_cache=False)
                    del result
        finally:
            for handle in handles:
                handle.remove()
        tensors = {name: combine_chunks(parts) for name, parts in chunks.items()}
        for name, tensor in tensors.items():
            assert len(chunks[name]) == len(blocks)
            assert tensor.shape == (len(blocks) * 128, targets[name].in_features)
            assert tensor.device.type == 'cpu' and tensor.dtype == torch.float32
            assert torch.isfinite(tensor).all()
        return tensors

    torch.cuda.synchronize()
    start = time.perf_counter()
    inputs = {f"calibration_{subset['subset_id']}": collect(subset['blocks'])
              for subset in manifest['calibration']}
    inputs['development'] = collect(manifest['development'])
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    for name, module in targets.items():
        assert not module._forward_hooks
        assert torch.equal(weight_before[name], module.weight.detach().cpu())
    snapshot = dict(inputs=inputs, weights=weight_before, metadata=dict(
        model=manifest['model'], revision=model.config._commit_hash,
        manifest_sha256=hashlib.sha256(raw).hexdigest(), layers=layer_names,
        dtype='float32', gpu=torch.cuda.get_device_name(0), sequence_length=128,
        collection_seconds=seconds, scope='FP activation collection only; no quantization',
        target_weights_unchanged=True, use_cache=False, attention='eager'))
    torch.save(snapshot, destination)
    print('Saved:', destination)
    print('Layers:', layer_names)
    print('Collection including hooks and CPU transfers:', seconds, 's')


if __name__ == '__main__':
    main()
