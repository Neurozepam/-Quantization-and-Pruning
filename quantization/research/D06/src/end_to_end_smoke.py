"""Evaluate causal token NLL with static activation fake quantization.

Uses the D05 manifest and cached calibration activations. Selected down_proj
inputs are replaced with pre-hooks; all weights remain FP32. --test runs CPU
checks without loading a model or evaluating development data.
"""
import argparse
import csv
import functools
import hashlib
import json
import math
import platform
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
D05 = ROOT.parent / 'D05'
D04 = ROOT.parents[1] / 'learning' / 'D04'
sys.path.insert(0, str(D04))
from day4_experiment import fake_quantize

MODEL = 'HuggingFaceTB/SmolLM2-135M'
REVISION = '93efa2f097d58c2a74874c7e644dbc9b0cee75a2'
MANIFEST_SHA = 'ee5d038f8c1c37a45624ce2df92e2090277c28c9ab5fb823b755e395210d59b4'
LAYERS = [f'model.layers.{i}.mlp.down_proj' for i in (0, 15, 29)]


def shifted_token_nll(logits, input_ids):
    """Return summed next-token NLL and target count for unpadded blocks."""
    prediction_logits = logits[:, :-1, :]
    target_ids = input_ids[:, 1:]

    vocab_size = logits.shape[-1]
    flat_logits = prediction_logits.reshape(-1, vocab_size)
    flat_targets = target_ids.reshape(-1)

    loss = F.cross_entropy(
        flat_logits,
        flat_targets,
        reduction='sum',
    )

    target_count = flat_targets.numel()

    return loss, target_count


def quantize_input_hook(module, inputs, *, alpha):
    """Replace the live first positional input without modifying it in place."""
    x = inputs[0]
    x_hat = fake_quantize(x, alpha)
    return (x_hat,) + inputs[1:]


def summarize(total_nll, targets):
    if targets <= 0 or not math.isfinite(total_nll):
        raise ValueError('Need finite NLL and positive target count')
    mean = total_nll / targets
    return dict(total_nll=total_nll, targets=targets,
                mean_token_nll=mean, perplexity=math.exp(mean))


def test_nll():
    probs = torch.tensor([[[.1, .7, .2], [.6, .3, .1], [.2, .2, .6]]], dtype=torch.float64)
    ids = torch.tensor([[2, 1, 0]])
    logits = probs.log()
    before = logits.clone()
    loss, count = shifted_token_nll(logits, ids)
    assert isinstance(loss, torch.Tensor) and loss.ndim == 0
    assert isinstance(count, int) and count == 2
    assert math.isclose(loss.item(), -math.log(.7) - math.log(.6), abs_tol=1e-10)
    assert torch.equal(logits, before) and torch.equal(ids, torch.tensor([[2, 1, 0]]))
    # Last-position logits cannot affect within-block NLL.
    changed = logits.clone()
    changed[:, -1, :] = torch.tensor([90., -90., 0.])
    other, n = shifted_token_nll(changed, ids)
    assert n == count and torch.allclose(loss, other)
    # A different first token cannot affect targets for these fixed toy logits.
    changed_ids = ids.clone()
    changed_ids[:, 0] = 0
    other, n = shifted_token_nll(logits, changed_ids)
    assert n == count and torch.allclose(loss, other)
    batch_loss, batch_n = shifted_token_nll(logits.repeat(2, 1, 1), ids.repeat(2, 1))
    assert batch_n == 4 and torch.allclose(batch_loss, 2 * loss)
    short_loss, short_n = shifted_token_nll(logits[:, :2], ids[:, :2])
    assert short_n == 1 and math.isclose(short_loss.item(), -math.log(.7), abs_tol=1e-10)
    combined = summarize(loss.item() + short_loss.item(), count + short_n)
    assert combined['targets'] == 3
    assert math.isclose(combined['mean_token_nll'], (-2 * math.log(.7) - math.log(.6)) / 3)
    assert math.isclose(summarize(130., 110)['mean_token_nll'], 130 / 110)
    print('shifted_token_nll tests: PASS')


def test_hook():
    module = torch.nn.Linear(3, 2, bias=False)
    x = torch.tensor([[[2., -.26, .26]]])
    before, weights = x.clone(), module.weight.detach().clone()
    marker = object()
    replacement = quantize_input_hook(module, (x, marker), alpha=.7)
    assert isinstance(replacement, tuple) and len(replacement) == 2
    assert replacement[1] is marker
    assert replacement[0].shape == x.shape and replacement[0].dtype == x.dtype
    assert replacement[0].device == x.device
    assert torch.allclose(replacement[0], torch.tensor([[[.7, -.3, .3]]]), atol=1e-6)
    assert torch.isfinite(replacement[0]).all()
    assert torch.equal(x, before)
    handle = module.register_forward_pre_hook(functools.partial(quantize_input_hook, alpha=.7))
    try:
        assert torch.allclose(module(x), F.linear(fake_quantize(x, .7), weights))
        # Different live inputs must produce different replacements.
        x2 = -x
        assert torch.allclose(module(x2), F.linear(fake_quantize(x2, .7), weights))
    finally:
        handle.remove()
    assert not module._forward_pre_hooks
    assert torch.equal(module.weight, weights) and torch.equal(x, before)
    assert torch.allclose(module(x), F.linear(x, weights))
    # Injection cleanup on an exception.
    handle = module.register_forward_pre_hook(functools.partial(quantize_input_hook, alpha=.7))
    try:
        try:
            raise RuntimeError('intentional cleanup test')
        finally:
            handle.remove()
    except RuntimeError:
        pass
    assert not module._forward_pre_hooks
    print('quantize_input_hook tests: PASS')


def state_digest(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((tuple(value.shape), str(value.dtype))).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def run_smoke():
    from transformers import AutoModelForCausalLM
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out = ROOT / 'results'
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    metadata = dict(run_id=run_id, model=MODEL, revision=REVISION,
                    python=platform.python_version(), torch=str(torch.__version__),
                    threads=torch.get_num_threads(), dtype='float32', attention='eager',
                    use_cache=False, calibration_subset=0, calibration_tokens=256,
                    scope='diagnostic fake quantization; NOT actual low-bit execution',
                    test_accessed=False)
    model = None
    digest_before = None
    try:
        raw = (D05 / 'data' / 'wikitext_blocks.json').read_bytes()
        assert hashlib.sha256(raw).hexdigest() == MANIFEST_SHA
        manifest = json.loads(raw)
        assert manifest['model'] == MODEL and manifest['tokenizer_revision'] == REVISION
        assert manifest['dataset_revision'] == 'b08601e04326c79dfdd32d625aee71d232d685c3'
        assert manifest['sequence_length'] == 128 and not manifest['test_accessed']
        blocks = manifest['development']
        assert len(blocks) == 16 and all(len(b['token_ids']) == 128 for b in blocks)
        snapshot = torch.load(D05 / 'data' / 'wikitext_activations.pt', weights_only=True)
        assert snapshot['metadata']['manifest_sha256'] == MANIFEST_SHA
        assert snapshot['metadata']['revision'] == REVISION
        thresholds = {rule: {} for rule in ('max_abs', 'p999')}
        for name in LAYERS:
            x = snapshot['inputs']['calibration_0'][name]
            assert x.shape == (256, 1536) and torch.isfinite(x).all()
            thresholds['max_abs'][name] = x.abs().max().item()
            thresholds['p999'][name] = torch.quantile(x.abs().flatten(), .999).item()
        assert all(math.isfinite(a) and a > 0 for d in thresholds.values() for a in d.values())
        metadata.update(thresholds=thresholds, manifest_sha256=MANIFEST_SHA)
        del snapshot
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required by this fixed smoke configuration')
        metadata['gpu'] = torch.cuda.get_device_name(0)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL, revision=REVISION, local_files_only=True,
            torch_dtype=torch.float32, attn_implementation='eager').to('cuda').eval()
        assert model.config._commit_hash == REVISION
        digest_before = state_digest(model)
        ids_list = [torch.tensor([b['token_ids']], device='cuda', dtype=torch.long) for b in blocks]

        def evaluate(label, rule=None, sites=()):
            handles = []
            total, targets = 0., 0
            try:
                for name in sites:
                    handles.append(model.get_submodule(name).register_forward_pre_hook(
                        functools.partial(quantize_input_hook, alpha=thresholds[rule][name])))
                with torch.inference_mode():
                    for ids in ids_list:
                        logits = model(input_ids=ids, use_cache=False).logits
                        loss, n = shifted_token_nll(logits, ids)
                        total += loss.item()
                        targets += n
                assert targets == 2032
                row = dict(run_id=run_id, variant=label, rule=rule or 'none',
                           site_count=len(sites), **summarize(total, targets))
                return row
            finally:
                for handle in handles:
                    handle.remove()
                assert all(not model.get_submodule(name)._forward_pre_hooks for name in LAYERS)

        rows.append(evaluate('fp'))
        for rule in ('max_abs', 'p999'):
            for sites in (LAYERS[:1], LAYERS):
                rows.append(evaluate(f'{rule}_{len(sites)}_sites', rule, sites))
        rows.append(evaluate('fp_after_hooks_removed'))
        assert math.isclose(rows[0]['total_nll'], rows[-1]['total_nll'], rel_tol=1e-6, abs_tol=1e-5)
        for row in rows:
            row['delta_mean_nll_vs_fp'] = row['mean_token_nll'] - rows[0]['mean_token_nll']
        metadata['status'] = 'ok'
    except Exception:
        metadata['status'] = 'failed'
        metadata['traceback'] = traceback.format_exc()
        raise
    finally:
        try:
            if model is not None and digest_before is not None:
                after = state_digest(model)
                metadata.update(state_sha256_before=digest_before, state_sha256_after=after,
                                original_state_unchanged=(digest_before == after))
                if digest_before != after:
                    metadata['status'] = 'failed'
                    raise AssertionError('Original model state changed')
        finally:
            metadata_path = out / f'end_to_end_smoke_{run_id}.json'
            metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
            if rows:
                # Completed rows are saved even if a later variant/check fails.
                fields = ['run_id', 'variant', 'rule', 'site_count', 'total_nll', 'targets',
                          'mean_token_nll', 'perplexity', 'delta_mean_nll_vs_fp']
                path = out / f'end_to_end_smoke_{run_id}.csv'
                with path.open('x', newline='', encoding='utf-8') as stream:
                    writer = csv.DictWriter(stream, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(rows)
                print('Saved:', path)
            print('Saved:', metadata_path, 'status:', metadata.get('status'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', choices=('nll', 'hook', 'all'))
    args = parser.parse_args()
    if args.test in (None, 'all', 'nll'):
        test_nll()
    if args.test in (None, 'all', 'hook'):
        test_hook()
    if args.test is None:
        run_smoke()


if __name__ == '__main__':
    main()
