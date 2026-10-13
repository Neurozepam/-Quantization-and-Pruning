"""Frozen D07 pilot: local output-MSE versus final logit-MSE threshold selection.

Run from VS Code or the repository root. --tests-only never loads the model.
"""
import argparse
import csv
import functools
import hashlib
import json
import math
import platform
import statistics
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT.parent
sys.path.insert(0, str(RESEARCH / 'D06' / 'src'))
from end_to_end_smoke import (LAYERS, MANIFEST_SHA, MODEL, REVISION,
                              quantize_input_hook, shifted_token_nll, state_digest)
from local_controls import errors
from downstream_metrics import logit_squared_error

REFERENCE_RUN = '20261012T131431525795Z'
TIMING_RUN = '20261013T133122714546Z'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def choose(rows, metric):
    return min(rows, key=lambda r: (r[metric], r['alpha']))


def forward_blocks(model, ids_list, reference=None, site=None, alpha=None,
                   check_budget=lambda: None):
    """One live hook, removed even on failure; FP logits cached on CPU."""
    handle = None
    cached, total_nll, targets, sse, elements = [], 0., 0, 0., 0
    try:
        if site is not None:
            handle = model.get_submodule(site).register_forward_pre_hook(
                functools.partial(quantize_input_hook, alpha=alpha))
        with torch.inference_mode():
            for i, ids in enumerate(ids_list):
                check_budget()
                logits = model(input_ids=ids, use_cache=False).logits
                loss, n = shifted_token_nll(logits, ids)
                total_nll += loss.item()
                targets += n
                if reference is None:
                    cached.append(logits.cpu())
                else:
                    error, count = logit_squared_error(reference[i].to(logits.device), logits)
                    sse += error
                    elements += count
        assert math.isfinite(total_nll) and math.isfinite(sse)
        result = dict(total_nll=total_nll, targets=targets,
                      mean_token_nll=total_nll / targets,
                      logit_mse=sse / elements if elements else 0.)
        return cached, result
    finally:
        if handle is not None:
            handle.remove()


def screening(pairs):
    deltas = [r['delta_mean_nll'] for r in pairs]
    by_site = {site: [r['delta_mean_nll'] for r in pairs if r['layer'] == site] for site in LAYERS}
    mean = sum(deltas) / len(deltas)
    site_means = {site: sum(v) / len(v) for site, v in by_site.items()}
    wins = sum(d < -1e-5 for d in deltas)
    go = len(pairs) == 24 and mean <= -.005 and wins >= 16 and max(site_means.values()) <= .005 and max(deltas) <= .02
    site_summary = {site: dict(mean=site_means[site], median=statistics.median(v),
                               minimum=min(v), maximum=max(v),
                               wins=sum(d < -1e-5 for d in v),
                               losses=sum(d > 1e-5 for d in v),
                               ties=sum(abs(d) <= 1e-5 for d in v))
                    for site, v in by_site.items()}
    return dict(mean_delta_nll=mean, wins=wins,
                losses=sum(d > 1e-5 for d in deltas),
                ties=sum(abs(d) <= 1e-5 for d in deltas),
                per_site_mean_delta=site_means, per_site_summary=site_summary,
                worst_pair_delta=max(deltas),
                quality_decision='GO' if go else 'NO-GO')


def run():
    from transformers import AutoModelForCausalLM

    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    destination = ROOT / 'results'
    destination.mkdir(exist_ok=True)
    prefix = destination / f'pilot_{run_id}'
    protocol = ROOT / 'pilot_protocol.md'
    manifest_path = RESEARCH / 'D05' / 'data' / 'wikitext_blocks.json'
    snapshot_path = RESEARCH / 'D05' / 'data' / 'wikitext_activations.pt'
    candidates_path = RESEARCH / 'D06' / 'results' / f'local_candidate_losses_{REFERENCE_RUN}.csv'
    controls_path = RESEARCH / 'D06' / 'results' / f'local_controls_{REFERENCE_RUN}.csv'
    timing = json.loads((destination / f'timing_check_{TIMING_RUN}.json').read_text(encoding='utf-8'))
    meta = dict(run_id=run_id, status='started', protocol_sha256=digest(protocol),
                model=MODEL, revision=REVISION, python=platform.python_version(),
                torch=str(torch.__version__), threads=torch.get_num_threads(),
                dtype='float32', attention='eager', use_cache=False,
                calibration_tokens=256, development_targets=2032,
                timing_run=TIMING_RUN, candidate_source_run=REFERENCE_RUN,
                test_accessed=False, scope='one-site activation fake quantization; NOT low-bit inference',
                calibration_search_seconds={}, development_evaluation_seconds={},
                selection_seconds={}, reference_seconds={})
    model, before = None, None
    budget_start, load_seconds = time.perf_counter(), 0.
    pairs, selected = [], []

    def check_budget():
        if time.perf_counter() - budget_start - load_seconds >= 7200:
            raise TimeoutError('120-minute pilot budget exceeded')
        if model is not None and torch.cuda.max_memory_reserved() > .9 * props.total_memory:
            raise MemoryError('Pilot exceeded frozen GPU memory cap')

    def ids_for(blocks):
        assert all(len(b['token_ids']) == 128 for b in blocks)
        return [torch.tensor([b['token_ids']], device='cuda', dtype=torch.long) for b in blocks]

    try:
        assert 'Параметры выбраны до запуска.' in protocol.read_text(encoding='utf-8')
        assert timing['status'] == 'ok' and timing['gate_decision'] == 'GO'
        meta['timing_protocol_sha256'] = timing['protocol_sha256']
        assert digest(manifest_path) == MANIFEST_SHA
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        assert manifest['model'] == MODEL and manifest['tokenizer_revision'] == REVISION
        assert manifest['dataset_revision'] == 'b08601e04326c79dfdd32d625aee71d232d685c3'
        assert not manifest['test_accessed'] and manifest['sequence_length'] == 128
        assert len(manifest['calibration']) == 8
        groups = {}
        with candidates_path.open(newline='', encoding='utf-8') as stream:
            for r in csv.DictReader(stream):
                groups.setdefault((int(r['subset_id']), r['layer']), []).append(
                    dict(alpha=float(r['alpha']), output_mse=float(r['output_mse'])))
        assert set(groups) == {(s, site) for s in range(8) for site in LAYERS}
        for rows in groups.values():
            rows.sort(key=lambda r: r['alpha'])
            assert len(rows) == len({r['alpha'] for r in rows}) == 203
            assert all(math.isfinite(v) and v >= 0 for r in rows for v in r.values())
            assert all(r['alpha'] > 0 for r in rows)
        historical_cost, local_winners = 0., {}
        with controls_path.open(newline='', encoding='utf-8') as stream:
            for r in csv.DictReader(stream):
                if r['method'] == 'cal_output_mse':
                    local_winners[int(r['subset_id']), r['layer']] = float(r['alpha'])
                    historical_cost += float(r['threshold_seconds']) + float(r['shared_search_seconds'])
        assert set(local_winners) == set(groups)
        meta.update(candidate_csv_sha256=digest(candidates_path), controls_csv_sha256=digest(controls_path),
                    reused_D06_local_preparation_seconds=historical_cost,
                    local_cost_note='Cached D06 losses reused; historical cost reported, not counted as work repeated in this run',
                    candidates={f'{s}:{site}': [r['alpha'] for r in rows] for (s, site), rows in groups.items()},
                    snapshot_sha256_before=digest(snapshot_path))
        assert meta['candidate_csv_sha256'] == timing['candidate_csv_sha256']
        assert meta['controls_csv_sha256'] == timing['controls_csv_sha256']
        snapshot = torch.load(snapshot_path, weights_only=True)
        assert snapshot['metadata']['manifest_sha256'] == MANIFEST_SHA
        assert snapshot['metadata']['revision'] == REVISION
        if not torch.cuda.is_available():
            raise RuntimeError('Frozen pilot requires CUDA')
        props = torch.cuda.get_device_properties(0)
        meta.update(gpu=props.name, device_capacity_bytes=props.total_memory,
                    actual_preparation_seconds=time.perf_counter() - budget_start)
        start = time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(
            MODEL, revision=REVISION, local_files_only=True,
            torch_dtype=torch.float32, attn_implementation='eager').to('cuda').eval()
        torch.cuda.synchronize()
        load_seconds = time.perf_counter() - start
        meta['model_load_seconds'] = load_seconds
        assert model.config._commit_hash == REVISION
        assert all(torch.equal(model.get_submodule(site).weight.cpu(), snapshot['weights'][site]) for site in LAYERS)
        before = state_digest(model)
        torch.cuda.reset_peak_memory_stats()
        with Path(f'{prefix}_candidates.csv').open('x', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=['subset_id', 'layer', 'alpha', 'local_output_mse', 'logit_mse'])
            writer.writeheader()
            for subset in sorted(manifest['calibration'], key=lambda s: s['subset_id']):
                sid = subset['subset_id']
                assert len(subset['blocks']) == 2 and all(b['split'] == 'train' for b in subset['blocks'])
                ids_list = ids_for(subset['blocks'])
                start = time.perf_counter()
                reference, fp = forward_blocks(model, ids_list, check_budget=check_budget)
                assert fp['targets'] == 254
                meta['reference_seconds'][f'calibration_{sid}'] = time.perf_counter() - start
                for site in LAYERS:
                    start = time.perf_counter()
                    rows = []
                    for r in groups[sid, site]:
                        _, metrics = forward_blocks(model, ids_list, reference, site, r['alpha'], check_budget)
                        assert metrics['targets'] == 254
                        row = dict(subset_id=sid, layer=site, alpha=r['alpha'],
                                   local_output_mse=r['output_mse'], logit_mse=metrics['logit_mse'])
                        rows.append(row)
                        writer.writerow(row)
                        stream.flush()
                    meta['calibration_search_seconds'][f'{sid}:{site}'] = time.perf_counter() - start
                    selection_start = time.perf_counter()
                    local, downstream = choose(rows, 'local_output_mse'), choose(rows, 'logit_mse')
                    meta['selection_seconds'][f'{sid}:{site}'] = time.perf_counter() - selection_start
                    assert local['alpha'] == local_winners[sid, site]
                    selected.append(dict(subset_id=sid, layer=site, local_alpha=local['alpha'],
                                         downstream_alpha=downstream['alpha'],
                                         local_cal_logit_mse=local['logit_mse'], downstream_cal_logit_mse=downstream['logit_mse'],
                                         local_cal_output_mse=local['local_output_mse'], downstream_cal_output_mse=downstream['local_output_mse']))
                    Path(f'{prefix}_selections.json').write_text(json.dumps(selected, indent=2), encoding='utf-8')
                    print(f'Calibration complete: subset {sid}, {site}', flush=True)
                del reference
        assert len(selected) == 24
        check_budget()
        dev_blocks = manifest['development']
        assert len(dev_blocks) == 16 and all(b['split'] == 'validation' for b in dev_blocks)
        ids_list = ids_for(dev_blocks)
        start = time.perf_counter()
        reference, fp = forward_blocks(model, ids_list, check_budget=check_budget)
        assert fp['targets'] == 2032
        meta['reference_seconds']['development'] = time.perf_counter() - start
        meta['development_fp'] = fp
        for selection in selected:
            sid, site = selection['subset_id'], selection['layer']
            start = time.perf_counter()
            results = {}
            for alpha in set((selection['local_alpha'], selection['downstream_alpha'])):
                _, metrics = forward_blocks(model, ids_list, reference, site, alpha, check_budget)
                assert metrics['targets'] == 2032
                with torch.inference_mode():
                    metrics['local_output_mse'] = errors(snapshot['inputs']['development'][site], snapshot['weights'][site], alpha)['output_mse']
                results[alpha] = metrics
            local, downstream = results[selection['local_alpha']], results[selection['downstream_alpha']]
            pair = dict(**selection, targets=2032, local_mean_nll=local['mean_token_nll'],
                        downstream_mean_nll=downstream['mean_token_nll'],
                        delta_mean_nll=downstream['mean_token_nll'] - local['mean_token_nll'],
                        local_ppl=math.exp(local['mean_token_nll']), downstream_ppl=math.exp(downstream['mean_token_nll']),
                        local_dev_logit_mse=local['logit_mse'], downstream_dev_logit_mse=downstream['logit_mse'],
                        local_dev_output_mse=local['local_output_mse'], downstream_dev_output_mse=downstream['local_output_mse'])
            pairs.append(pair)
            path = Path(f'{prefix}_development.csv')
            with path.open('a' if len(pairs) > 1 else 'x', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(pair))
                if len(pairs) == 1:
                    writer.writeheader()
                writer.writerow(pair)
            meta['development_evaluation_seconds'][f'{sid}:{site}'] = time.perf_counter() - start
            print(f'Development complete: subset {sid}, {site}', flush=True)
        start = time.perf_counter()
        _, fp_after = forward_blocks(model, ids_list, check_budget=check_budget)
        meta['reference_seconds']['development_after'] = time.perf_counter() - start
        assert math.isclose(fp['total_nll'], fp_after['total_nll'], rel_tol=1e-6, abs_tol=1e-5)
        assert all(not model.get_submodule(site)._forward_pre_hooks for site in LAYERS)
        check_budget()
        meta.update(status='ok', screening=screening(pairs), fp_after=fp_after)
    except Exception as exc:
        meta.update(status='budget_stopped' if isinstance(exc, TimeoutError) else 'failed',
                    screening={'quality_decision': 'INCONCLUSIVE'}, traceback=traceback.format_exc())
        raise
    finally:
        meta.update(elapsed_seconds_excluding_load=time.perf_counter() - budget_start - load_seconds,
                    completed_selections=len(selected), completed_development_pairs=len(pairs))
        if model is not None and before is not None:
            after = state_digest(model)
            meta.update(state_sha256_before=before, state_sha256_after=after,
                        original_state_unchanged=before == after,
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                        peak_reserved_bytes=torch.cuda.max_memory_reserved())
            if before != after or meta['peak_reserved_bytes'] > .9 * props.total_memory:
                meta.update(status='failed', screening={'quality_decision': 'INCONCLUSIVE'},
                            check_error='Model state changed or memory cap exceeded')
        if 'snapshot_sha256_before' in meta:
            meta['snapshot_unchanged'] = digest(snapshot_path) == meta['snapshot_sha256_before']
            if not meta['snapshot_unchanged']:
                meta.update(status='failed', screening={'quality_decision': 'INCONCLUSIVE'})
        with Path(f'{prefix}.json').open('x', encoding='utf-8') as stream:
            json.dump(meta, stream, ensure_ascii=False, indent=2)
        print(f'Results: {prefix}*', flush=True)
        if meta['status'] == 'ok':
            print(json.dumps(meta['screening'], ensure_ascii=False, indent=2), flush=True)
        elif meta.get('check_error') or meta.get('snapshot_unchanged') is False:
            raise RuntimeError('Final integrity/resource checks failed; inspect metadata')


def tests():
    rows = [dict(alpha=2., mse=1.), dict(alpha=1., mse=1.)]
    assert choose(rows, 'mse')['alpha'] == 1.

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.site = torch.nn.Linear(1, 2, bias=False)
        def forward(self, input_ids, use_cache=False):
            return SimpleNamespace(logits=self.site(input_ids.float().unsqueeze(-1)))

    model = TinyModel().eval()
    ids = [torch.tensor([[0, 1, 0]])]
    before = state_digest(model)
    reference, fp = forward_blocks(model, ids)
    _, quant = forward_blocks(model, ids, reference, 'site', .7)
    assert fp['targets'] == quant['targets'] == 2 and quant['logit_mse'] >= 0
    assert not model.site._forward_pre_hooks and state_digest(model) == before
    try:
        forward_blocks(model, [torch.tensor([[999, 999, 999]])], reference, 'site', .7)
    except (IndexError, RuntimeError):
        pass
    else:
        raise AssertionError('Expected invalid target error')
    assert not model.site._forward_pre_hooks
    pairs = [dict(layer=site, delta_mean_nll=-.01) for site in LAYERS for _ in range(8)]
    assert screening(pairs)['quality_decision'] == 'GO'
    pairs[0]['delta_mean_nll'] = .03
    assert screening(pairs)['quality_decision'] == 'NO-GO'
    print('PASS: selector ties, CPU evaluation, hook cleanup, unchanged weights, screening')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tests-only', action='store_true')
    args = parser.parse_args()
    tests()
    if not args.tests_only:
        run()
