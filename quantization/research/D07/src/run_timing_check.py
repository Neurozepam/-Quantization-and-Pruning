"""Calibration-only timing gate. No development/test evaluation or threshold search."""
import csv
import functools
import hashlib
import json
import math
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT.parent
sys.path.insert(0, str(RESEARCH / 'D06' / 'src'))
from end_to_end_smoke import (LAYERS, MANIFEST_SHA, MODEL, REVISION,
                              quantize_input_hook, shifted_token_nll, state_digest)
from downstream_metrics import logit_squared_error
from timing_check import benchmark_cuda


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run():
    from transformers import AutoModelForCausalLM

    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = ROOT / 'results' / f'timing_check_{run_id}.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    protocol = ROOT / 'pilot_protocol.md'
    manifest_path = RESEARCH / 'D05' / 'data' / 'wikitext_blocks.json'
    candidates_path = RESEARCH / 'D06' / 'results' / 'local_candidate_losses_20261012T131431525795Z.csv'
    controls_path = RESEARCH / 'D06' / 'results' / 'local_controls_20261012T131431525795Z.csv'
    metadata = dict(run_id=run_id, status='started', protocol_sha256=digest(protocol),
                    scope='calibration-only timing gate; NOT a quality experiment',
                    development_evaluated=False, test_accessed=False,
                    model=MODEL, revision=REVISION, python=platform.python_version(),
                    torch=str(torch.__version__), threads=torch.get_num_threads(),
                    calibration_subset=0, calibration_tokens=256,
                    dtype='float32', attention='eager', use_cache=False,
                    reference_cache='two CPU logit tensors; copied to GPU inside measured calls')
    model = None
    before = None
    handles = []
    try:
        if 'Параметры выбраны до запуска.' not in protocol.read_text(encoding='utf-8'):
            raise RuntimeError('Experiment protocol must be specified before running')
        assert digest(manifest_path) == MANIFEST_SHA
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        assert manifest['model'] == MODEL
        assert manifest['tokenizer_revision'] == REVISION
        assert manifest['dataset_revision'] == 'b08601e04326c79dfdd32d625aee71d232d685c3'
        assert not manifest['test_accessed']
        subset = next(s for s in manifest['calibration'] if s['subset_id'] == 0)
        blocks = subset['blocks']
        assert len(blocks) == 2
        assert all(b['split'] == 'train' and len(b['token_ids']) == 128 for b in blocks)
        del subset, manifest

        prep_start = time.perf_counter()
        with candidates_path.open(newline='', encoding='utf-8') as stream:
            candidates = list(csv.DictReader(stream))
        groups = {}
        for row in candidates:
            key = int(row['subset_id']), row['layer']
            groups.setdefault(key, []).append(float(row['alpha']))
        assert set(groups) == {(s, site) for s in range(8) for site in LAYERS}
        assert all(len(values) == len(set(values)) == 203 for values in groups.values())
        assert all(math.isfinite(a) and a > 0 for values in groups.values() for a in values)
        p999 = {}
        for row in candidates:
            if int(row['subset_id']) == 0 and row['is_p999'] == 'True':
                assert row['layer'] not in p999
                p999[row['layer']] = float(row['alpha'])
        assert set(p999) == set(LAYERS)
        costs = {}
        with controls_path.open(newline='', encoding='utf-8') as stream:
            for row in csv.DictReader(stream):
                if row['method'] == 'cal_output_mse':
                    costs[int(row['subset_id']), row['layer']] = (
                        float(row['threshold_seconds']) + float(row['shared_search_seconds']))
        assert set(costs) == set(groups)
        preparation_read_seconds = time.perf_counter() - prep_start
        preparation_seconds = preparation_read_seconds + sum(costs.values())
        metadata.update(candidate_csv_sha256=digest(candidates_path),
                        controls_csv_sha256=digest(controls_path),
                        candidate_counts={f'{s}:{site}': len(v) for (s, site), v in groups.items()},
                        timing_alphas=p999, preparation_read_seconds=preparation_read_seconds,
                        historical_local_search_and_threshold_seconds=sum(costs.values()),
                        preparation_estimate_seconds=preparation_seconds,
                        preparation_cost_note='D06 measured shared CPU search + threshold time, charged once per pair; cached results reused')
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required by frozen protocol')
        props = torch.cuda.get_device_properties(0)
        metadata.update(gpu=props.name, device_capacity_bytes=props.total_memory)
        load_start = time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(
            MODEL, revision=REVISION, local_files_only=True,
            torch_dtype=torch.float32, attn_implementation='eager').to('cuda').eval()
        torch.cuda.synchronize()
        metadata['model_load_seconds'] = time.perf_counter() - load_start
        assert model.config._commit_hash == REVISION
        before = state_digest(model)
        ids_list = [torch.tensor([b['token_ids']], device='cuda', dtype=torch.long) for b in blocks]
        reference = []
        fp_totals = []
        compute_start = time.perf_counter()

        def check_deadline():
            if time.perf_counter() - compute_start >= 600:
                raise TimeoutError('Timing gate exceeded 10-minute compute limit')

        def reference_call():
            check_deadline()
            reference.clear()
            total_nll, targets = 0., 0
            with torch.inference_mode():
                for ids in ids_list:
                    check_deadline()
                    logits = model(input_ids=ids, use_cache=False).logits
                    nll, n = shifted_token_nll(logits, ids)
                    total_nll += nll.item()
                    targets += n
                    reference.append(logits.cpu())
            assert targets == 254 and math.isfinite(total_nll)
            fp_totals.append(total_nll)
            return None

        reports = {'fp_reference': benchmark_cuda(reference_call)}

        def quantized_call():
            check_deadline()
            total_nll, targets, total_sse = 0., 0, 0.
            with torch.inference_mode():
                for ids, fp_cpu in zip(ids_list, reference):
                    check_deadline()
                    logits = model(input_ids=ids, use_cache=False).logits
                    nll, n = shifted_token_nll(logits, ids)
                    sse, count = logit_squared_error(fp_cpu.to(logits.device), logits)
                    assert count == 127 * logits.shape[-1]
                    total_nll += nll.item()
                    targets += n
                    total_sse += sse
            assert targets == 254 and math.isfinite(total_nll) and math.isfinite(total_sse)
            return None

        for site in LAYERS:
            check_deadline()
            try:
                handles.append(model.get_submodule(site).register_forward_pre_hook(
                    functools.partial(quantize_input_hook, alpha=p999[site])))
                reports[site] = benchmark_cuda(quantized_call)
            finally:
                for handle in handles:
                    handle.remove()
                handles.clear()
        reference_call()
        assert all(math.isclose(fp_totals[0], nll, rel_tol=1e-6, abs_tol=1e-5) for nll in fp_totals)
        assert all(not model.get_submodule(site)._forward_pre_hooks for site in LAYERS)
        seconds_per_block = max(t for report in reports.values() for t in report['seconds']) / 2
        counts = dict(calibration_quantized=2 * sum(len(v) for v in groups.values()),
                      development_quantized=24 * 2 * 16,
                      fp_calibration_reference=8 * 2,
                      fp_development_reference=16,
                      fp_development_after_check=16)
        predicted = 1.5 * seconds_per_block * sum(counts.values()) + preparation_seconds
        peak_reserved = max(r['peak_reserved_bytes'] for r in reports.values())
        metadata.update(reports=reports, block_pass_counts=counts,
                        seconds_per_block_conservative=seconds_per_block,
                        safety_factor=1.5, predicted_pilot_seconds=predicted,
                        predicted_pilot_minutes=predicted / 60,
                        time_gate_passed=predicted <= 7200,
                        memory_gate_passed=peak_reserved <= .9 * props.total_memory,
                        timing_compute_seconds=time.perf_counter() - compute_start,
                        gate_decision='GO' if predicted <= 7200 and peak_reserved <= .9 * props.total_memory else 'STOP',
                        status='ok', candidate_quality_results_recorded=False)
    except Exception:
        metadata.update(status='failed', gate_decision='STOP', traceback=traceback.format_exc())
        raise
    finally:
        for handle in handles:
            handle.remove()
        if model is not None and before is not None:
            after = state_digest(model)
            metadata.update(state_sha256_before=before, state_sha256_after=after,
                            original_state_unchanged=before == after)
            if before != after:
                metadata.update(status='failed', gate_decision='STOP', error='Model state changed')
        with output.open('x', encoding='utf-8') as stream:
            json.dump(metadata, stream, ensure_ascii=False, indent=2)
        print(output)
        if metadata['status'] == 'ok':
            print(f"{metadata['gate_decision']}: forecast {metadata['predicted_pilot_minutes']:.2f} minutes")
        if metadata.get('original_state_unchanged') is False:
            raise AssertionError('Model state changed')


if __name__ == '__main__':
    run()
