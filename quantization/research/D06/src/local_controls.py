"""Compare static clipping rules on cached FP calibration/development inputs.

Default: subset 0. --all-subsets evaluates all eight saved subsets;
--refine-grid adds geometric midpoints while preserving original candidates.
--tests-only checks selection and nested-grid construction without loading data.
"""
import argparse
import csv
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
D05 = ROOT.parent / 'D05'
D04 = ROOT.parents[1] / 'learning' / 'D04'
sys.path.insert(0, str(D04))
from day4_experiment import fake_quantize, measure_errors

MANIFEST_SHA = 'ee5d038f8c1c37a45624ce2df92e2090277c28c9ab5fb823b755e395210d59b4'
REVISION = '93efa2f097d58c2a74874c7e644dbc9b0cee75a2'
LAYERS = [f'model.layers.{i}.mlp.down_proj' for i in (0, 15, 29)]


def choose_candidate(calibration_rows, objective):
    """Select the minimum calibration objective, breaking ties by alpha."""
    calibration_rows = sorted(calibration_rows, key=lambda row: (row[objective], row['alpha']))
    return calibration_rows[0]


def test_choose_candidate():
    rows = [dict(alpha=1., activation_mse=.1, output_mse=.3),
            dict(alpha=2., activation_mse=.2, output_mse=.1),
            dict(alpha=3., activation_mse=.4, output_mse=.1)]
    before = json.dumps(rows, sort_keys=True)
    assert choose_candidate(rows, 'activation_mse') == rows[0]
    assert choose_candidate(rows, 'output_mse') == rows[1], 'Keep lowest alpha in a tie'
    assert choose_candidate(rows[:1], 'output_mse') == rows[0]
    assert json.dumps(rows, sort_keys=True) == before, 'Input rows changed'
    # Objective, not alpha magnitude, determines the winner.
    other = [dict(alpha=1., activation_mse=4., output_mse=5.),
             dict(alpha=2., activation_mse=2., output_mse=1.)]
    assert choose_candidate(other, 'activation_mse') == other[1]
    assert choose_candidate(other, 'output_mse') == other[1]
    print('choose_candidate tests: PASS')


def candidate_grid(refine=False):
    """Nested grid: preserve exact original floats; add geometric midpoints."""
    coarse = torch.logspace(-2, 3, 101, dtype=torch.float32).tolist()
    if not refine:
        return coarse
    midpoints = [math.sqrt(a * b) for a, b in zip(coarse[:-1], coarse[1:])]
    refined = sorted(set(coarse + midpoints))
    assert len(refined) == 201 and set(coarse).issubset(refined)
    return refined


def test_candidate_grid():
    coarse, refined = candidate_grid(), candidate_grid(True)
    assert len(coarse) == 101 and len(refined) == 201
    assert set(coarse).issubset(refined)
    assert coarse[0] == refined[0] and coarse[-1] == refined[-1]
    assert all(a < b for a, b in zip(refined[:-1], refined[1:]))
    print('Nested candidate grid tests: PASS')


def file_digest(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def errors(x, w, alpha):
    return measure_errors(x, fake_quantize(x, alpha), w, alpha)


def normalized_output_mse(output_mse, power):
    if not math.isfinite(power) or power <= 0:
        raise ValueError('Need finite positive FP output power')
    return output_mse / power


def run(subset_ids, refine_grid=False):
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    destination = ROOT / 'results'
    destination.mkdir(parents=True, exist_ok=True)
    results, searches = [], []
    path = D05 / 'data' / 'wikitext_activations.pt'
    before = file_digest(path)
    metadata = dict(run_id=run_id, python=platform.python_version(),
                    torch=str(torch.__version__), threads=torch.get_num_threads(),
                    subset_ids=subset_ids, layers=LAYERS, calibration_tokens_per_run=256,
                    development_tokens_per_site=2048, bit_codes='-7..7', dtype='float32',
                    base_grid='101 original logspace points + 100 geometric midpoints' if refine_grid else 'torch.logspace(-2,3,101,float32)',
                    refine_grid=refine_grid,
                    candidate_additions='this subset max-abs and p99.9',
                    scope='local cached-FP-input controls; not full forward', test_accessed=False,
                    snapshot_sha256_before=before)
    try:
        manifest = D05 / 'data' / 'wikitext_blocks.json'
        assert file_digest(manifest) == MANIFEST_SHA
        snapshot = torch.load(path, weights_only=True)
        assert snapshot['metadata']['manifest_sha256'] == MANIFEST_SHA
        assert snapshot['metadata']['revision'] == REVISION
        originals = {name: w.clone() for name, w in snapshot['weights'].items()}
        with torch.inference_mode():
            for subset_id in subset_ids:
                for layer in LAYERS:
                    x = snapshot['inputs'][f'calibration_{subset_id}'][layer]
                    dev = snapshot['inputs']['development'][layer]
                    w = snapshot['weights'][layer]
                    assert x.shape == (256, 1536) and dev.shape == (2048, 1536)
                    assert w.shape == (576, 1536)
                    assert all(torch.isfinite(t).all() for t in (x, dev, w))
                    x_before, dev_before = x.clone(), dev.clone()
                    start = time.perf_counter()
                    max_abs = x.abs().max().item()
                    percentile = torch.quantile(x.abs().flatten(), .999).item()
                    threshold_seconds = time.perf_counter() - start
                    grid = candidate_grid(refine_grid)
                    alphas = sorted(set(grid + [max_abs, percentile]))
                    assert all(math.isfinite(a) and a > 0 for a in alphas)
                    start = time.perf_counter()
                    candidates = [dict(alpha=a, **errors(x, w, a)) for a in alphas]
                    for row in candidates:
                        assert all(math.isfinite(v) for v in row.values())
                    act_best = choose_candidate(candidates, 'activation_mse')
                    out_best = choose_candidate(candidates, 'output_mse')
                    search_seconds = time.perf_counter() - start
                    for row in candidates:
                        searches.append(dict(run_id=run_id, subset_id=subset_id, layer=layer,
                                             is_base_grid=row['alpha'] in grid,
                                             is_original_grid=row['alpha'] in candidate_grid(),
                                             is_p999=row['alpha'] == percentile,
                                             is_max_abs=row['alpha'] == max_abs, **row))
                    selections = [('max_abs', max_abs), ('p999', percentile),
                                  ('cal_activation_mse', act_best['alpha']),
                                  ('cal_output_mse', out_best['alpha'])]
                    cal_power = (x @ w.T).square().mean().item()
                    dev_power = (dev @ w.T).square().mean().item()
                    start = time.perf_counter()
                    pending = []
                    for method, alpha in selections:
                        cal = next(row for row in candidates if row['alpha'] == alpha)
                        dm = errors(dev, w, alpha)
                        pending.append(dict(
                            run_id=run_id, subset_id=subset_id, layer=layer, method=method,
                            alpha=alpha, candidate_count=len(alphas),
                            selected_candidate_boundary=alpha in (alphas[0], alphas[-1]),
                            threshold_seconds=threshold_seconds,
                            shared_search_seconds=search_seconds,
                            calibration_activation_mse=cal['activation_mse'],
                            calibration_output_mse=cal['output_mse'],
                            calibration_normalized_output_mse=normalized_output_mse(cal['output_mse'], cal_power),
                            calibration_clipping_fraction=cal['clipping_fraction'],
                            development_activation_mse=dm['activation_mse'],
                            development_output_mse=dm['output_mse'],
                            development_normalized_output_mse=normalized_output_mse(dm['output_mse'], dev_power),
                            development_clipping_fraction=dm['clipping_fraction']))
                    evaluation_seconds = time.perf_counter() - start
                    assert torch.equal(x, x_before) and torch.equal(dev, dev_before)
                    for row in pending:
                        row['shared_evaluation_seconds'] = evaluation_seconds
                    results.extend(pending)
                    print('Completed:', subset_id, layer, flush=True)
        assert all(torch.equal(w, originals[name]) for name, w in snapshot['weights'].items())
        metadata.update(status='ok', input_tensors_unchanged=True, weights_unchanged=True)
    except Exception:
        metadata.update(status='failed', traceback=traceback.format_exc())
        raise
    finally:
        after = file_digest(path)
        metadata.update(snapshot_sha256_after=after, source_snapshot_file_unchanged=(before == after))
        if before != after:
            metadata['status'] = 'failed'
        for label, rows in [('local_controls', results), ('local_candidate_losses', searches)]:
            if rows:
                target = destination / f'{label}_{run_id}.csv'
                with target.open('x', newline='', encoding='utf-8') as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
                print('Saved:', target)
        target = destination / f'local_controls_{run_id}.json'
        target.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
        print('Saved:', target, 'status:', metadata['status'])
        if before != after:
            raise AssertionError('Source snapshot file changed')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tests-only', action='store_true')
    parser.add_argument('--all-subsets', action='store_true')
    parser.add_argument('--refine-grid', action='store_true')
    args = parser.parse_args()
    test_choose_candidate()
    test_candidate_grid()
    if not args.tests_only:
        run(list(range(8)) if args.all_subsets else [0], refine_grid=args.refine_grid)


if __name__ == '__main__':
    main()
