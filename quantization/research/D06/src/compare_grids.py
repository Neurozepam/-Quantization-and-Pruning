"""Verify nested-grid control and save paired coarse/refined diagnostics."""
import argparse
import csv
import json
import math
import statistics as st
from pathlib import Path


def load(path):
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def index(rows, fields):
    result = {tuple(r[f] for f in fields): r for r in rows}
    assert len(result) == len(rows)
    return result


def same(a, b):
    return math.isclose(float(a), float(b), rel_tol=1e-6, abs_tol=1e-6)


def write(path, rows):
    with path.open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('coarse', type=Path)
    parser.add_argument('refined', type=Path)
    args = parser.parse_args()
    cmeta = json.loads(args.coarse.with_suffix('.json').read_text(encoding='utf-8'))
    rmeta = json.loads(args.refined.with_suffix('.json').read_text(encoding='utf-8'))
    for meta in (cmeta, rmeta):
        assert meta['status'] == 'ok' and meta['source_snapshot_file_unchanged']
        assert meta['input_tensors_unchanged'] and meta['weights_unchanged']
        assert not meta['test_accessed'] and meta['subset_ids'] == list(range(8))
    assert rmeta['refine_grid'] and not cmeta.get('refine_grid', False)
    for field in ('snapshot_sha256_before', 'layers', 'calibration_tokens_per_run',
                  'development_tokens_per_site', 'dtype', 'bit_codes', 'threads'):
        assert cmeta[field] == rmeta[field], field
    keys = ('layer', 'subset_id', 'method')
    coarse = index(load(args.coarse), keys)
    refined = index(load(args.refined), keys)
    assert coarse.keys() == refined.keys() and len(coarse) == 96
    losses_keys = ('layer', 'subset_id', 'alpha')
    cp = args.coarse.with_name(f"local_candidate_losses_{cmeta['run_id']}.csv")
    rp = args.refined.with_name(f"local_candidate_losses_{rmeta['run_id']}.csv")
    old_losses, new_losses = index(load(cp), losses_keys), index(load(rp), losses_keys)
    assert old_losses.keys() <= new_losses.keys()
    for key, old in old_losses.items():
        for field in ('activation_mse', 'output_mse', 'clipping_fraction'):
            assert same(old[field], new_losses[key][field]), (key, field)
    pairs, summary = [], []
    for key, old in coarse.items():
        new = refined[key]
        layer, subset, method = key
        if method in ('max_abs', 'p999'):
            for field in ('alpha', 'calibration_output_mse', 'development_output_mse'):
                assert same(old[field], new[field]), (key, field)
            continue
        objective = 'activation_mse' if method == 'cal_activation_mse' else 'output_mse'
        field = f'calibration_{objective}'
        cal_c, cal_r = float(old[field]), float(new[field])
        assert cal_r <= cal_c + 1e-6 * max(1., abs(cal_c))
        dev_c, dev_r = float(old['development_output_mse']), float(new['development_output_mse'])
        pairs.append(dict(layer=layer, subset_id=int(subset), method=method,
                          coarse_alpha=float(old['alpha']), refined_alpha=float(new['alpha']),
                          coarse_cal_objective=cal_c, refined_cal_objective=cal_r,
                          cal_relative_percent=100*(cal_r/cal_c-1),
                          coarse_dev_output_mse=dev_c, refined_dev_output_mse=dev_r,
                          dev_relative_percent=100*(dev_r/dev_c-1)))
    for layer in cmeta['layers']:
        for method in ('cal_activation_mse', 'cal_output_mse'):
            pp = [p for p in pairs if p['layer']==layer and p['method']==method]
            old = [coarse[layer, str(i), method] for i in range(8)]
            new = [refined[layer, str(i), method] for i in range(8)]
            summary.append(dict(layer=layer, method=method,
                                alpha_changed_count=sum(p['coarse_alpha']!=p['refined_alpha'] for p in pp),
                                cal_relative_percent_mean=st.mean(p['cal_relative_percent'] for p in pp),
                                dev_relative_percent_mean=st.mean(p['dev_relative_percent'] for p in pp),
                                dev_better_count=sum(p['dev_relative_percent']<0 for p in pp),
                                dev_worse_count=sum(p['dev_relative_percent']>0 for p in pp),
                                coarse_alpha_sample_std=st.stdev(float(r['alpha']) for r in old),
                                refined_alpha_sample_std=st.stdev(float(r['alpha']) for r in new),
                                coarse_dev_nmse_sample_std=st.stdev(float(r['development_normalized_output_mse']) for r in old),
                                refined_dev_nmse_sample_std=st.stdev(float(r['development_normalized_output_mse']) for r in new)))
    checks = dict(coarse_run=cmeta['run_id'], refined_run=rmeta['run_id'],
                  original_candidate_count=len(old_losses), refined_candidate_count=len(new_losses),
                  old_candidates_preserved=True, original_candidate_losses_match=True,
                  fixed_baselines_unchanged=True, selected_calibration_objectives_nonincreasing=True,
                  coarse_shared_search_seconds_total=sum(float(coarse[layer,str(i),'cal_output_mse']['shared_search_seconds']) for layer in cmeta['layers'] for i in range(8)),
                  refined_shared_search_seconds_total=sum(float(refined[layer,str(i),'cal_output_mse']['shared_search_seconds']) for layer in cmeta['layers'] for i in range(8)),
                  comparison_tolerance=dict(relative=1e-6, absolute=1e-6),
                  independent_subsets=False)
    tag = rmeta['run_id']
    write(args.refined.parent/f'grid_comparison_{tag}.csv', pairs)
    write(args.refined.parent/f'grid_comparison_summary_{tag}.csv', summary)
    path=args.refined.parent/f'grid_comparison_checks_{tag}.json'
    with path.open('x', encoding='utf-8') as stream:
        json.dump(checks, stream, indent=2)
    print(json.dumps(checks, indent=2))
    for row in summary:
        print(row)


if __name__ == '__main__':
    main()
