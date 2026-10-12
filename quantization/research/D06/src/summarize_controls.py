"""Summarize paired local controls; no seed selection or independence assumption."""
import argparse
import csv
import json
import math
import statistics as st
from pathlib import Path

METHODS = ('max_abs', 'p999', 'cal_activation_mse', 'cal_output_mse')


def write_csv(path, rows):
    with path.open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    args = parser.parse_args()
    source = args.source
    metadata = json.loads(source.with_suffix('.json').read_text(encoding='utf-8'))
    assert metadata['status'] == 'ok' and metadata['source_snapshot_file_unchanged']
    assert metadata['input_tensors_unchanged'] and metadata['weights_unchanged']
    assert metadata['subset_ids'] == list(range(8)) and not metadata['test_accessed']
    with source.open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 96
    index = {(r['layer'], int(r['subset_id']), r['method']): r for r in rows}
    assert len(index) == len(rows)
    run_id = metadata['run_id']
    pairs, summary, spread = [], [], []
    for layer in metadata['layers']:
        layer_pairs = []
        for subset in range(8):
            p = index[layer, subset, 'p999']
            q = index[layer, subset, 'cal_output_mse']
            cal_p, cal_q = float(p['calibration_output_mse']), float(q['calibration_output_mse'])
            dev_p, dev_q = float(p['development_output_mse']), float(q['development_output_mse'])
            assert cal_q <= cal_p + 1e-6 * max(1., cal_p)
            assert dev_p > 0 and cal_p > 0
            difference = dev_q - dev_p
            pair = dict(run_id=run_id, layer=layer, subset_id=subset,
                        p999_alpha=float(p['alpha']), search_alpha=float(q['alpha']),
                        p999_cal_output_mse=cal_p, search_cal_output_mse=cal_q,
                        p999_dev_output_mse=dev_p, search_dev_output_mse=dev_q,
                        cal_difference=cal_q-cal_p, dev_difference=difference,
                        dev_relative_percent=100*difference/dev_p,
                        search_worse_on_dev=difference>0, exact_tie=difference==0)
            pairs.append(pair)
            layer_pairs.append(pair)
        differences = [p['dev_relative_percent'] for p in layer_pairs]
        summary.append(dict(run_id=run_id, layer=layer, subsets=8,
                            worse_count=sum(p['search_worse_on_dev'] for p in layer_pairs),
                            better_count=sum(p['dev_difference']<0 for p in layer_pairs),
                            exact_tie_count=sum(p['exact_tie'] for p in layer_pairs),
                            dev_relative_percent_mean=st.mean(differences),
                            dev_relative_percent_median=st.median(differences),
                            dev_relative_percent_min=min(differences),
                            dev_relative_percent_max=max(differences),
                            independent_subsets=False))
        for method in METHODS:
            rr = [index[layer, i, method] for i in range(8)]
            alphas = [float(r['alpha']) for r in rr]
            nmse = [float(r['development_normalized_output_mse']) for r in rr]
            assert all(math.isfinite(x) for x in alphas+nmse)
            spread.append(dict(run_id=run_id, layer=layer, method=method,
                               alpha_min=min(alphas), alpha_max=max(alphas),
                               alpha_unique_count=len(set(alphas)),
                               dev_normalized_mse_mean=st.mean(nmse),
                               dev_normalized_mse_sample_std=st.stdev(nmse),
                               dev_normalized_mse_min=min(nmse),
                               dev_normalized_mse_max=max(nmse),
                               candidate_boundary_count=sum(r['selected_candidate_boundary']=='True' for r in rr),
                               independent_subsets=False))
    for label, data in [('failure_mode', pairs), ('failure_mode_summary', summary),
                        ('threshold_spread', spread)]:
        destination = source.parent / f'{label}_{run_id}.csv'
        write_csv(destination, data)
        print('Saved:', destination)
    for row in summary:
        print(row)


if __name__ == '__main__':
    main()
