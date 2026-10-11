"""D05 timing only: cached FP model and local search on unchanged D04 tensors."""
import csv
import json
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

RESEARCH = Path(__file__).resolve().parents[1]
D04 = RESEARCH.parents[1] / 'learning' / 'D04'
sys.path.insert(0, str(D04))
from day4_experiment import fake_quantize, measure_errors, test_fake_quantize, test_measure_errors

MODEL = 'HuggingFaceTB/SmolLM2-135M'
REVISION = '93efa2f097d58c2a74874c7e644dbc9b0cee75a2'


def main():
    out = RESEARCH / 'results'
    out.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    rows = []
    metadata = {
        'run_id': run_id, 'python': platform.python_version(),
        'torch': str(torch.__version__), 'threads': torch.get_num_threads(),
        'model': MODEL, 'revision': REVISION, 'dtype': 'float32',
        'local_files_only': True, 'warmups': 1, 'repeats': 3,
        'scope': 'timing smoke; synthetic D04 data; not quality evaluation',
    }

    def record(operation, repeat, seconds, device, tokens=0, candidates=0, **extra):
        rows.append(dict(run_id=run_id, operation=operation, repeat=repeat,
                         seconds=seconds, device=device, tokens=tokens,
                         candidates=candidates, **extra))

    try:
        test_fake_quantize()
        test_measure_errors()
        snapshot = torch.load(D04 / 'artefacts' / 'calibration_snapshot.pt', weights_only=True)
        assert snapshot['metadata']['model_revision'] == REVISION
        x = snapshot['inputs']['representative']
        w = snapshot['weight']
        assert x.shape == (256, 1536) and w.shape == (576, 1536)
        assert torch.isfinite(x).all() and torch.isfinite(w).all()
        x_before, w_before = x.clone(), w.clone()
        alphas = torch.logspace(-1, 1, 21).tolist()
        metadata['alphas'] = alphas
        metadata['snapshot_metadata'] = snapshot['metadata']

        def search():
            errors = [measure_errors(x, fake_quantize(x, a), w, a)['output_mse']
                      for a in alphas]
            best = min(range(len(errors)), key=errors.__getitem__)
            return alphas[best], errors[best]

        with torch.inference_mode():
            search()
            for repeat in range(3):
                start = time.perf_counter()
                alpha, mse = search()
                record('local_search_d04_cpu', repeat, time.perf_counter() - start,
                       'cpu', len(x), len(alphas), selected_alpha=alpha,
                       calibration_output_mse=mse)
                print(rows[-1], flush=True)
        assert torch.equal(x, x_before) and torch.equal(w, w_before)
        metadata['snapshot_inputs_unchanged'] = True

        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable: FP GPU timing not performed')
        metadata['gpu'] = torch.cuda.get_device_name(0)
        tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
        torch.cuda.synchronize()
        start = time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(
            MODEL, revision=REVISION, local_files_only=True,
            torch_dtype=torch.float32, attn_implementation='eager',
        ).to('cuda').eval()
        torch.cuda.synchronize()
        record('model_load_and_transfer', 0, time.perf_counter() - start, 'cuda')
        assert model.config._commit_hash == REVISION
        metadata['resolved_model_revision'] = model.config._commit_hash
        corpus = json.loads((D04 / 'artefacts' / 'calibration_texts.json').read_text(encoding='utf-8'))
        # Real token IDs from a repeated synthetic text, solely for timing.
        text = (corpus['representative'][0] + '\n') * 32
        with torch.inference_mode():
            for length in (32, 128):
                tokens = tokenizer(text, return_tensors='pt', truncation=True,
                                   max_length=length).to('cuda')
                assert tokens['input_ids'].shape == (1, length)
                def forward():
                    result = model(**tokens, use_cache=False)
                    assert result.logits.shape[:2] == (1, length)
                    del result
                forward()
                torch.cuda.synchronize()
                for repeat in range(3):
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    forward()
                    torch.cuda.synchronize()
                    record('fp_forward', repeat, time.perf_counter() - start,
                           'cuda', length, peak_allocated_bytes=torch.cuda.max_memory_allocated())
                    print(rows[-1], flush=True)
        metadata['status'] = 'ok'
    except Exception:
        metadata['status'] = 'failed'
        metadata['traceback'] = traceback.format_exc()
        raise
    finally:
        # Append, retaining all previous runs. Metadata uses unique run IDs.
        path = out / 'timings.csv'
        fields = ['run_id', 'operation', 'repeat', 'seconds', 'device', 'tokens',
                  'candidates', 'selected_alpha', 'calibration_output_mse', 'peak_allocated_bytes']
        exists = path.exists() and path.stat().st_size > 0
        with path.open('a', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            if not exists:
                writer.writeheader()
            writer.writerows(rows)
        (out / f'timing_metadata_{run_id}.json').write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')
        print('Saved:', path, 'status:', metadata.get('status'), flush=True)


if __name__ == '__main__':
    main()
