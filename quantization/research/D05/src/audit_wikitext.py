"""Inspect train/validation only, with pinned remote revision. No quantization tuning."""
import hashlib
import json
from pathlib import Path

import datasets
import torch
import transformers
from datasets import load_dataset
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
REPO = 'Salesforce/wikitext'
REVISION = 'b08601e04326c79dfdd32d625aee71d232d685c3'
CONFIG = 'wikitext-2-raw-v1'


def main():
    assert str(torch.__version__) == '2.5.1+cu121'
    assert transformers.__version__ == '4.40.2'
    files = {
        split: hf_hub_download(REPO, f'{CONFIG}/{split}-00000-of-00001.parquet',
                               repo_type='dataset', revision=REVISION)
        for split in ('train', 'validation')
    }
    data = load_dataset('parquet', data_files=files)
    report = {
        'repo': REPO, 'config': CONFIG, 'revision': REVISION,
        'datasets_version': datasets.__version__, 'torch_version': str(torch.__version__),
        'transformers_version': transformers.__version__,
        'test_downloaded_by_this_script': False, 'splits': {},
    }
    nonempty = {}
    for split, ds in data.items():
        assert ds.column_names == ['text']
        texts = ds['text']
        nonempty[split] = {text.strip() for text in texts if text.strip()}
        with open(files[split], 'rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        report['splits'][split] = {
            'rows': len(ds), 'blank_rows': sum(not text.strip() for text in texts),
            'unique_nonempty_stripped_rows': len(nonempty[split]),
            'parquet_sha256': digest, 'datasets_fingerprint': ds._fingerprint,
        }
    report['exact_nonempty_stripped_row_overlap_train_validation'] = len(
        nonempty['train'] & nonempty['validation'])
    path = ROOT / 'results' / 'wikitext_audit.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    print('Saved:', path)


if __name__ == '__main__':
    main()
