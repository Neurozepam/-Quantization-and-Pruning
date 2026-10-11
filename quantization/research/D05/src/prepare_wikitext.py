import hashlib
import json
import random
import re
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
DATA_REPO = 'Salesforce/wikitext'
DATA_REVISION = 'b08601e04326c79dfdd32d625aee71d232d685c3'
MODEL = 'HuggingFaceTB/SmolLM2-135M'
MODEL_REVISION = '93efa2f097d58c2a74874c7e644dbc9b0cee75a2'
LENGTH = 128
POOL_SIZE = 64
SEED = 42
# Single-level article title; excludes nested '= = Section = =' headings.
ARTICLE_TITLE = re.compile(r'^\s*=\s+([^=].*?)\s+=\s*$')


def iter_articles(rows):
    """Combine consecutive rows until the next top-level title."""
    title, start_row, body = None, None, []
    for row_id, text in enumerate(rows):
        match = ARTICLE_TITLE.fullmatch(text)
        if match:
            if title is not None:
                yield dict(title=title, start_row=start_row, text=''.join(body))
            title, start_row, body = match.group(1), row_id, [text]
        elif title is not None:
            body.append(text)
    if title is not None:
        yield dict(title=title, start_row=start_row, text=''.join(body))


def build_pool(rows, tokenizer, split):
    pool = []
    for article in iter_articles(rows):
        # Full articles are only tokenized; model inputs are fixed-length slices.
        ids = tokenizer.encode(article['text'], add_special_tokens=False, verbose=False)
        if len(ids) < LENGTH:
            continue
        pool.append(dict(
            split=split, article_title=article['title'],
            article_start_row=article['start_row'],
            article_sha256=hashlib.sha256(article['text'].encode('utf-8')).hexdigest(),
            token_ids=ids,
        ))
        if len(pool) == POOL_SIZE:
            break
    required = 2 if split == 'train' else 16
    assert len(pool) >= required, f'Not enough eligible articles: {split}'
    return pool


def sample_blocks(pool, rng, count, length):
    samples = rng.sample(pool, count)
    blocks = []
    for sample in samples:
        start = rng.randint(0, len(sample['token_ids']) - length)
        blocks.append(dict(
            split=sample['split'],
            article_title=sample['article_title'],
            article_start_row=sample['article_start_row'],
            article_sha256=sample['article_sha256'],
            token_start=start,
            token_ids=sample['token_ids'][start:start + length]
        ))
    return blocks


def check_blocks(blocks, count, length):
    assert len(blocks) == count
    assert len({b['article_start_row'] for b in blocks}) == count
    for block in blocks:
        assert len(block['token_ids']) == length
        assert block['token_start'] >= 0
        assert all(isinstance(token, int) for token in block['token_ids'])


def test_sample_blocks():
    pool = [dict(split='toy', article_title=str(i), article_start_row=i,
                 article_sha256=str(i), token_ids=list(range(i * 100, i * 100 + 20)))
            for i in range(5)]
    original = json.dumps(pool, sort_keys=True)
    a = sample_blocks(pool, random.Random(42), count=2, length=8)
    b = sample_blocks(pool, random.Random(42), count=2, length=8)
    assert a == b, 'Same seed must give the same blocks'
    check_blocks(a, 2, 8)
    assert json.dumps(pool, sort_keys=True) == original, 'Pool was modified'
    for block in a:
        source = pool[block['article_start_row']]
        start = block['token_start']
        assert block['token_ids'] == source['token_ids'][start:start + 8]
    # Exact-length article has only one valid start: zero.
    exact = [dict(pool[0], token_ids=list(range(8)))]
    block = sample_blocks(exact, random.Random(42), 1, 8)[0]
    assert block['token_start'] == 0 and block['token_ids'] == list(range(8))
    print('sample_blocks tests: PASS')


def main():
    test_sample_blocks()
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL, revision=MODEL_REVISION, local_files_only=True)
    files = {split: hf_hub_download(
        DATA_REPO, f'wikitext-2-raw-v1/{split}-00000-of-00001.parquet',
        repo_type='dataset', revision=DATA_REVISION, local_files_only=True)
        for split in ('train', 'validation')}
    data = load_dataset('parquet', data_files=files)
    pools = {split: build_pool(ds['text'], tokenizer, split) for split, ds in data.items()}
    assert not ({a['article_sha256'] for a in pools['train']} &
                {a['article_sha256'] for a in pools['validation']})
    calibration = []
    for subset_id in range(8):
        blocks = sample_blocks(pools['train'], random.Random(SEED + subset_id), 2, LENGTH)
        check_blocks(blocks, 2, LENGTH)
        calibration.append(dict(subset_id=subset_id, seed=SEED + subset_id, blocks=blocks))
    development = sample_blocks(pools['validation'], random.Random(1042), 16, LENGTH)
    check_blocks(development, 16, LENGTH)
    cal_ids = {tuple(b['token_ids']) for s in calibration for b in s['blocks']}
    assert not (cal_ids & {tuple(b['token_ids']) for b in development})
    manifest = dict(
        dataset=DATA_REPO, dataset_config='wikitext-2-raw-v1',
        dataset_revision=DATA_REVISION, model=MODEL, tokenizer_revision=MODEL_REVISION,
        add_special_tokens=False, sequence_length=LENGTH, pool_size_cap=POOL_SIZE,
        actual_pool_sizes={split: len(pool) for split, pool in pools.items()},
        scope='smoke pilot: up to first 64 eligible articles per split, not full-corpus sampling',
        calibration_subsets_can_overlap=True,
        fragments_processed_separately=True, test_accessed=False,
        calibration=calibration, development=development,
    )
    path = ROOT / 'data' / 'wikitext_blocks.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f'Refusing to overwrite existing manifest: {path}')
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Saved:', path)
    print('Calibration: 8 subsets x 256 tokens; development: 2048 tokens')


if __name__ == '__main__':
    main()
