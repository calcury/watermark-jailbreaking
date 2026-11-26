"""Score original trajectories and branched continuations using StrongREJECT."""
import argparse
import json
from pathlib import Path

from datasets import Dataset
from strong_reject.evaluate import evaluate_dataset
from transformers import AutoTokenizer

from reproduce import read_jsonl


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--batch-size', type=int, default=1)
    args = p.parse_args()
    rows = read_jsonl(args.input)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    flattened = []
    tokenizers = {}
    for row in rows:
        if row['model'] not in tokenizers:
            tokenizers[row['model']] = AutoTokenizer.from_pretrained(row['model'])
        tokenizer = tokenizers[row['model']]
        for name in ('clean', 'watermarked'):
            flattened.append({'id': row['id'], 'goal': row['goal'], 'trajectory': name,
                'offset': None, 'sample': None,
                'response': tokenizer.decode(row[name + '_ids'], skip_special_tokens=True)})
        for branch in row['branches']:
            flattened.append({'id': row['id'], 'goal': row['goal'], 'trajectory': branch['trajectory'],
                'offset': branch['offset'], 'sample': branch['sample'], 'response': branch['response']})
    data = Dataset.from_list([{'forbidden_prompt': r['goal'], 'response': r['response']} for r in flattened])
    scored = evaluate_dataset(data, ['strongreject_finetuned'], batch_size=args.batch_size)
    if len(scored) != len(flattened):
        raise ValueError('StrongREJECT returned an unexpected number of scores')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as out:
        for row, score in zip(flattened, scored['score']):
            out.write(json.dumps({**row, 'strongreject': float(score)}, ensure_ascii=False, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
