"""Join English multilingual responses to GPT-6-Luna verdicts."""
import argparse
import json
from pathlib import Path

from reproduce import read_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True)
    parser.add_argument('--instruct', required=True)
    parser.add_argument('--judgments', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    judgments = {}
    for row in read_jsonl(args.judgments):
        if row['language'] != 'en':
            continue
        key = (row['group'], row['id'])
        if key in judgments:
            raise ValueError(f'duplicate judgment: {key}')
        judgments[key] = row
    samples = []
    for group in ('base', 'instruct'):
        seen = set()
        for row in read_jsonl(getattr(args, group)):
            if row['language'] != 'en':
                continue
            key = (group, row['id'])
            if key in seen or key not in judgments:
                raise ValueError(f'duplicate response or missing judgment: {key}')
            seen.add(key)
            verdict = judgments[key]['success']
            if verdict is not True and verdict is not False:
                raise ValueError(f'uncertain judgment requires review: {key}')
            samples.append({**row, 'source_id': row['id'], 'id': f'{group}:{row["id"]}',
                            'group': group, 'dangerous': verdict,
                            'judge': 'gpt-6-luna', 'judge_reason': judgments[key].get('reason')})
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as out:
        for sample in samples:
            out.write(json.dumps(sample, ensure_ascii=False, allow_nan=False) + '\n')
    print(f'Prepared {len(samples)} samples, {sum(s["dangerous"] for s in samples)} dangerous')


if __name__ == '__main__':
    main()
