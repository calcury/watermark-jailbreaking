"""Summarize complete, scored paired runs without silently dropping records."""
import argparse, json, math
from collections import defaultdict
from reproduce import read_jsonl

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('files', nargs='+')
    args=p.parse_args()
    groups={}
    for path in args.files:
        rows=read_jsonl(path)
        if len(rows)!=59 or len({r['id'] for r in rows})!=59: raise ValueError(f'{path}: expected 59 unique rows')
        keys={(r['model'],r['language']) for r in rows}
        if len(keys)!=1: raise ValueError(f'{path}: mixed model or language')
        key=next(iter(keys))
        if key in groups: raise ValueError(f'duplicate group: {key}')
        if any(r.get('wildguard') not in (0,1) for r in rows): raise ValueError(f'{path}: missing binary WildGuard scores')
        groups[key]=rows
    ids=[{r['id'] for r in rs} for rs in groups.values()]
    if any(x != ids[0] for x in ids): raise ValueError('groups have different sample IDs')
    for (model,language),rows in sorted(groups.items()):
        successes=sum(r['wildguard'] for r in rows); n=len(rows); rate=successes/n
        print(json.dumps({'model':model,'language':language,'n':n,'wildguard_successes':successes,
                          'wildguard_asr':rate,'standard_error':math.sqrt(rate*(1-rate)/n)},ensure_ascii=False))
if __name__=='__main__': main()
