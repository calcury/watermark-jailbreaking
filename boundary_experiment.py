"""Generate paired trajectories, branch continuations, and activation features."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import LogitsProcessorList, set_seed

from reproduce import load_model, make_watermark, read_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, help='JSONL with unique id and prompt')
    parser.add_argument('--output', required=True, help='New output JSONL')
    parser.add_argument('--model', default='Qwen/Qwen2.5-3B-Instruct')
    parser.add_argument('--watermark', choices=['KGW', 'SWEET'], default='KGW')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--max-tokens', type=int, default=128)
    parser.add_argument('--samples', type=int, default=16)
    parser.add_argument('--offsets', type=int, nargs='+', default=[1, 4, 8, 16])
    parser.add_argument('--layers', type=int, nargs='+', default=[-1])
    args = parser.parse_args()
    if args.samples < 1 or args.max_tokens < 1 or any(x < 1 for x in args.offsets):
        parser.error('samples, max-tokens and offsets must be positive')
    rows = read_jsonl(args.input)
    if not rows or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('input must have unique ids and at least one row')
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    model, tokenizer = load_model(args.model)
    wm = make_watermark(args.watermark, model, tokenizer, str(model.device),
                        args.temperature, args.max_tokens, 0)
    output.parent.mkdir(parents=True, exist_ok=True)

    def generate(prefix, seed, watermarked=False, remaining=None):
        set_seed(seed)
        kwargs = dict(do_sample=True, temperature=args.temperature, top_k=0,
                      max_new_tokens=remaining or args.max_tokens,
                      pad_token_id=tokenizer.eos_token_id)
        if watermarked:
            kwargs['logits_processor'] = LogitsProcessorList([wm.logits_processor])
        ids = model.generate(prefix, attention_mask=torch.ones_like(prefix), **kwargs)
        return ids[0, prefix.shape[1]:].tolist()

    with output.open('x', encoding='utf-8') as out, torch.inference_mode():
        for index, row in enumerate(rows):
            prompt = tokenizer.apply_chat_template(
                [{'role': 'user', 'content': row['prompt']}], tokenize=True,
                add_generation_prompt=True, return_tensors='pt').to(model.device)
            seed = args.seed + index * 100000
            clean = generate(prompt, seed)
            marked = generate(prompt, seed, True)
            divergence = next((i for i, (a, b) in enumerate(zip(clean, marked)) if a != b), None)
            result = {**row, 'model': args.model, 'watermark': args.watermark,
                      'seed': seed, 'temperature': args.temperature, 'max_new_tokens': args.max_tokens,
                      'prompt_token_ids': prompt[0].tolist(), 'clean_ids': clean,
                      'watermarked_ids': marked, 'divergence_index': divergence, 'branches': []}
            if divergence is not None:
                for offset in sorted(set(args.offsets)):
                    position = divergence + offset
                    for name, ids in [('clean', clean), ('watermarked', marked)]:
                        if position > len(ids) or position >= args.max_tokens:
                            continue
                        prefix_ids = ids[:position]
                        if tokenizer.eos_token_id in prefix_ids:
                            continue
                        prefix = torch.cat([prompt, torch.tensor([prefix_ids], device=model.device)], dim=1)
                        states = model(prefix, output_hidden_states=True).hidden_states
                        features = {str(layer): states[layer][0, -1].float().cpu().tolist()
                                    for layer in args.layers}
                        for sample in range(args.samples):
                            continuation = generate(prefix, seed + 1000 + sample,
                                                    remaining=args.max_tokens - position)
                            result['branches'].append({'trajectory': name, 'offset': offset,
                                'prefix_length': position, 'sample': sample, 'features': features if sample == 0 else None,
                                'response': tokenizer.decode(prefix_ids + continuation, skip_special_tokens=True),
                                'continuation_ids': continuation})
            out.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + '\n')
            out.flush()
            print(f'{index + 1}/{len(rows)}', flush=True)


if __name__ == '__main__':
    main()
