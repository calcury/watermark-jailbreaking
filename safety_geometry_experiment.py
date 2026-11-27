"""Prepare multilingual labels and extract states for safety_geometry_analysis.py."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from reproduce import read_jsonl


def prepare(args):
    judgments = {}
    for row in read_jsonl(args.judgments):
        key = (row['group'], row['language'], row['id'])
        if key in judgments:
            raise ValueError(f'duplicate judgment {key}')
        judgments[key] = row['success']
    records = []
    seen = set()
    for specification in args.responses:
        group, path = specification.split('=', 1)
        for row in read_jsonl(path):
            if row['language'] not in args.languages:
                continue
            key = (group, row['language'], row['id'])
            if key in seen or key not in judgments:
                raise ValueError(f'duplicate response or missing judgment {key}')
            if type(judgments[key]) is not bool:
                raise ValueError(f'uncertain judgment {key} requires review before preparing')
            if not all(isinstance(row.get(k), str) and row[k].strip()
                       for k in ('id', 'goal', 'prompt')) or not isinstance(row.get('response'), str):
                raise ValueError(f'empty or invalid text fields {key}')
            seen.add(key)
            records.append({k: row[k] for k in ('id', 'language', 'goal', 'prompt', 'response')}
                           | {'group': group, 'dangerous': judgments[key],
                              'source_model': row['model'], 'judge': 'gpt-6-luna'})
    if not records:
        raise ValueError('no selected responses')
    grouped = defaultdict(list)
    for row in records:
        grouped[(row['id'], row['group'])].append(row)
    for key, rows in grouped.items():
        if {r['language'] for r in rows} != set(args.languages):
            raise ValueError(f'incomplete language pairing {key}')
        if len({r['goal'] for r in rows}) != 1:
            raise ValueError(f'original goal differs across languages {key}')
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        for row in sorted(records, key=lambda r: (r['id'], r['language'], r['group'])):
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps({'responses': len(records), 'dangerous': sum(r['dangerous'] for r in records)}))


def english_anchors(rows, tokenizer):
    """Use opposite-label English responses at their first token divergence."""
    grouped = defaultdict(list)
    for row in rows:
        if row['language'] == 'en':
            grouped[row['id']].append(row)
    anchors = {}
    for ident, candidates in grouped.items():
        safe = sorted((r for r in candidates if not r['dangerous']), key=lambda r: r['group'])
        dangerous = sorted((r for r in candidates if r['dangerous']), key=lambda r: r['group'])
        if not safe or not dangerous:
            continue
        safe_ids = tokenizer.encode(safe[0]['response'], add_special_tokens=False)
        dangerous_ids = tokenizer.encode(dangerous[0]['response'], add_special_tokens=False)
        divergence = next((i for i, (a, b) in enumerate(zip(safe_ids, dangerous_ids)) if a != b), None)
        if divergence is None:
            continue
        anchors[ident] = {'prefix': safe_ids[:divergence], 'safe_token': safe_ids[divergence],
                          'danger_token': dangerous_ids[divergence], 'safe_group': safe[0]['group'],
                          'danger_group': dangerous[0]['group']}
    return anchors


def extract(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if args.device == 'musa':
        import torch_musa  # noqa: F401 - register device before loading.
    rows = read_jsonl(args.input)
    keys = [(r['id'], r['language'], r['group']) for r in rows]
    if not rows or len(set(keys)) != len(keys):
        raise ValueError('input must contain unique prepared records')
    if any(type(r['dangerous']) is not bool for r in rows):
        raise ValueError('dangerous must be boolean')
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    tokenizer = AutoTokenizer.from_pretrained(args.template_tokenizer, local_files_only=args.offline)
    model_tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=args.offline)
    if tokenizer.get_vocab() != model_tokenizer.get_vocab():
        raise ValueError('model and template token vocabularies must be identical')
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=getattr(torch, args.dtype), local_files_only=args.offline,
        attn_implementation='eager').to(args.device).eval()
    head = model.get_output_embeddings()
    anchors = english_anchors(rows, tokenizer)
    prompt_states, answer_states = [], []
    answer_valid, answer_lengths = [], []
    prompt_cache, anchor_cache = {}, {}
    anchor_keys, anchor_states, directions, biases, model_margins = [], [], [], [], []
    anchor_tokens, anchor_prefixes = [], []

    def forward(ids):
        tensor = torch.tensor([ids], dtype=torch.long, device=args.device)
        context_limit = getattr(model.config, 'max_position_embeddings', None)
        if context_limit and len(ids) > context_limit:
            raise ValueError('input exceeds model context length')
        result = model(tensor, attention_mask=torch.ones_like(tensor), output_hidden_states=True, use_cache=False)
        return result.hidden_states[-1][0, -1].float().cpu().numpy(), result.logits[0, -1]

    with torch.inference_mode():
        for index, row in enumerate(rows):
            key = (row['id'], row['language'])
            if key not in prompt_cache:
                prompt = tokenizer.apply_chat_template(
                    [{'role': 'user', 'content': row['prompt']}], tokenize=True, add_generation_prompt=True)
                state, _ = forward(prompt)
                prompt_cache[key] = (row['prompt'], prompt, state)
            text, prompt, state = prompt_cache[key]
            if text != row['prompt']:
                raise ValueError(f'groups have different prompts for {key}')
            answer = tokenizer.encode(row['response'], add_special_tokens=False)
            valid = len(answer) >= args.answer_tokens
            answer_state = np.zeros_like(state)
            if valid:
                answer_state, _ = forward(prompt + answer[:args.answer_tokens])
            prompt_states.append(state)
            answer_states.append(answer_state)
            answer_valid.append(valid)
            answer_lengths.append(len(answer))
            if row['id'] in anchors and key not in anchor_cache:
                anchor = anchors[row['id']]
                anchored, logits = forward(prompt + anchor['prefix'])
                safe, dangerous = anchor['safe_token'], anchor['danger_token']
                safe_weight = head.weight[safe].float().cpu().numpy().astype(np.float64)
                danger_weight = head.weight[dangerous].float().cpu().numpy().astype(np.float64)
                direction = safe_weight - danger_weight
                bias = 0.0 if head.bias is None else float((head.bias[safe] - head.bias[dangerous]).item())
                observed = float((logits[safe].float() - logits[dangerous].float()).item())
                analytic = float(direction @ anchored.astype(np.float64) + bias)
                # Reject model architectures whose final hidden state is not the head input.
                if abs(observed - analytic) > args.head_tolerance:
                    raise ValueError(f'head identity failed for {key}: error={abs(observed - analytic)}')
                anchor_keys.append(key)
                anchor_states.append(anchored)
                directions.append(direction)
                biases.append(bias)
                model_margins.append(observed)
                anchor_tokens.append([safe, dangerous])
                anchor_prefixes.append(json.dumps(anchor['prefix']))
                anchor_cache[key] = True
            print(f'{index + 1}/{len(rows)}', flush=True)
    dimension = len(prompt_states[0])
    metadata = {'model': args.model, 'template_tokenizer': args.template_tokenizer,
                'prompt_format': 'shared_chat_template', 'device': args.device, 'dtype': args.dtype,
                'answer_tokens': args.answer_tokens, 'head_tolerance': args.head_tolerance,
                'input_sha256': hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
                'state_space': 'final_normalized_lm_head_input', 'torch_version': torch.__version__,
                'anchors': anchors}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as stream:
        np.savez_compressed(stream, ids=np.array([r['id'] for r in rows]),
            languages=np.array([r['language'] for r in rows]), groups=np.array([r['group'] for r in rows]),
            labels=np.array([r['dangerous'] for r in rows]), prompt=np.stack(prompt_states),
            answer=np.stack(answer_states), answer_lengths=np.array(answer_lengths),
            answer_valid=np.array(answer_valid),
            anchor_ids=np.array([k[0] for k in anchor_keys], dtype=str),
            anchor_languages=np.array([k[1] for k in anchor_keys], dtype=str),
            anchor=np.array(anchor_states, dtype=np.float64).reshape(-1, dimension),
            directions=np.array(directions, dtype=np.float64).reshape(-1, dimension),
            biases=np.array(biases), model_margins=np.array(model_margins),
            candidate_tokens=np.array(anchor_tokens, dtype=np.int64).reshape(-1, 2),
            anchor_prefixes=np.array(anchor_prefixes, dtype=str), metadata=np.array(json.dumps(metadata)))
    print(f'Saved {len(rows)} responses and {len(anchor_keys)} anchored states to {output}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('prepare')
    p.add_argument('--responses', action='append', required=True, help='Repeat group=JSONL for each group/language')
    p.add_argument('--judgments', required=True)
    p.add_argument('--languages', nargs='+', default=['en', 'zh-CN', 'ja', 'ar'])
    p.add_argument('--output', required=True)
    p.set_defaults(function=prepare)
    p = commands.add_parser('extract')
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--model', required=True, help='HF identifier or local snapshot path')
    p.add_argument('--template-tokenizer', default='Qwen/Qwen2.5-3B-Instruct', help='Shared across base and Instruct runs')
    p.add_argument('--device', choices=['cpu', 'cuda', 'musa'], default='cuda')
    p.add_argument('--dtype', choices=['float16', 'bfloat16', 'float32'], default='float16')
    p.add_argument('--offline', action='store_true')
    p.add_argument('--answer-tokens', type=int, default=8)
    p.add_argument('--head-tolerance', type=float, default=0.05)
    p.set_defaults(function=extract)
    args = parser.parse_args()
    if args.command == 'extract' and (args.answer_tokens < 1 or args.head_tolerance <= 0):
        parser.error('answer-tokens and head-tolerance must be positive')
    if args.command == 'prepare' and ('en' not in args.languages or len(args.languages) != len(set(args.languages))):
        parser.error('languages must be unique and include en')
    args.function(args)


if __name__ == '__main__':
    main()
