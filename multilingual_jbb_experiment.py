"""Generate paired multilingual JBB responses; score separately with reproduce.py."""
import argparse
import json
from pathlib import Path
from reproduce import load_model, read_jsonl

MODELS = {'base': 'Qwen/Qwen2.5-3B', 'instruct': 'Qwen/Qwen2.5-3B-Instruct'}
LANGUAGES = ('en', 'zh-CN', 'ja', 'ar')

def generate(args):
    import torch
    from transformers import set_seed
    rows = [r for r in read_jsonl(args.input) if r['language'] == args.language]
    if len(rows) != 59 or len({r['id'] for r in rows}) != 59:
        raise ValueError('Expected exactly 59 distinct samples for the selected language')
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    model, tokenizer = load_model(MODELS[args.model])
    with output.open('x', encoding='utf-8') as f:
        for index, row in enumerate(rows):
            set_seed(args.seed + index)
            if args.model == 'instruct':
                text = tokenizer.apply_chat_template([{'role': 'user', 'content': row['prompt']}], tokenize=False, add_generation_prompt=True)
            else:
                text = row['prompt']
            inputs = tokenizer(text, return_tensors='pt').to(model.device)
            with torch.inference_mode():
                ids = model.generate(**inputs, do_sample=True, temperature=args.temperature, top_k=0,
                                     max_new_tokens=args.max_tokens, pad_token_id=tokenizer.eos_token_id)
            response = tokenizer.decode(ids[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            record = {**row, 'dataset': 'jbb', 'setting': 'direct', 'model': MODELS[args.model],
                      'prompt_format': 'raw' if args.model == 'base' else 'chat_template',
                      'watermark': 'none', 'seed': args.seed + index, 'temperature': args.temperature,
                      'max_new_tokens': args.max_tokens, 'response': response}
            f.write(json.dumps(record, ensure_ascii=False) + '\n'); f.flush()
            print(f'{index+1}/{len(rows)}', flush=True)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', default='multilingual_jbb/translations.jsonl')
    p.add_argument('--model', choices=MODELS, required=True)
    p.add_argument('--language', choices=LANGUAGES, required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--temperature', type=float, default=.7)
    p.add_argument('--max-tokens', type=int, default=512)
    a = p.parse_args()
    if a.temperature <= 0 or a.max_tokens <= 0: p.error('temperature and max-tokens must be positive')
    generate(a)

if __name__ == '__main__': main()
