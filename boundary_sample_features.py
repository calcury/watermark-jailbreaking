"""Replay labeled responses under both Qwen models and save pooled hidden states."""
import argparse
from pathlib import Path

import numpy as np
import torch

from reproduce import load_model, read_jsonl


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--layers', type=int, nargs='+', default=[-1])
    args = p.parse_args()
    rows = read_jsonl(args.input)
    if not rows:
        raise ValueError('empty input')
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    model, tokenizer = load_model(args.model)
    vectors = []
    with torch.inference_mode():
        for row in rows:
            prefix = tokenizer.apply_chat_template([{'role': 'user', 'content': row['prompt']}],
                tokenize=True, add_generation_prompt=True, return_tensors='pt').to(model.device)
            answer = tokenizer(row['response'], add_special_tokens=False, return_tensors='pt').input_ids.to(model.device)
            if answer.shape[1] == 0:
                raise ValueError(f'empty response: {row["id"]}')
            combined = torch.cat([prefix, answer], dim=1)
            states = model(combined, output_hidden_states=True).hidden_states
            vectors.append(np.stack([states[layer][0, prefix.shape[1]:].float().mean(dim=0).cpu().numpy()
                                     for layer in args.layers]))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as out:
        np.savez_compressed(out, features=np.stack(vectors), ids=np.array([r['id'] for r in rows]),
                            labels=np.array([r['dangerous'] for r in rows]),
                            source_ids=np.array([r['source_id'] for r in rows]),
                            groups=np.array([r['group'] for r in rows]),
                            model=np.array(args.model), layers=np.array(args.layers))


if __name__ == '__main__':
    main()
