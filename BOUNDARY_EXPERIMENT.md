# Safety boundary experiment

Run from the repository root after installing the dependencies and MarkLLM described in README.md. Supply a JSONL input with unique `id` and `prompt` fields. The model and tokenizer must be available on the server.

```bash
python boundary_experiment.py --input data/jbb.jsonl --output outputs/boundary.jsonl --watermark KGW --samples 16 --layers -1
```

Each row records both sampled token trajectories and the first differing index (zero-based). Branches at offsets 1, 4, 8 and 16 replay each trajectory, turn off the watermark and sample full responses. Paired branch samples use the same seeds. Feature vectors are stored on sample zero only. Rows with no differing token have no branches. A position at or beyond the output limit, or after EOS, is skipped. This script generates potentially harmful content. Judge the complete branch responses separately, blinded to trajectory, and calculate per-request harmful continuation rates with uncertainty clustered by request.

For a preliminary rank-one activation analysis, create a separate JSONL label file with `id`, `clean`, and `watermarked` binary labels for the original paired trajectories. Use independently reviewed labels. Then run:

```bash
python boundary_analysis.py --input outputs/boundary.jsonl --labels labels.jsonl --output outputs/projections.jsonl --layer -1 --offset 1
```

Each test fold uses a mean activation difference fitted only on safe-to-unsafe training pairs. The output contains held-out projections. Split related prompts together externally before analysis if the input contains paraphrases. Small numbers of safe-to-unsafe pairs produce unstable directions. These outputs do not themselves estimate the continuation risk or establish a causal subspace. Additional scoring, random-direction controls, logit-lens analysis, visualization, and cross-topic validation remain separate work.
