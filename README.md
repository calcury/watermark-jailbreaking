# 水印与越狱评测复现

本仓库实现 [Han 等，2026](https://aclanthology.org/2026.findings-acl.1797.pdf) 的部分静态提示实验流程：获取 JBB/AdvBench，加载 Qwen 指令模型，在解码时施加水印，使用 WildGuard 和 StrongREJECT 评测。输出为 JSONL，每条包含原始目标、提示、模型输出及评测结果。

**复现范围和限制：**当前接入 MarkLLM 的 KGW、Unigram、SWEET、MorphMark、DIP 和无水印基线；EXP、AutoDAN/DSN 的优化、检测 AUC/TPR、人工评估和论文中的拒绝词实验尚未实现。可将外部已有的固定攻击提示按下述 JSONL 格式输入，但这不等同于论文中按目标模型优化的攻击。此处使用 Qwen2.5-1.5B 和 3B，1.5B 不在论文实验模型中；评测仅保留 WildGuard 和 StrongREJECT，因此无法计算论文中的 ΔLG。论文未公开所有解码和攻击细节；本实现采用 MarkLLM 的其他默认参数并关闭 top-k 截断，不能保证逐值复现。

## 环境与获取

推荐 Linux、Python 3.10 和 CUDA GPU。模型权重和两个评审模型需要大量显存、磁盘和时间。WildGuard、Gemma 等需要先在 Hugging Face 接受各自条款，使用 `hf auth login` 登录。请勿提交 token、数据或生成内容。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
git clone https://github.com/THU-BPM/MarkLLM.git third_party/MarkLLM
pip install -r third_party/MarkLLM/requirements.txt
pip install 'git+https://github.com/dsbowen/strong_reject.git'
python reproduce.py download-data
python reproduce.py download-models qwen-1.5b qwen-3b wildguard
```

数据下载锁定上游版本、SHA-256 和条数；失败时检查网络及访问许可。权重缓存在 Hugging Face 默认缓存中，按需指定 `HF_HOME`。StrongREJECT 首次运行会按其官方包下载微调评审器及依赖。生成模型可选 `qwen-1.5b`、`qwen-3b`。请记录 MarkLLM 和依赖的具体版本，作者未提供对应版本，仓库当前使用的是上游实现。

## 运行

```bash
python reproduce.py generate --input data/jbb.jsonl --output outputs/jbb-none.jsonl --model qwen-3b --watermark none
python reproduce.py generate --input data/jbb.jsonl --output outputs/jbb-kgw.jsonl --model qwen-3b --watermark KGW
python reproduce.py judge --input outputs/jbb-kgw.jsonl --output outputs/wg.jsonl --judge wildguard
python reproduce.py judge --input outputs/wg.jsonl --output outputs/scored.jsonl --judge strongreject
python reproduce.py summarize --input outputs/scored.jsonl
```

先对 `none` 和每一种水印分别运行完整评测，在相同模型、数据集和 setting 下相减得到 ΔWG、ΔSR。`summarize` 输出均值和标准误；论文 ΔSR 的保守不确定度为两组标准误平方和的平方根。小规模调试可使用 `--limit 2 --min-tokens 1 --max-tokens 8`，不可与论文结果比较。默认种子 123、temperature 0.7、最少 75、最多 512 新 token；论文参数见附录表 5、6。MarkLLM 的运行配置写在 `outputs/configs`。

输入的每行至少包含 `id`、`dataset`、`goal`、`prompt`、`setting`。`goal` 是原始有害目标，`prompt` 是实际输入目标模型的固定提示；StrongREJECT 使用 `goal`，安全评审使用实际 `prompt`。请保持不同水印使用同一份提示，并保留模型、数据、依赖版本及完整结果。评审无法解析时程序会报错，避免静默当作安全。

### Colab T4 的 WildGuard 评审

WildGuard 为 7B 模型；在 T4 显存或内存不足时可使用 4-bit NF4 量化。量化会改变评审模型的数值，可能改变判定结果，与论文的全精度评审不完全可比；两组必须使用同样设置。安装适用于当前 PyTorch/CUDA 的 `bitsandbytes` 后运行：

```bash
pip install bitsandbytes
python -u reproduce.py judge --input outputs/jbb-none.jsonl --output outputs/jbb-none-wg-4bit.jsonl --judge wildguard --quantization 4bit --debug
```

每条评分完成后即写入 JSONL，控制台显示样本进度、耗时；`--debug` 额外显示输入长度、设备映射和内存。中断后对同一输入、输出和设置追加 `--resume` 继续；已有文件默认不会覆盖。旧版本的输出不包含恢复所需配置，请使用新文件。`--limit 2` 可先做小规模测试。Colab 运行时重置会删除临时文件，请及时下载。若是在模型加载时进程被系统终止，日志可能没有 Python 异常；释放其他进程或减少内存占用后重试。StrongREJECT 仍使用其官方包，不支持此量化和断点续跑。

## 绘制结果

将无水印和水印结果传给绘图脚本，输出默认保存在 `outputs/`。输入可以是生成结果，也可以是包含评审结果的 JSONL；两组必须具有相同的样本 ID、模型、目标、提示和实验设置。

```bash
python plot_results.py --baseline outputs/jbb-none.jsonl --watermarked outputs/jbb-kgw.jsonl
```

脚本生成回复字符数对比图 `response_lengths.png`。如果两组都包含 WildGuard 或 StrongREJECT 评分，还会分别生成 `wildguard_asr.png`、`strongreject_score.png` 并打印均值与差值。可使用 `--output-dir` 指定其他目录，例如 Google Drive。回复长度不等于攻击成功率；小样本结果不宜直接与论文比较。

## 验证

```bash
python -m unittest discover -s tests
```

下载的基准和生成内容可能包含令人不适的文本，应按所在组织的数据处理规则保存。
