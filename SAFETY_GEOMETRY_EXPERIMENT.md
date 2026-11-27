# 安全子空间与预测方向实验

本实验检验两个问题。安全相关表示是否集中在少数可泛化方向，以及换语言产生的位移通过哪些方向改变候选 token 的相对 logit。当前代码完成输入准备、模型状态提取、交叉验证分析和数学恒等式测试，运行时读取已有 Luna 标签。

## 预先固定的设计

- 使用同一批请求的 en、zh-CN、ja、ar 结果，以及按 group、language、id 连接的 GPT-6-Luna success 标签。true 为危险，false 为安全。缺失、重复或不确定标签均报错。
- Qwen base 与 Qwen-Instruct 分别提取。两次运行指定同一个 Instruct tokenizer 和聊天模板，程序核对词表一致。已有多语言回答的原始生成格式有所不同，这一因素须随结果报告。
- 记录请求末尾的 prompt state，以及读入固定 8 个回答 token 后的 answer state。默认训练空间来自英文 answer state。空回答与长度不足的回答仍保留 prompt state，answer 数组用零向量占位并记录 `answer_valid=false`。拟合和配对分析均排除无效 answer state，并保留其他读出。完整回答标签用于标记前缀，该前缀本身未必包含危险内容。
- 所有表示位于最终归一化的 LM-head 输入坐标。这样输出头的两个候选 token logit 差是线性函数。代码核对输出头公式与实际模型 logits 的误差，支持在 MUSA 上显式加载 float16。
- 固定五折，以原始请求分折。一个请求的全部语言和全部来源模型一起留出。训练集中按请求、类别聚合多个来源回答，随后匹配安全和危险样本数。
- 预先报告 k=1、2、4、8。每折每类拟合中心化 SVD，保存均值和正交基。程序要求样本数和数值秩足够，避免静默改变 k。

## 三种读出

**prompt 读出**只更换用户输入语言。用 answer state 拟合的空间来测量 prompt state 属于跨位置迁移分析，距离还可能包含记录位置差异。以 `--fit-position prompt` 重跑可检查同位置结果。该设置按照完整回答的结果给请求状态打标签，同一请求不同来源回答的标签可能不同，需结合 `--fit-groups instruct` 或 `--fit-groups base` 分别分析。Instruct 英文只有 2 个危险例，当前数据不足以支持其独立的高秩危险空间。

**answer 读出**同时改变提示语言与已有回答前缀。它描述自然生成轨迹的差异，包含输出内容、措辞和翻译长度的影响。

**anchored_margin 读出**先在一个请求的英文回答中找一对异标签回答，取首次不同 token 之前的共同前缀。对所有语言都强制输入同一段英文共同前缀，并固定这两个候选 token。此时语言变化只来自用户输入，候选词和回答前缀完全一致。只有英文来源回答标签不同且存在不同 token 的请求进入这一分析，输出 summary 明确记录此选择条件。

两个候选来自安全和危险完整回答。它们的相对 logit 是一个候选选择代理量。其正负与实际续写安全性的对应仍需通过独立续写、StrongREJECT 和人工复核验证。共同英文前缀也会影响生成语境，anchored_margin 的行为应与原始多语言生成结果分别解释。

## 数学测量

采用 samples-by-features 矩阵，中心化后做 `H = U S Vt`，取 `B = Vt[:k].T` 作为特征空间正交基。投影以 `B @ (B.T @ h)` 计算，无需构建稠密 d×d 矩阵。

保存每类的奇异值、参与率 `(sum(s²))² / sum(s⁴)`、熵有效秩、解释方差，以及安全和危险子空间的主角度和重叠度。这些是当前样本规模下的经验维数，样本匹配只能控制样本量造成的秩上限差异。扩大独立请求数并检验谱是否稳定，才有依据讨论总体维数。

对同一请求的英语状态 h_en 和另一语言状态 h_l，定义 Δ=h_l−h_en。保存到安全／危险仿射空间的距离、投影能量、空间外位移比例以及距离变化。

对固定候选定义 g=W_safe−W_danger，b=b_safe−b_danger。安全候选的相对 margin 为 m(h)=g·h+b。程序验证以下恒等式并记录数值误差。

```
Δm = g·PΔ + g·(I−P)Δ
R(g) = ||Pg||² / ||g||²
m_projected(h) = g·[μ + P(h−μ)] + b
```

`projected_margin_other` 是输出头上的表示投影反事实。它可能处于模型自然激活分布以外，解释时需要结合真实续写行为。

每折、每个 k 抽取 20 个随机正交基作为参照，保持相同类别中心。记录随机对照的平均距离变化和方向保留率。各向同性参考值分别为 `E[outside_fraction]=(d−k)/d` 和 `E[R]=k/d`，高维条件下外部位移比例大本身很常见。

按实际生成来源 `--label-group instruct` 的英／其他语言标签，把留出请求分为 safe_to_safe、safe_to_danger、danger_to_safe、danger_to_danger。每组输出请求数、均值和按请求 bootstrap 区间。区间条件于当前拟合折和基，只有 1–2 个请求的组会退化，代码不把它转换成显著性结论。

## 服务器运行

依赖复用 requirements.txt。以下命令在仓库根目录运行。数据目录应包含 outputs/base-en.jsonl 等八个结果文件，以及 llm-judge.jsonl。

```bash
data=/root/autodl-tmp/multilingual-jbb-20261126
inputs=()
for group in base instruct; do
  for language in en zh-CN ja ar; do
    inputs+=(--responses "$group=$data/outputs/$group-$language.jsonl")
  done
done
python safety_geometry_experiment.py prepare "${inputs[@]}" \
  --judgments /path/to/llm-judge.jsonl \
  --output outputs/safety-geometry/prepared.jsonl
```

在 MTT S4000 上使用装有 torch_musa 的 Python，并指定缓存中实际存在的 snapshot 路径。base 和 Instruct 按顺序运行，以避免设备争用。以下模板中的两个路径需要替换为服务器缓存路径。

```bash
python safety_geometry_experiment.py extract \
  --input outputs/safety-geometry/prepared.jsonl \
  --model /path/to/Qwen2.5-3B-Instruct/snapshot \
  --template-tokenizer /path/to/Qwen2.5-3B-Instruct/snapshot \
  --device musa --dtype float16 --offline --answer-tokens 8 \
  --output outputs/safety-geometry/instruct.npz

python safety_geometry_experiment.py extract \
  --input outputs/safety-geometry/prepared.jsonl \
  --model /path/to/Qwen2.5-3B/snapshot \
  --template-tokenizer /path/to/Qwen2.5-3B-Instruct/snapshot \
  --device musa --dtype float16 --offline --answer-tokens 8 \
  --output outputs/safety-geometry/base.npz

OPENBLAS_NUM_THREADS=1 python safety_geometry_analysis.py \
  --input outputs/safety-geometry/instruct.npz \
  --output-dir outputs/safety-geometry/instruct-analysis \
  --label-group instruct --ranks 1 2 4 8 --random-directions 20

OPENBLAS_NUM_THREADS=1 python safety_geometry_analysis.py \
  --input outputs/safety-geometry/base.npz \
  --output-dir outputs/safety-geometry/base-analysis \
  --label-group instruct --ranks 1 2 4 8 --random-directions 20
```

两次分析中的 `label-group instruct` 固定为相同的实际回答标签，以比较两个模型对同一行为变化的表示。要研究 base 自己的行为翻转，另选新输出目录并指定 `--label-group base`。不同模型的子空间主角度仅在各自坐标内计算。

## 输出与判断

- `prepared.jsonl` 保存输入、回答和 Luna 标签。
- 每个模型的 NPZ 保存 prompt、answer 和 anchored 状态、候选方向、实际模型 margin、token ID、共同前缀和提取配置。配置包括输入 SHA-256。
- `bases.npz` 保存每折每类均值和最大秩基，可据此构造各个 k 的 P。
- `spectra.jsonl` 保存谱、有效秩、主角度和训练／留出请求 ID。
- `pairs.jsonl` 保存每个留出请求的语言位移及 margin 分解。
- `summary.json` 按语言、行为翻转、子空间类别、秩和读出方式汇总。

符合“安全预测分量保留”的结果应同时表现为较高的方向保留率、翻译后 margin 保持安全候选优势，以及实际生成维持安全。符合“子空间内部跨界”的结果应表现为空间内贡献主导 margin 的符号变化。符合“空间外贡献导致变化”的结果应表现为空间外贡献与行为翻转共同变化，并超过随机基参照。安全样本有效秩下降仅回答压缩问题的一部分，需要与这些行为关联证据一起解释。

数学测试可在装有 NumPy 的环境运行。

```bash
python -m unittest discover -s tests -p test_safety_geometry.py -v
```
