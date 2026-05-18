# FFN / FeedForward

## 模块作用

FFN 是 Transformer Block 里的 token 内部加工层。Attention 负责让 token 之间交换信息，FFN 负责对每个 token 当前的 hidden 向量做非线性变换。

它不混合序列维度，只处理每个位置自己的 hidden 表示。

## 为什么这样设计

普通线性层只能做线性变换，表达能力有限。FFN 先升维、做非线性和门控，再降回原 hidden size：

```text
H -> I -> H
```

其中 `H` 是模型主干宽度，`I` 是中间扩展维度。升维给模型更多通道表达复杂特征，降维保证输出还能接回 residual。

## 核心公式

门控 FFN / SwiGLU 风格可以写成：

$$
\mathrm{FFN}(x) = W_{down}(\sigma(W_{gate}x) \odot W_{up}x)
$$

其中：

- $\sigma$ 通常是 SiLU
- $\odot$ 是逐元素相乘
- `gate_proj` 产生门控分支
- `up_proj` 产生内容分支
- `down_proj` 把中间维度压回 hidden size

SiLU 公式：

$$
\mathrm{SiLU}(x) = x \cdot \mathrm{sigmoid}(x)
$$

相比 ReLU 的硬截断，SiLU 更平滑，更适合做门控。

## 在整体架构中的作用

在 Decoder Block 中，FFN 通常位于 Attention 后面：

```text
hidden_states
-> RMSNorm
-> Attention
-> residual add
-> RMSNorm
-> FFN
-> residual add
```

Attention 解决“看哪些 token”，FFN 解决“每个 token 内部如何加工表示”。

## 数据流和 shape

输入：

```text
x: [B, T, H]
```

每一维含义：

- `B`: batch size，一次训练/推理的样本数
- `T`: seq len，当前序列中的 token 数
- `H`: hidden size，每个 token 的表示维度

当前 tensor 表示：每个 token 在当前层的 hidden 表示。

FFN 内部：

```text
up_proj(x):      [B, T, I]
gate_proj(x):    [B, T, I]
activation:      [B, T, I]
multiply:        [B, T, I]
down_proj(...):  [B, T, H]
```

其中 `I = intermediate_size`，表示 FFN 内部扩展后的通道数。

输出：

```text
out: [B, T, H]
```

输出必须回到 `H`，这样才能和 residual 分支相加。

## 这样设计的好处

- 增强非线性表达能力：不仅是线性映射，还能学习复杂特征组合。
- token 内部加工：不改变 token 间关系，只重塑每个 token 的特征。
- 门控更灵活：一条分支生成内容，一条分支控制哪些通道通过。
- residual 友好：输入输出 shape 都是 `[B, T, H]`，方便接入 Transformer Block。

## 容易混淆的点

- FFN 不混合不同 token，混合 token 的是 Attention。
- `intermediate_size` 不是序列长度，而是 hidden 通道扩展维度。
- `up_proj` 和 `gate_proj` 都输出 `[B, T, I]`，两者逐元素相乘要求 shape 完全一致。
- 输出必须通过 `down_proj` 回到 `[B, T, H]`，否则不能 residual add。
- ReLU 和 SiLU 都是激活函数；SiLU 更平滑，形式上自带门控感。

## self 当前核心实现

```python
class FeedForward(nn.Module):
    def __init__(self, args: MokioMindConfig):
        super().__init__()
        if args.intermediate_size is None:
            intermediate_size = int(args.hidden_size * 8 / 3)
            args.intermediate_size = 64 * ((intermediate_size + 63) // 64)

        self.up_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)
        self.gate_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)

        self.dropout = nn.Dropout(args.dropout)
        self.act_fn = ACT2FN[args.hidden_act]

    def forward(self, x: torch.Tensor):
        return self.dropout(
            self.down_proj(self.act_fn(self.up_proj(x)) * self.gate_proj(x))
        )
```
