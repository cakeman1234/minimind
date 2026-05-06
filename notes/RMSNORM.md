# RMSNorm

## 模块作用

RMSNorm 用来稳定每个 token 的 hidden 向量尺度。它出现在 Decoder Block 的 PreNorm 位置：

- Attention 前：`RMSNorm -> Attention -> residual`
- MLP 前：`RMSNorm -> MLP -> residual`

本质是只沿 hidden 维归一化，不改变 batch、seq 结构。

## 数据流和 shape

输入通常是：

```text
x: [B, T, H]
```

RMSNorm 内部：

```text
x.pow(2).mean(-1, keepdim=True): [B, T, 1]
rsqrt(...):                      [B, T, 1]
weight:                          [H]
output:                          [B, T, H]
```

公式：

$$
y = x \cdot \frac{1}{\sqrt{\mathrm{mean}(x^2) + \epsilon}} \cdot w
$$

这里的 `mean` 只在最后一维 `H` 上做。

## ref 核心设计

参考实现：

```python
self.weight = nn.Parameter(torch.ones(dim))

def norm(self, x):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

def forward(self, x):
    return (self.weight * self.norm(x.float())).type_as(x)
```

关键点：

- `weight` 是可训练参数，shape 是 `[H]`
- `keepdim=True` 保证 `[B, T, 1]` 可以广播回 `[B, T, H]`
- `torch.rsqrt` 表示 `1 / sqrt(...)`
- `x.float()` 用 float32 做归一化，更适合混合精度训练
- `.type_as(x)` 再转回输入 dtype，保持后续计算 dtype 一致

## ref 和 self 的区别

本次检查时，`minimind_self/model/model.py` 还是空文件，所以 self 侧暂时没有 RMSNorm 实现。

这意味着后续如果 DecoderLayer 或 Attention 引用 `RMSNorm`，会直接缺少定义。最小实现需要至少包含：

```python
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x_float = x.float()
        normed = x_float * torch.rsqrt(x_float.pow(2).mean(-1, keepdim=True) + self.eps)
        return (normed * self.weight).type_as(x)
```

## 容易混淆的点

- RMSNorm 不是 LayerNorm：RMSNorm 不减均值，只控制向量尺度。
- `labels`、`seq_len`、`batch` 都和 RMSNorm 无关，它只处理 hidden 维。
- `weight` 必须是 `nn.Parameter`，否则优化器不会更新。
- 不要漏掉 `keepdim=True`，否则 shape 会变成 `[B, T]`，广播容易出错。
- ref 类默认 `eps=1e-5`，但 MiniMind 实际创建时通常传入 config 里的 `rms_norm_eps=1e-6`。
