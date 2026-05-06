# RoPE / YaRN

## 模块作用

RoPE 把位置信息注入到 Attention 的 `q/k` 上，而不是直接加到 token embedding。

在 MiniMind 中它位于 Attention 内部：

```text
hidden_states
-> q_proj / k_proj
-> q_norm / k_norm
-> apply_rotary_pos_emb(q, k)
-> attention score
```

它只作用于 `q/k`，不作用于 `v`。因为 Attention 权重来自 `q @ k^T`，位置关系应该影响“看哪里”，而不是直接改变被取出的内容。

## 核心公式

RoPE 的简化形式：

$$
x' = x \cdot \cos + rotate(x) \cdot \sin
$$

其中：

$$
rotate([x_1, x_2]) = [-x_2, x_1]
$$

YaRN 的核心是对 RoPE 频率做长上下文缩放：

$$
f' = f \cdot (1 - r + \frac{r}{s})
$$

这里 `ramp=r` 是不同频段的平滑过渡系数，`factor=s` 是扩展倍率。

## 数据流和 shape

预计算阶段：

```text
dim = head_dim
freqs:     [head_dim / 2]
t:         [max_position_embeddings]
outer:     [max_position_embeddings, head_dim / 2]
freqs_cos: [max_position_embeddings, head_dim]
freqs_sin: [max_position_embeddings, head_dim]
```

Attention 中：

```text
q:   [B, T, n_heads, head_dim]
k:   [B, T, n_kv_heads, head_dim]
cos: [T, head_dim]
sin: [T, head_dim]
```

`cos.unsqueeze(1)` 后可广播到 q/k：

```text
cos: [T, 1, head_dim]
q':  [B, T, n_heads, head_dim]
k':  [B, T, n_kv_heads, head_dim]
```

RoPE 不改变 shape，只改变 q/k 的数值。

## ref 设计要点

ref 在 `MiniMindModel.__init__` 中预计算 cos/sin，并注册为 buffer：

```python
self.register_buffer("freqs_cos", freqs_cos, persistent=False)
self.register_buffer("freqs_sin", freqs_sin, persistent=False)
```

这样 cos/sin 会跟随模型迁移 device，但不会进入 checkpoint，因为它们可以由 config 重新计算。

推理使用 KV cache 时，ref 会用 `start_pos` 切片：

```python
position_embeddings = (
    self.freqs_cos[start_pos:start_pos + seq_length],
    self.freqs_sin[start_pos:start_pos + seq_length],
)
```

这保证新增 token 使用的是它在完整上下文中的真实位置，而不是每次都从 0 开始。

## self 和 ref 的本质差异

self 已经实现了 RoPE 的核心公式，`rotate_half` 逻辑和 ref 等价。

当前主要差异是工程完整性：

- self 的 `precompute_freqs_cis` 函数签名目前不可运行，需要补默认值写法。
- self 使用了 `Optional`、`math`，但当前文件缺少对应 import。
- self 在 `rope_scaling=None` 时仍会使用 `orig_max`，会触发未定义变量问题。
- self 暂时没有读取 `attention_factor`。
- self 的 `apply_rotary_pos_emb` 没有把输出 `.to(q.dtype)` / `.to(k.dtype)`，混合精度下可能导致 dtype 被提升。
- 当前还没看到和模型主干集成：需要在模型中注册 buffer，并在 forward 中按 `start_pos` 切片。

## 容易写错的点

- RoPE 的 `dim` 应该是 `head_dim`，不是 `hidden_size`。
- `head_dim` 最好是偶数，否则 `rotate_half` 的前后半维语义不对。
- RoPE 只旋转 q/k，不旋转 v。
- KV cache 生成时不能每步都使用位置 0，需要使用 `start_pos`。
- `cos/sin` 的 shape 要和 q/k 的布局匹配；当前 `unsqueeze_dim=1` 适配 `[B, T, heads, D]`。
- YaRN 缩放的是频率 `freqs`，不是直接缩放 q/k。

## self 当前核心代码

```python
def precompute_freqs_cis(dim:int, end:int(32*1024), rope_base, rope_scaling:Optional[dict]=None):
    freqs, attn_factor = (1.0 / (rope_base ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim)), 1.0)

    if rope_scaling is not None:
        orig_max, factor, beta_fast, beta_slow = (
            rope_scaling["original_max_position_embeddings"],
            rope_scaling["factor"],
            rope_scaling["beta_fast"],
            rope_scaling["beta_slow"]
        )

    if end > orig_max:
        inv_dim = lambda b : (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
        low = max(math.floor(inv_dim(beta_fast)), 0)
        high = min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
        ramp = torch.clamp(
            (torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001),
            0,
            1
        )
        freqs = freqs * (1 - ramp + ramp / factor)

    t = torch.arange(end, device=freqs.device).float()
    freqs = torch.outer(t, freqs).float()

    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor

    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))

    return q_embed, k_embed
```
