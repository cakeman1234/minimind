# GQA / Attention

## 模块作用

这一段实现的不是“完整 Transformer”，而是 Decoder Block 里最核心的自注意力部分，重点是：

- 用 `q_proj / k_proj / v_proj` 把输入 hidden state 投影成注意力空间
- 用 `repeat_kv` 支持 `num_key_value_heads < num_attention_heads`
- 在有 KV cache 时，把历史 `k/v` 和当前 `k/v` 拼起来继续做因果注意力

这里的设计本质上是在实现 **GQA (Grouped Query Attention)**：

$$
n_{kv} < n_q
$$

即 Query 头数多，Key/Value 头数少，再把较少的 K/V 头复制给多个 Q 头共享。

## 为什么这样设计

标准多头注意力里，`q/k/v` 头数相同，表达直接，但 K/V cache 的显存和带宽开销也更大。  
GQA 的核心思路是：

- 保留较多的 Query heads，维持“看问题的角度”数量
- 减少 Key/Value heads，降低投影和缓存成本
- 用 `repeat_kv` 把较少的 K/V 头扩成和 Q 一样的头数，方便继续做标准 attention

这使得实现层面仍然可以沿用：

$$
\text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V
$$

只是 `K/V` 的来源先做了一次 head 复用。

## 在整体架构中的作用

它位于一个 Decoder Layer 的 self-attention 子模块里。  
从数据流上看，前面模块提供 `[B, T, H]` 的 hidden states，这里负责把它变成“当前 token 从历史 token 聚合出来的新上下文表示”，最后再回到 `[B, T, H]`，供 residual 和后续 MLP 使用。

可以把它理解成：

```text
hidden_states
-> q/k/v 投影
-> reshape 成多头
-> q/k 注入 RoPE
-> 拼接 KV cache
-> repeat_kv
-> attention
-> 合并 heads
-> o_proj
```

## 关键数据流和 shape

### 1. 输入 hidden states

```text
x: [B, T, H]
```

- `B`: batch size，当前 batch 里有多少条样本
- `T`: 当前这次 forward 传入多少个 token
- `H`: 每个 token 的 hidden size
- 这个 tensor 表示“当前层收到的 token 表示”

### 2. 线性投影后的扁平 q/k/v

```text
xq: [B, T, n_q * D]
xk: [B, T, n_kv * D]
xv: [B, T, n_kv * D]
```

- `n_q`: query 头数
- `n_kv`: key/value 头数
- `D`: 单个 head 的维度
- 此时最后一维还是“所有 head 拼接后的大向量”，head 结构还没有显式展开

### 3. reshape 成多头表示

```text
q: [B, T, n_q, D]
k: [B, T, n_kv, D]
v: [B, T, n_kv, D]
```

这里 tensor 的语义已经变了：

- `q` 表示每个 token 在每个 query head 上发出的查询
- `k` 表示每个 token 在每个 kv head 上提供的索引特征
- `v` 表示每个 token 在每个 kv head 上携带的内容

### 4. RoPE 之前的位置切片

当前实现里，`cos/sin` 的切片职责已经上移到了模型层。  
也就是说，`Attention` 拿到的 `position_embedding` 不是整张 `[max_seq_len, D]` 表，而是当前这次 forward 已经对齐好的局部片段：

```text
cos: [T, D]
sin: [T, D]
```

这里：

- `T` 是当前这次 query 的 token 数
- `D` 是单 head 维度
- 第 `i` 行表示“当前第 `i` 个 token 的真实全局位置对应的旋转参数”

这一点很重要，因为有 KV cache 时，位置偏移应当只计算一次。  
当前设计里，**模型层负责按 `start_pos:start_pos+seq_len` 切片，Attention 只消费切好的结果**。

### 5. RoPE 之后

```text
q: [B, T, n_q, D]
k: [B, T, n_kv, D]
```

shape 不变，但数值语义变成“带位置信息的 q/k”。  
这里最容易混淆的是：RoPE 改的是 `q/k` 的坐标，不改 `v`。

### 6. 拼接 KV cache

假设历史缓存长度是 `P`：

```text
k: [B, P + T, n_kv, D]
v: [B, P + T, n_kv, D]
```

- `P` 是 `past_len`
- 当前 query 长度仍然是 `T`
- 但 key/value 长度已经变成 `P + T`

这一步之后，当前 attention 的 query 和 key/value 不再是同长度，所以后面的 `scores` 一般会从方阵变成矩形。

### 7. repeat_kv

```text
repeat_kv(k): [B, P + T, n_q, D]
repeat_kv(v): [B, P + T, n_q, D]
```

它不是重新计算新的 K/V，而是沿着“kv head 这一维”复制：

$$
n_{\text{repeat}} = \frac{n_q}{n_{kv}}
$$

这样每个 query head 都能拿到一份可对齐的 `k/v`。

### 8. 转置到 attention 计算布局

```text
q: [B, n_q, T, D]
k: [B, n_q, P + T, D]
v: [B, n_q, P + T, D]
```

此时：

- 第 2 维是 head
- 第 3 维是 token 位置
- 最后一维是单 head 特征

### 9. attention score / prob / output

```text
scores: [B, n_q, T, P + T]
probs:  [B, n_q, T, P + T]
out:    [B, n_q, T, D]
```

- `scores` 表示每个 query token 对所有 key token 的原始打分
- `probs` 表示 softmax 后的注意力分布
- `out` 表示按注意力权重汇聚后的上下文向量

如果没有 KV cache，那么 `P = 0`，`scores` 就退化回：

```text
[B, n_q, T, T]
```

### 10. 合并 heads

```text
out.transpose(1, 2): [B, T, n_q, D]
reshape:             [B, T, n_q * D]
o_proj:              [B, T, H]
```

这里 tensor 重新回到模型 hidden space，可以接 residual。

## 这样设计的好处

- K/V 头更少，KV cache 更省显存
- attention 主公式不需要改，工程实现仍然清晰
- `repeat_kv` 把“结构压缩”和“标准 attention 计算”解耦了
- 配合 KV cache 时，生成阶段能避免重复计算历史 token 的 `k/v`

## 容易混淆的点

- `xq/xk/xv` 和 `q/k/v` 不是一回事  
  前者是扁平投影张量，后者是拆头后的注意力张量。进入 attention 几何计算后，应只操作 `q/k/v`。

- `seq_len` 和 `past_len` 的语义不同  
  `seq_len` 是这次新输入 token 数，`past_len` 是缓存里历史 token 数。  
  有 cache 时，`q_len = seq_len`，但 `kv_len = past_len + seq_len`。

- 有 KV cache 时，`scores` 不再是方阵  
  它会从 `[T, T]` 变成 `[T, P + T]`，所以 causal mask 也必须跟着变成矩形。

- `repeat_kv` 复制的是 `kv head`，不是复制 token，也不是复制 hidden size 维度

- `past_len` 影响的是“模型层如何切 RoPE 位置表”，而不是每层 Attention 都自己再切一次  
  否则会出现位置二次偏移

## 模型级位置调度

当前实现已经进入“模型主干 + 多层 block”阶段，因此 RoPE 的位置调度不再只属于 `Attention`，而是模型级逻辑的一部分：

```text
MokioMind.forward
-> 根据 past_key_values 计算 start_pos
-> 从 freqs_cos / freqs_sin 中切出当前 [T, D]
-> 传给每一层 block
-> block 再传给 self_attn
```

这层设计的意义是：

- `MokioMind` 负责管理全局位置
- `Attention` 负责消费已经对齐好的位置参数

职责分开之后，`Attention` 就不需要再自己推导“当前位置在整段序列里是多少”。

## 当前 self 中的核心实现代码

```python
def repeat_kv(x:torch.Tensor, num_repeats:int):
    bs, slen, num_key_value_heads, head_dim = x.shape

    if num_repeats == 1:
        return x

    return x[:, :, :, None, :].expand(
        bs, slen, num_key_value_heads, num_repeats, head_dim
    ).reshape(bs, slen, num_key_value_heads * num_repeats, head_dim)


class Attention(nn.Module):
    def __init__(self, args:__module__):
        super().__init__()

        self.num_key_value_heads = (
            args.num_key_value_heads
            if args.num_key_value_heads is not None
            else args.num_attention_heads
        )

        assert args.num_attention_heads % self.num_key_value_heads == 0

        self.n_local_heads = args.num_attention_heads
        self.head_dim = args.hidden_size // args.num_attention_heads
        self.num_repeats = self.n_local_heads // self.num_key_value_heads

        self.q_proj = nn.Linear(args.hidden_size, args.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(args.num_attention_heads * self.head_dim, args.hidden_size, bias=False)

        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention") and args.flash_attention

    def forward(
        self,
        x: torch.Tensor,
        position_embedding: Tuple[torch.Tensor, torch.Tensor],
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache=False,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        bsz, seq_len, _ = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        q = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        k = xk.view(bsz, seq_len, self.num_key_value_heads, self.head_dim)
        v = xv.view(bsz, seq_len, self.num_key_value_heads, self.head_dim)

        cos, sin = position_embedding
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=1)
            v = torch.cat([past_key_value[1], v], dim=1)

        past_kv = (k, v) if use_cache else None

        q = q.transpose(1, 2)
        k = repeat_kv(k, self.num_repeats).transpose(1, 2)
        v = repeat_kv(v, self.num_repeats).transpose(1, 2)

        if self.flash and seq_len > 1 and (attention_mask is None or torch.all(attention_mask == 1)):
            attn_mask = (
                None
                if attention_mask is None
                else attention_mask.view(bsz, 1, 1, -1).expand(bsz, self.n_local_heads, seq_len, -1).bool()
            )
            output = F.scaled_dot_product_attention(
                q, k, v, attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True
            )
        else:
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

            q_len = q.size(-2)
            kv_len = k.size(-2)
            past_len = kv_len - q_len

            causal_mask = torch.triu(
                torch.full((q_len, kv_len), float("-inf"), device=scores.device, dtype=scores.dtype),
                diagonal=past_len + 1,
            )
            scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)

            if attention_mask is not None:
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
                scores = scores + extended_attention_mask

            probs = F.softmax(scores, dim=-1).type_as(q)
            probs = self.attn_dropout(probs)
            output = probs @ v

        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv
```
