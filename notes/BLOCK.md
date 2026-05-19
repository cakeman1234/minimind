# Decoder Block

## 模块作用

这次新增的 `MokioMindBlock` 做的事情很明确：把已经实现好的 `RMSNorm`、`Attention`、`FeedForward` 按 Decoder-only Transformer 的标准顺序接起来，形成一层可堆叠的 block。

它对应的主路径是：

```text
hidden_states
-> input_layernorm
-> self_attn
-> residual add
-> post_attention_layernorm
-> mlp
-> residual add
```

本质上，它不是新算子，而是“层级组装逻辑”。

## 为什么这样设计

这里采用的是 **Pre-Norm + Residual** 结构：

$$
h_1 = x + \mathrm{Attn}(\mathrm{Norm}(x))
$$

$$
h_2 = h_1 + \mathrm{FFN}(\mathrm{Norm}(h_1))
$$

这样设计的核心考虑是：

- attention 和 FFN 都接收更稳定的输入分布
- 残差路径始终保留原始信息
- 多层堆叠时更容易训练

## 在整体架构中的作用

`MokioMindBlock` 是模型“单层 Transformer”的定义。  
当前 `self` 里的主干已经是：

```text
Embedding
-> Block 1
-> Block 2
-> ...
-> Final Norm
```

也就是说，这次实现到的是 **backbone / hidden states 主干**，还没有把 `lm_head` 接上。  
所以这层的作用不是单独提升能力，而是规定“每一层如何做上下文建模 + token 内部变换”。

## 这样设计的好处

- Block 输入输出 shape 一致，便于堆叠
- `Attention` 负责跨 token 信息交互
- `FeedForward` 负责 token 内部通道变换
- `past_key_value` 只穿过 attention 路径，职责边界清楚

## 关键数据流和 shape 变化

### 1. Block 输入

```text
hidden_states: [B, T, H]
```

- `B`: batch size
- `T`: 当前这次 forward 处理的 token 数
- `H`: hidden size
- 当前 tensor 表示“上一层输出的 token 表示”

### 2. Attention 前的 RMSNorm

```text
input_layernorm(hidden_states): [B, T, H]
```

shape 不变。  
这一步只是把每个 token 的 hidden 向量按最后一维归一化，让 attention 看到更稳定的输入。

### 3. Self-Attention 输出

```text
self_attn(...): [B, T, H]
present_key_value:
    k: [B, P + T, n_kv, D]
    v: [B, P + T, n_kv, D]
```

这里：

- 输出的 `[B, T, H]` 表示“融合了上下文后的 token 表示”
- `P` 是 `past_len`
- `n_kv` 是 key/value 头数
- `D` 是单 head 维度

如果没有 KV cache，那么 `P = 0`。

### 4. 第一次 residual add

```text
residual + attn_output: [B, T, H]
```

这一行的语义很重要：  
不是替换原始输入，而是把“原始 token 表示”和“attention 更新量”叠加起来。

### 5. MLP 前的 RMSNorm

```text
post_attention_layernorm(hidden_states): [B, T, H]
```

依然不改 shape。  
这里的 tensor 表示“已经过 attention + residual 的层内表示”，接下来交给 FFN 做按 token 独立的通道变换。

### 6. FeedForward 输出

```text
mlp(...): [B, T, H]
```

虽然 FFN 内部通常会先升维到 `intermediate_size`，再降回 `H`，但对 block 外部来说，输入输出仍然保持 `[B, T, H]`。

### 7. 第二次 residual add

```text
hidden_states + mlp_output: [B, T, H]
```

这一步之后得到 block 最终输出。  
所以整个 block 的一个核心性质是：

- 序列长度 `T` 不变
- hidden size `H` 不变
- 变化的是 token 表示内容，不是张量外形

## 容易混淆的点

- 这个 block 有两次 residual，不是一次  
  一次包 `self_attn`，一次包 `mlp`。

- `position_embedding` 不是 block 自己生成的  
  它只是被原样传给 `self_attn`。

- block 自己不处理 KV cache 细节  
  cache 的生成和更新都发生在 `self_attn` 内部，block 只负责向上传递 `present_key_value`。

- `Attention` 负责跨 token 建模，`FFN` 负责单 token 通道变换  
  两者虽然输入输出 shape 一样，但做的事情完全不同。

## 当前 self 中的核心实现代码

```python
class MokioMindBlock(nn.Module):
    def __init__(self, Layer_id:int, config:MokioMindConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.self_attn = Attention(config)

        self.lay_id = Layer_id
        self.input_layernorm = RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForward(config)

    def forward(self, hidden_states, postion_embedding, past_key_value=None, use_cache=False, attention_mask=None):
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states),
            postion_embedding,
            past_key_value,
            use_cache,
            attention_mask
        )

        hidden_states = residual + hidden_states
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))

        return hidden_states, present_key_value
```
