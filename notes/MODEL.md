# Backbone / CausalLM

## 模块作用

这一轮增量把代码从“单层 block 级别”推进到了“模型主干 + 语言模型包装器”级别，核心有两部分：

- `MokioMind`: 负责从 `input_ids` 生成最终 `hidden_states`
- `MokioMindForCausalLM`: 负责把 `hidden_states` 投影到词表维度，得到 `logits`

可以把它理解成两条连续的数据流：

```text
input_ids
-> MokioMind
-> hidden_states
-> MokioMindForCausalLM / lm_head
-> logits
```

## 为什么这样设计

这是一种典型的“backbone + task head”拆分：

- backbone 只负责表示学习
- task head 负责把表示变成具体任务输出

对于因果语言模型，这个任务输出就是：

$$
\text{logits} \in \mathbb{R}^{B \times T \times V}
$$

其中：

- `B`: batch size
- `T`: 位置数
- `V`: vocab size

这比把 `lm_head` 直接写死在 backbone 里更清晰，也更方便以后把 backbone 拿去接别的头。

## 在整体架构中的作用

当前实现里的 `MokioMind` 主干路径是：

```text
input_ids
-> embed_tokens
-> dropout
-> N x MokioMindBlock
-> final RMSNorm
-> hidden_states
```

而 `MokioMindForCausalLM` 负责补上：

```text
hidden_states
-> lm_head
-> logits
-> 输出结构化结果
```

所以这次改动的意义不是新增某个局部算子，而是把模型整体推进到“可以产出词表预测”的阶段。

## 这样设计的好处

- `MokioMind` 和 `MokioMindForCausalLM` 职责分离
- 模型主干可以只输出 hidden states，便于调试或迁移到别的任务
- `lm_head` 可以单独控制是否与 embedding 权重共享
- `Logits_to_keep` 这类切片逻辑可以只放在外层 wrapper，不污染 backbone

## 关键数据流和 shape 变化

### 1. 输入 ids

```text
input_ids: [B, T]
```

- `B`: batch size
- `T`: 当前这次 forward 处理的 token 数
- 当前 tensor 表示“离散 token 索引”

### 2. Embedding 后

```text
hidden_states: [B, T, H]
```

- `H`: hidden size
- 这个 tensor 表示“每个 token 的连续向量表示”

### 3. 经过 N 层 block 后

```text
hidden_states: [B, T, H]
```

shape 不变，但语义变成“已经融合了上下文的最后一层表示”。

### 4. 模型级 RoPE 位置切片

当前实现中，模型层会先按全局位置切好：

```text
freqs_cos / freqs_sin: [max_seq_len, D]
position_embedding:    [T, D]
```

其中：

- `D` 是单 head 维度
- `position_embedding` 表示“当前这次输入 token 对应的局部位置表”

这一步之后，block / attention 直接消费切好的位置参数，不再自己重复切片。

### 5. `lm_head` 之后

```text
hidden_states: [B, T_keep, H]
logits:        [B, T_keep, V]
```

这里：

- `V` 是 vocab size
- `T_keep` 是最终保留下来计算 logits 的位置数

`logits` 的语义不是概率，而是：

> 每个位置对整个词表的原始打分

后续如果需要概率，通常再做 softmax。

### 6. `slice_indices` / `Logits_to_keep`

当前实现里，外层 wrapper 不是一定对所有 `T` 个位置都算输出，而是允许只保留一部分位置：

```python
slice_indices = slice(-Logits_to_keep, None)
```

这表示：

- 若 `Logits_to_keep = 1`，只保留最后一个位置
- 若 `Logits_to_keep = 3`，只保留最后三个位置

因此 `T_keep` 不一定等于原始 `T`。

## 容易混淆的点

- `hidden_states` 不是词表输出  
  它是模型内部表示，真正的词表打分是 `logits`。

- `logits` 不是概率  
  它是未归一化分数；概率需要再经过 softmax。

- `MokioMind` 和 `MokioMindForCausalLM` 的职责不同  
  前者负责 backbone，后者负责 task head 和结构化输出。

- 当前位置的 RoPE 切片现在是模型层职责  
  Attention 不应该再根据 `past_len` 重复做位置偏移。

## 当前 self 中的核心实现代码

```python
class MokioMind(nn.Module):
    def __init__(self, config:MokioMindConfig):
        super().__init__()

        self.vocab_size, self.num_hidden_layers = (
            config.vocab_size,
            config.num_hidden_layers,
        )

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([MokioMindBlock(i, config) for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=config.hidden_size // config.num_attention_heads,
            end=config.max_position_embeddings,
            rope_base=config.rope_theta,
            rope_scaling=config.rope_scaling,
        )

        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        use_cache=False,
        **kwargs
    ):
        batch_size, seq_len = input_ids.shape

        if hasattr(past_key_values, "layers"):
            past_key_values = None

        past_key_values = past_key_values or [None] * len(self.layers)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        hidden_states = self.dropout(self.embed_tokens(input_ids))

        position_embedding = (
            self.freqs_cos[start_pos:start_pos + seq_len],
            self.freqs_sin[start_pos:start_pos + seq_len],
        )

        presents = []
        for layer_idx, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            hidden_states, present_key_value = layer(
                hidden_states, position_embedding, past_key_value, use_cache, attention_mask
            )
            presents.append(present_key_value)

        hidden_states = self.norm(hidden_states)
        return hidden_states, presents


class MokioMindForCausalLM(PretrainedModel, GenerationMixin):
    config_class = MokioMindConfig

    def __init__(self, config:MokioMindConfig):
        self.config = config
        super().__init__(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 权重共享
        self.model.embed_tokens.weight = self.lm_head.weight

        self.OUT = CausalLMOutputWithPast()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        use_cache=False,
        Logits_to_keep: Union[int, torch.Tensor] = 0,
        **args
    ):
        hideen_states, past_key_values = self.model(
            input_ids, attention_mask, past_key_values, use_cache, **args
        )

        slice_indices = (
            slice(-Logits_to_keep, None)
            if isinstance(Logits_to_keep, int)
            else Logits_to_keep
        )

        logits = self.lm_head(hideen_states[:, slice_indices, :])
        self.OUT.__setitem__("logits", logits)
        self.OUT.__setitem__("past_key_values", past_key_values)
        self.OUT.__setitem__("hidden_states", hideen_states)

        return self.OUT
```
