import math
from typing import Optional, Tuple

from transformers import PretrainedConfig


class MokioMindConfig(PretrainedConfig):
    model_type = "mokiomind"

    def __init__(
        self,
        dropout: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_act: str = "silu",
        hidden_size: int = 512,
        intermediate_size: int = None,
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 8,
        num_key_value_heads: int = 2,
        vocab_size: int = 6400,
        rms_norm_eps: float = 1e-05,
        rope_theta: int = 1000000,
        inference_rope_scaling: bool = False,
        flash_attention: bool = True,
        ############ MoE ############
        use_moe: bool = False,
        num_experts_per_tok: int = 2,
        n_routed_experts: int = 4,
        n_shared_experts: int = 1,
        scoring_func: str = "softmax",
        aux_loss_alpha: float = 0.01,
        seq_aux: bool = True,
        norm_topk_prob: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        self.flash_attention = flash_attention
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob
        self.aux_loss_alpha = aux_loss_alpha
        self.scoring_func = scoring_func

        self.rope_scaling = (
            {
                "beta_fast": 32,
                "beta_slow": 1,
                "factor": 16,
                "original_max_position_embeddings": 2048,
                "attention_factor": 1.0,
                "type": "yarn",
            }
            if self.inference_rope_scaling
            else None
        )


import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers.activations import ACT2FN

# -------------------实现rmsnorm—-------------
class RMSNorm(nn.Module):
    
    def __init__(self, dim:int, eps:float=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * x
    
    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x)

# ------------------实现yarn-------------------
def precompute_freqs_cis(dim:int, end:int(32 * 1024), rope_base, rope_scaling:Optional[dict]=None):
    # 初始化rope频率
    freqs, attn_factor = (1.0 / (rope_base ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim)), 1.0)

    if rope_scaling is not None:
        # 使用超参数
        orig_max, factor, beta_fast, beta_slow = (rope_scaling["original_max_position_embeddings"], 
                                                  rope_scaling["factor"], 
                                                  rope_scaling["beta_fast"], 
                                                  rope_scaling["beta_slow"]
    )
        
    # 推理长度大于训练长度， 使用缩放
    if end > orig_max:
        # 计算波长b到i的映射
        inv_dim = lambda b : (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))

        # 划分高低维度， low : 不需要缩放的高频部分， high : 需要缩放的低频部分
        low = max(math.floor(inv_dim(beta_fast)), 0)
        high = min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)

        # 计算缩放因子
        # low之前ramp为0， high之后ramp为1， 之间线性变化
        ramp = torch.clamp(
            (torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001),
            0,
            1
        )

        # if ramp == 0： 高频， 系数为1， 原频率不变
        # if ramp == 1: 低频， 系数为1/factor， 对频率进行线性插值缩放
        # else: 平滑过渡
        freqs = freqs * (1 - ramp + ramp / factor)

    # 根据end， 生成位置索引t
    t = torch.arange(end, device=freqs.device).float()

    # 计算外积， 得到每个位置的旋转角度
    freqs = torch.outer(t, freqs).float()

    freqs_cos = (
        torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    )
    freqs_sin = (
        torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    )

    return freqs_cos, freqs_sin

# -----------------实现rope-------------------
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):

    # 二维旋转公式：[a, b] -> [-b, a]
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)
    
    # x_rorated = x * cos + rotate_half(x) * sin
    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
    
    return q_embed, k_embed

def repeat_kv(x:torch.Tensor, num_repeats:int):
    # 获取维度
    bs, slen, num_key_value_heads, head_dim = x.shape

    if num_repeats == 1:
        return x
    
    # x[:, :, :, None, :].shape -> (bs, slen, num_key_value_heads, 1, head_dim)
    return x[:, :, :, None, :].expand(bs, slen, num_key_value_heads, num_repeats, head_dim).reshape(bs, slen, num_key_value_heads * num_repeats, head_dim)

class Attention(nn.Module):
    def __init__(self, args:__module__):
        super().__init__()

        self.num_key_value_heads = args.num_key_value_heads if args.num_key_value_heads is not None else args.num_attention_heads
        
        assert args.num_attention_heads % self.num_key_value_heads == 0, "num_attention_heads must be divisible by num_key_value_heads"

        self.n_local_heads = args.num_attention_heads 
        self.head_dim = args.hidden_size // args.num_attention_heads
        self.num_repeats = self.n_local_heads // self.num_key_value_heads

        # 投影层
        self.q_proj = nn.Linear(args.hidden_size, args.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(args.num_attention_heads * self.head_dim, args.hidden_size, bias=False)

        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout

        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attention

    def forward(self, x:torch.Tensor, position_embedding:Tuple[torch.Tensor, torch.Tensor], past_key_value:Optional[Tuple[torch.Tensor, torch.Tensor]]=None, use_cache=False, attention_mask:Optional[torch.Tensor]=None):
        # 线性投影得到q, k, v
        bsz, seq_len, _ = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # 把输入拆分成多个头
        q = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        k = xk.view(bsz, seq_len, self.num_key_value_heads, self.head_dim)
        v = xv.view(bsz, seq_len, self.num_key_value_heads, self.head_dim)

        # q, k应用rope位置编码
        # past_len : 已经计算过的token数
        # seq_len : 当前输入的token数
        # 在有kvcache时， rope位置为past_len到past_len + seq_len
        past_len = 0 if past_key_value is None else past_key_value[0].shape[1]

        cos, sin = position_embedding
        cos = cos[past_len:past_len + seq_len]
        sin = sin[past_len:past_len + seq_len]
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # 拼接kvcache
        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=1)
            v = torch.cat([past_key_value[1], v], dim=1)

        past_kv = (k, v) if use_cache else None

        # 把 kv head repeat 到 q head的数量
        q = q.transpose(1, 2)
        k = repeat_kv(k, self.num_repeats).transpose(1, 2)
        v = repeat_kv(v, self.num_repeats).transpose(1, 2)

        # 进行attention计算
        if self.flash and seq_len > 1 and (attention_mask is None or torch.all(attention_mask == 1)):
            # 没有kvcache, scores是[T, T]， 直接用flash attention， flash attention会自动处理causal mask
            attn_mask = (None if attention_mask is None else attention_mask.view(bsz, 1, 1, -1).expand(bsz, self.n_local_heads, seq_len, -1).bool())

            output = F.scaled_dot_product_attention(q, k, v, attn_mask, dropout_p=self.dropout if self.training else 0.0, is_causal = True)
        else:
            # 有kvcache， scores会变成[T, past_len + T]
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

            q_len = q.size(-2)
            kv_len = k.size(-2)
            past_len = kv_len - q_len

            causal_mask = torch.triu(
                torch.full(
                    (q_len, kv_len),
                    float("-inf"),
                    device=scores.device,
                    dtype=scores.dtype,
                ),
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

        # output shape -> (bsz, n_local_heads, seq_len, head_dim) -> (bsz, seq_len, n_local_heads * head_dim)
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))

        return output, past_kv
        
    
class FeedForward(nn.Module):
    def __init__(self, args:MokioMindConfig):
        super().__init__()
        if args.intermediate_size is None:
            intermediate_size = int(args.hidden_size * 8 / 3)
            args.intermediate_size = 64 * ((intermediate_size + 63) // 64)  # 向上取整到64的倍数
        
        # 升维
        self.up_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        # 降维
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)
        # 门控
        self.gate_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)

        self.dropout = nn.Dropout(args.dropout)
        self.act_fn = ACT2FN[args.hidden_act]

    def forward(self, x:torch.Tensor):
        return self.dropout(self.down_proj(self.act_fn(self.up_proj(x)) * self.gate_proj(x)))
    
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
        hidden_states, present_key_value = self.self_attn(self.input_layernorm(hidden_states), postion_embedding, past_key_value, use_cache, attention_mask)

        hidden_states = residual + hidden_states
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))

        return hidden_states, present_key_value
    
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

        # Rope预计算
        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim = config.hidden_size // config.num_attention_heads,
            end = config.max_position_embeddings,
            rope_base = config.rope_theta,
            rope_scaling = config.rope_scaling,
        )

        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self, input_ids:Optional[torch.Tensor]=None, attention_mask:Optional[torch.Tensor]=None, past_key_values:Optional[Tuple[Tuple[torch.Tensor]]]=None, use_cache=False, **kwargs):
        batch_size, seq_len = input_ids.shape

        if hasattr(past_key_values, 'layers'):
            past_key_values = None
        
        past_key_values = past_key_values or [None] * len(self.layers)

        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        hidden_states = self.dropout(self.embed_tokens(input_ids))

        position_embedding = (
            self.freqs_cos[start_pos : start_pos + seq_len],
            self.freqs_sin[start_pos : start_pos + seq_len],
        )

        presents = []

        for layer_idx, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            hidden_states, present_key_value = layer(hidden_states, position_embedding, past_key_value, use_cache, attention_mask)
            presents.append(present_key_value)

        hidden_states = self.norm(hidden_states)

        return hidden_states, presents