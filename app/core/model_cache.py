"""
모델 정의: GPT-2 (공용) + KV Cache 지원
03_pretrain.py, 04_finetune.py, 05_generate.py에서 임포트

KV Cache:
  - 학습 시: 기존과 완전히 동일 (cache 사용 안 함)
  - 생성 시: past_key_values를 넘기면 새 토큰만 계산 → 10~20배 빠름
  - 수학적으로 동일한 결과 (품질 차이 없음)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from anticipation.vocab import VOCAB_SIZE


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))

    def forward(self, x, past_kv=None):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hd)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # KV cache: 이전 key/value와 합치기
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)  # (B, nh, past_T + T, hd)
            v = torch.cat([past_v, v], dim=2)

        present_kv = (k, v)  # 다음 step을 위해 저장

        total_T = k.size(2)  # past + current 전체 길이

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        # causal mask: query 위치 기준으로 적용
        att = att.masked_fill(self.bias[:, :, total_T - T:total_T, :total_T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y, present_kv


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x, past_kv=None):
        attn_out, present_kv = self.attn(self.ln_1(x), past_kv=past_kv)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, present_kv


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        n_params = sum(p.numel() for p in self.parameters())
        print(f"모델 파라미터: {n_params / 1e6:.1f}M")

    def forward(self, idx, targets=None, past_key_values=None):
        """
        Args:
            idx: (B, T) 토큰 인덱스
            targets: (B, T) 학습 시 타겟 (생성 시 None)
            past_key_values: list of (key, value) per layer (생성 시 사용)

        Returns:
            logits: (B, T, vocab_size)
            loss: scalar or None
            present_key_values: list of (key, value) per layer (생성 시 반환)
        """
        B, T = idx.size()

        # position offset: cache가 있으면 이미 처리된 길이만큼 밀어줌
        if past_key_values is not None and past_key_values[0] is not None:
            past_len = past_key_values[0][0].size(2)  # 캐시된 시퀀스 길이
        else:
            past_len = 0

        assert past_len + T <= self.config.block_size, \
            f"시퀀스 길이 {past_len + T} > block_size {self.config.block_size}"

        pos = torch.arange(past_len, past_len + T, dtype=torch.long, device=idx.device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        # 각 layer 통과하면서 KV cache 수집
        present_key_values = []
        for i, block in enumerate(self.transformer.h):
            past_kv = past_key_values[i] if past_key_values is not None else None
            x, present_kv = block(x, past_kv=past_kv)
            present_key_values.append(present_kv)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        # 학습 시 (targets 있음): 기존처럼 2개 반환 → 호환성 유지
        # 생성 시 (past_key_values 사용): 3개 반환 → KV cache 포함
        if past_key_values is not None:
            return logits, loss, present_key_values
        else:
            return logits, loss


class ModelConfig:
    """기본 모델 설정"""
    vocab_size = VOCAB_SIZE
    n_layer = 12
    n_head = 12
    n_embd = 768
    block_size = 1024
    dropout = 0.1