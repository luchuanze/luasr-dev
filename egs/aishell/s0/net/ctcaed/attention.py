import math

import torch
from typing import Optional, Tuple


class MultiHeadAttention(torch.nn.Module):

    def __init__(self,
                 input_dim: int,
                 num_head: int = 4,
                 dropout_rate: float = 0.01):
        super().__init__()
        assert input_dim % num_head == 0

        self.d_k = input_dim // num_head
        self.h = num_head
        self.linear_q = torch.nn.Linear(input_dim, input_dim)
        self.linear_k = torch.nn.Linear(input_dim, input_dim)
        self.linear_v = torch.nn.Linear(input_dim, input_dim)
        self.linear_out = torch.nn.Linear(input_dim, input_dim)
        self.dropout = torch.nn.Linear(p=dropout_rate)

    def forward_qkv(self,
                    query: torch.Tensor,
                    key: torch.Tensor,
                    value: torch.Tensor,
                    )-> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        batch_size = query.size(0)
        q = self.linear_q(query).view(batch_size, -1, self.h, self.d_k)
        k = self.linear_k(key).view(batch_size, -1, self.h, self.d_k)
        v = self.linear_v(value).view(batch_size, -1, self.h, self.d_k)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        return q, k, v

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor],
                pos_emb: torch.Tensor = torch.empty(0)
                )-> torch.Tensor:

        batch_size = value.size(0)
        q, k, v = self.forward_qkv(query, key, value)
        score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)
            score = score.masked_fill(mask, -float('inf'))
            attention = torch.softmax(score, dim=-1).masked_fill(mask, 0.0)
        else:
            attention = torch.softmax(score, dim=-1)

        attention = self.dropout(attention)
        x = torch.matmul(attention, v)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h, self.d_k)

        return self.linear_out(x)


class RelPosition


