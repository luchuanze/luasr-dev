
from typing import Optional, Tuple

import torch


class TransformerEncoderLayer(torch.nn.Module):
    def __init__(self,
                 size: int,
                 self_attn: torch.nn.Module,
                 feed_forward: torch.nn.Module,
                 dropout_rate: float,
                 normalize_before: bool = True,
                 concat_after: bool = False
                 ):

        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.norm1 = torch.nn.LayerNorm(size, eps=1e-12)
        self.norm2 = torch.nn.LayerNorm(size, eps=1e-12)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.size = size
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        self.concat_linear = torch.nn.Linear(size + size, size)

    def forward(self,
                x: torch.Tensor,
                mask: torch.Tensor,
                pos_emb: torch.Tensor,
                mask_pad: Optional[torch.Tensor] = None,
                output_cache: Optional[torch.Tensor] = None,
                cnn_cache: Optional[torch.Tensor] = None,
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        residual = x
        if self.normalize_before:
            x = self.norm1(x)

        if output_cache is None:
            x_q = x
        else:
            assert output_cache.size(0) == x.size(0)
            assert output_cache.size(0) == self.size
            assert output_cache.size(0) < x.size(1)

        if self.concat_after:
            x_concat = torch.cat((x, self.self_attn(x_q, x, x, mask)), dim=-1)
            x = residual + self.concat_linear(x_concat)
        else:
            x = residual + self.dropout(self.self_attn(x_q, x, x, mask))

        if not self.normalize_before:
            x = self.norm1(x)

        residual = x

        if self.normalize_before:
            x = self.norm2(x)
        x = residual + self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm2(x)

        if output_cache is not None:
            x = torch.cat([output_cache, x], dim=1)

        fake_cnn_cache = torch.tensor([0.0], dtype=x.dtype, device=x.device)

        return x, mask, fake_cnn_cache


class ConformerEncoderLayer(torch.nn.Module):

    def __init__(self,
                 size: int,
                 self_attn: torch.nn.Module,
                 feed_forward: Optional[torch.nn.Module] = None,
                 feed_forward_macaron: Optional[torch.nn.Module] = None,
                 conv_module: Optional[torch.nn.Module] = None,
                 dropout_rate: float = 0.1,
                 normalize_before: bool = True,
                 concat_after: bool = False):
        super(ConformerEncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.feed_forward_macaron = feed_forward_macaron
        self.conv_module = conv_module
        self.norm_ff = torch.nn.LayerNorm(size, eps=1e-12)
        self.norm_mha = torch.nn.LayerNorm(size, eps=1e-12)
        if feed_forward_macaron is not None:
            self.norm_ff_macaron = torch.nn.LayerNorm(size, eps=1e-12)
            self.ff_scale = 0.5
        else:
            self.ff_scale = 1.0

        if self.conv_module is not None:
            self.norm_conv = torch.nn.LayerNorm(size, eps=1e-12)
            self.norm_final = torch.nn.LayerNorm(size, eps=1e-12)

        self.dropout = torch.nn.Dropout(dropout_rate)
        self.size = size
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        self.concat_linear = torch.nn.Linear(size + size, size)

    def forward(self,
                x: torch.Tensor,
                mask: torch.Tensor,
                pos_emb: torch.Tensor,
                mask_pad: Optional[torch.Tensor] = None,
                output_cache: Optional[torch.Tensor] = None,
                cnn_cache: Optional[torch.Tensor] = None,
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # whether to use macaron style
        if self.feed_forward_macaron is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_ff_macaron(x)
            x = residual + self.ff_scale * self.dropout(self.feed_forward_macaron(x))
            if not self.normalize_before:
                self.feed_forward_macaron(x)

        # multi-headed self-attention module
        residual = x
        if self.normalize_before:
            x = self.norm_mha(x)

        if output_cache is None:
            x_q = x
        else:
            assert output_cache.size(0) == x.size(0)
            assert output_cache.size(1) < x.size(1)
            assert output_cache.size(2) == self.size
            chunk = x.size(1) - output_cache.size(1)
            x_q = x[:, -chunk:, :]
            residual = residual[:, -chunk:, :]
            mask = mask[:, -chunk:, :]

        x_att = self.self_attn(x_q, x, x, mask, pos_emb)
        if self.concat_after:
            x_concat = torch.cat((x, x_att), dim=-1)
            x = residual + self.concat_linear(x_concat)
        else:
            x = residual + self.dropout(x_att)

        if not self.normalize_before:
            x = self.norm_mha(x)

        # convolution module
        new_cnn_cache = torch.tensor([0.0], dtype=x.dtype, device=x.device)
        if self.conv_module is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_conv(x)
                x, new_cnn_cache = self.conv_module(x, mask_pad, cnn_cache)
                x = residual + self.dropout(x)

                if not self.normalize_before:
                    x = self.norm_conv(x)

        # feed forward module
        residual = x
        if self.normalize_before:
            x = self.norm_ff(x)
        x = residual + self.ff_scale * self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm_ff(x)

        if self.conv_module is not None:
            x = self.norm_final(x)

        if output_cache is not None:
            x = torch.cat([output_cache, x], dim=1)

        return x, mask, new_cnn_cache







