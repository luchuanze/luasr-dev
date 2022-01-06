
import torch
import math
from typing import Tuple, List, Optional
from ctcaed.block import TransformerEncoderLayer
from ctcaed.attention import MultiHeadAttention
from ctcaed.forwardfeed import PositionwiseFeedForward


def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """Make mask tensor containing indices of padded part.

    See description of make_non_pad_mask.

    Args:
        lengths (torch.Tensor): Batch of lengths (B,).
    Returns:
        torch.Tensor: Mask tensor containing indices of padded part.

    Examples:
        >>> lengths = [5, 3, 2]
        >>> make_pad_mask(lengths)
        masks = [[0, 0, 0, 0 ,0],
                 [0, 0, 0, 1, 1],
                 [0, 0, 1, 1, 1]]
                 :param lengths:
                 :param max_len:
    """
    batch_size = lengths.size(0)
    max_len = max_len if max_len > 0 else lengths.max().item()
    seq_range = torch.arange(0,
                             max_len,
                             dtype=torch.int64,
                             device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = lengths.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    return mask

def subsequnet_chunk_mask(
        size: int,
        chunk_size: int,
        num_left_chunks: int = -1,
        device: torch.device = torch.device("cpu")
) -> torch.Tensor:

    ret = torch.zeros(size, size, device=device, dtype=torch.bool)

    for i in range(size):
        if num_left_chunks < 0:
            start = 0
        else:
            start = max((i // chunk_size - num_left_chunks) * chunk_size, 0)
        end = min((i // chunk_size + 1) * chunk_size, size)
        ret[i, start:end] = True

    return ret

def add_optional_chunk_mask(x: torch.Tensor, masks: torch.Tensor,
                            use_dynamic_chunk: bool,
                            use_dynamic_lef_chuck: bool,
                            decoding_chunk_size: int,
                            static_chunk_size: int,
                            num_decoding_left_chunks: int):

    if use_dynamic_lef_chuck:
        max_len = x.size(1)
        if decoding_chunk_size < 0:
            chunk_size = max_len
            num_left_chunks = -1
        elif decoding_chunk_size > 0:
            chunk_size = decoding_chunk_size
            num_left_chunks = num_decoding_left_chunks
        else:
            chunk_size = torch.randint(1, max_len, (1, )).item()
            num_left_chunks = -1
            if chunk_size > max_len // 2:
                chunk_size = max_len
            else:
                chunk_size = chunk_size % 25 + 1
                if use_dynamic_lef_chuck:
                    max_left_chunks = (max_len-1) // chunk_size
                    num_left_chunks = torch.randint(0, max_left_chunks, (1, )).item()
        chunk_masks = subsequnet_chunk_mask(x.size(1), chunk_size, num_left_chunks, x.device)
        chunk_masks = chunk_masks.unsqueeze(0)
        chunk_masks = masks & chunk_masks

    elif static_chunk_size > 0:
        num_left_chunks = num_decoding_left_chunks
        chunk_masks = subsequnet_chunk_mask(x.size(0), static_chunk_size, num_left_chunks, x.device)
        chunk_masks = chunk_masks.unqueeze(0)
        chunk_masks = masks & chunk_masks
    else:
        chunk_masks = masks

    return chunk_masks


class SubsamplingConv(torch.nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.subsampling_rate = 1

        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, out_dim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(out_dim, out_dim, 3, 2),
            torch.nn.ReLU()
        )

        self.out = torch.nn.Sequential(
            torch.nn.Linear(out_dim * (((in_dim - 1) // 2 - 1) // 2), out_dim)
        )

        self.subsampling_rate = 4

    def forward(self,
                x: torch.Tensor,
                x_mask: torch.Tensor,
                offset: int = 0
                ):
        x = x.unsqueeze(1)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        return x, x_mask[:, :, :-2:2][:, :, :-2:2]


class PositionalEncoding(torch.nn.Module):
    """Positional encoding.

    :param int d_model: embedding dim
    :param float dropout_rate: dropout rate
    :param int max_len: maximum input length

    PE(pos, 2i)   = sin(pos/(10000^(2i/dmodel)))
    PE(pos, 2i+1) = cos(pos/(10000^(2i/dmodel)))
    """
    def __init__(self,
                 d_model: int,
                 dropout_rate: float,
                 max_len: int = 5000,
                 reverse: bool = False):
        """Construct an PositionalEncoding object."""
        super().__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(self.d_model)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.max_len = max_len

        self.pe = torch.zeros(self.max_len, self.d_model)
        position = torch.arange(0, self.max_len,
                                dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32) *
            -(math.log(10000.0) / self.d_model))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0)

    def forward(self,
                x: torch.Tensor,
                offset: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add positional encoding.

        Args:
            x (torch.Tensor): Input. Its shape is (batch, time, ...)
            offset (int): position offset

        Returns:
            torch.Tensor: Encoded tensor. Its shape is (batch, time, ...)
            torch.Tensor: for compatibility to RelPositionalEncoding
        """
        assert offset + x.size(1) < self.max_len
        self.pe = self.pe.to(x.device)
        pos_emb = self.pe[:, offset:offset + x.size(1)]
        x = x * self.xscale + pos_emb
        return self.dropout(x), self.dropout(pos_emb)

    def position_encoding(self, offset: int, size: int) -> torch.Tensor:
        """ For getting encoding in a streaming fashion

        Attention!!!!!
        we apply dropout only once at the whole utterance level in a none
        streaming way, but will call this function several times with
        increasing input size in a streaming scenario, so the dropout will
        be applied several times.

        Args:
            offset (int): start offset
            size (int): requried size of position encoding

        Returns:
            torch.Tensor: Corresponding encoding
        """
        assert offset + size < self.max_len
        return self.dropout(self.pe[:, offset:offset + size])


class RelPositionalEncoding(PositionalEncoding):
    """Relative positional encoding module.
    See : Appendix B in https://arxiv.org/abs/1901.02860
    Args:
        d_model (int): Embedding dimension.
        dropout_rate (float): Dropout rate.
        max_len (int): Maximum input length.
    """
    def __init__(self, d_model: int, dropout_rate: float, max_len: int = 5000):
        """Initialize class."""
        super().__init__(d_model, dropout_rate, max_len, reverse=True)

    def forward(self,
                x: torch.Tensor,
                offset: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute positional encoding.
        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).
        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).
            torch.Tensor: Positional embedding tensor (1, time, `*`).
        """
        assert offset + x.size(1) < self.max_len
        self.pe = self.pe.to(x.device)
        x = x * self.xscale
        pos_emb = self.pe[:, offset:offset + x.size(1)]
        return self.dropout(x), self.dropout(pos_emb)


class BaseEncoder(torch.nn.Module):
    def __init__(self,
                 input_size: int,
                 output_size: int = 256,
                 attention_heads: int = 4,
                 linear_units: int = 2048,
                 num_blocks: int = 6,
                 dropout_rate: float = 0.1,
                 positional_dropout_rate: float = 0.1,
                 attention_dropout_rate: float = 0.0,
                 input_layer: str = "conv2d",
                 pos_enc_layer_type: str = "abs_pos",
                 normalize_before: bool = True,
                 concat_after: bool = False,
                 static_chunk_size: int = 0,
                 use_dynamic_chuck: bool = False,
                 global_cmvn: torch.nn.Module = None,
                 use_dynamic_left_chunk: bool = False):
        super().__init__()
        self.output_size = output_size
        self.global_cmvn = global_cmvn

        self.subsampling_conv = SubsamplingConv(input_size, output_size)

        if pos_enc_layer_type == 'abs_pos':
            pos_enc = PositionalEncoding
        else:
            pos_enc = RelPositionalEncoding

        self.embed = pos_enc(output_size, positional_dropout_rate)

        self.use_dynamic_chunk = use_dynamic_chuck
        self.use_dynamic_left_chunk = use_dynamic_left_chunk
        self.static_chunk_size = static_chunk_size
        self.normalize_before = normalize_before
        self.after_norm = torch.nn.LayerNorm(output_size, eps=1e-12)

    def forward(self,
                x: torch.Tensor,
                x_lens: torch.Tensor,
                decoding_chunk_size: int,
                num_decoding_left_chunks: int = -1):
        T = x.size(1)
        mask = ~make_pad_mask(x_lens, T).unsqueeze(1)
        x, mask = self.subsampling_conv(x, mask)

        x, pos_emb = self.embed(x)

        mask_pad = mask

        chunk_mask = add_optional_chunk_mask(x,
                                             mask,
                                             self.use_dynamic_chunk,
                                             self.use_dynamic_left_chunk,
                                             decoding_chunk_size,
                                             self.static_chunk_size,
                                             num_decoding_left_chunks)

        for block in self.encoders:
            x, chunk_mask, _ = block(x, chunk_mask, pos_emb, mask_pad)
        if self.normalize_before:
            xs = self.after_norm(x)

        return x, mask


class TransformerEncoder(BaseEncoder):
    def __init__(self,
                 input_size: int,
                 output_size: int = 256,
                 attention_heads: int = 4,
                 linear_units: int = 2048,
                 num_blocks: int = 6,
                 dropout_rate: float = 0.1,
                 positional_dropout_rate: float = 0.1,
                 attention_dropout_rate: float = 0.0,
                 input_layer: str = "conv2d",
                 pos_enc_layer_type: str = "abs_pos",
                 normalize_before: bool = True,
                 concat_after: bool = False,
                 static_chunk_size: int = 0,
                 use_dynamic_chunk: bool = False,
                 global_cmvn: torch.nn.Module = None,
                 use_dynamic_left_chunk: bool = False):

        super().__init__(input_size, output_size, attention_heads,
                         linear_units, num_blocks, dropout_rate,
                         positional_dropout_rate, attention_dropout_rate,
                         input_layer, pos_enc_layer_type, normalize_before,
                         concat_after, static_chunk_size, use_dynamic_chunk,
                         global_cmvn, use_dynamic_left_chunk)

        self.encoders = torch.nn.ModuleList([
            TransformerEncoderLayer(
                output_size,
                MultiHeadAttention(attention_heads, output_size, attention_dropout_rate),
                PositionwiseFeedForward(output_size, linear_units, dropout_rate),
                dropout_rate,
                normalize_before,
                concat_after) for i in range(num_blocks)
        ])


class ConformerEncoder(BaseEncoder):
    def __init__(self,
                 input_size: int,
                 output_size: int = 256,
                 attention_heads: int = 4,
                 linear_units: int = 2048,
                 num_blocks: int = 6,
                 dropout_rate: float = 0.1,
                 positional_dropout_rate: float = 0.1,
                 attention_dropout_rate: float = 0.0,
                 input_layer: str = "conv2d",
                 pos_enc_layer_type: str = "abs_pos",
                 normalize_before: bool = True,
                 concat_after: bool = False,
                 static_chunk_size: int = 0,
                 use_dynamic_chunk: bool = False,
                 global_cmvn: torch.nn.Module = None,
                 use_dynamic_left_chunk: bool = False,
                 macaron_style: bool = True,
                 activation_type: bool = True,
                 use_cnn_module: bool = True,
                 cnn_module_kernel: int = 15,
                 causal: bool = False,
                 cnn_module_norm: str = 'batch_norm'):

        super().__init__(input_size, output_size, attention_heads,
                         linear_units, num_blocks, dropout_rate,
                         positional_dropout_rate, attention_dropout_rate,
                         input_layer, pos_enc_layer_type, normalize_before,
                         concat_after, static_chunk_size, use_dynamic_chunk,
                         global_cmvn, use_dynamic_left_chunk)

        if pos_enc_layer_type == "no_pos":
            selfattn_layer = MultiHeadAttention
        else:
            selfattn_layer =
