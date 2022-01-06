
import torch
from ctcaed.encoder import BaseEncoder

IGNORE_ID = -1


class CtcAedModel(torch.nn.Module):
    def __init__(self,
                 vocab_size,
                 encoder,
                 # decoder,
                 # ctc,
                 ctc_weight: float = 0.5,
                 ignore_id: int = IGNORE_ID):

        assert (0.0 <= ctc_weight <= 1.0)

        super().__init__()
        # eos id is then same as sos
        self.sos = vocab_size - 1
        self.eos = vocab_size - 1
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.ctc_weight = ctc_weight

        self.encoder = encoder
        # self.decoder = decoder
        # self.ctc = ctc

    def forward(self,
                speech: torch.Tensor,
                speech_lengths: torch.Tensor,
                token: torch.Tensor,
                token_lengths: torch.Tensor):

        assert token_lengths.dim() == 1
        assert speech.shape[0] == speech_lengths.shape[0] == token.shape[0] == token_lengths.shape[0]

        encoder_out = self.encoder(speech, speech_lengths)

        return encoder_out


def create_model(configs):

    input_dim = configs['input_dim']
    vocab_size = configs['output_dim']

    encoder = BaseEncoder(input_size=input_dim)
    return CtcAedModel(vocab_size=vocab_size, encoder=encoder)


