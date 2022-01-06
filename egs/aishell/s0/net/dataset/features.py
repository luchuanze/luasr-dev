
import logging

import numpy as np
import torchaudio


def extract(batch, feature_conf):

    """
    Extract fbank feature from waveform

    :param batch:
    :param feature_conf:
    :return: (keys, feats, tokens)
    """

    keys = []
    feats = []
    tokens = []
    frames = []

    for i, x in enumerate(batch):
        try:
            wav_path = x[1]
            waveform, sample_rate = torchaudio.load(wav_path)
            waveform = waveform * (1 << 15)
            fb = torchaudio.compliance.kaldi.fbank(
                waveform,
                num_mel_bins=feature_conf['num_mel_bins'],
                frame_length=feature_conf['frame_length'],
                frame_shift=feature_conf['frame_shift'],
                dither=feature_conf['dither'],
                energy_floor=0.0,
                sample_frequency=sample_rate
            )

            fb = fb.detach().numpy()
            feats.append(fb)
            keys.append(x[0])
            tokens.append(x[2])
            frames.append(fb.shape[0])

        except(Exception) as e:
            print(e)
            logging.warning('read utterance {} error'.format(x[0]))
            pass

    # sort it by frame size for pack/pad operation
    order = np.argsort(frames)[::-1]
    stored_keys = [keys[i] for i in order]
    stored_feats = [feats[i] for i in order]
    stored_tokens = [np.array(tokens[i], dtype=np.int32) for i in order]

    return stored_keys, stored_feats, stored_tokens





